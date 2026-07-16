import torch
import torch
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils

from torchdiffeq import odeint_adjoint as odeint


class GENERIC_MMSE_AVERAGE(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method

    def model_forward(self, x, t):
        if self.args.model == "ot":
            return self.model(x, t)

        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            t_ = t[:, None, None, None]
            v = model_fn(x.type(torch.float), t * 999)
            return v

    def learning_rate_strat(self, lr, t):
        t = t.view(-1, 1, 1, 1)
        gamma_styles = {
            '1_minus_t': lambda lr, t: lr * (1 - t),
            'sqrt_1_minus_t': lambda lr, t: lr * torch.sqrt(1 - t),
            'constant': lambda lr, t: lr,
            'alpha_1_minus_t': lambda lr, t: lr * (1 - t)**self.args.alpha
        }
        return gamma_styles.get(self.args.gamma_style, lambda lr, t: lr)(lr, t)

    def grad_datafit(self, x, y, H, H_adj, l=0.0, x_ref=0.0):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) + l * (x - x_ref) #/ (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1) + l*x#/self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')

    def interpolation_step(self, x, t, eps=0):
        sigma_sample = 1.0
        if self.args.interpolation_mode == 'random':
            eps = torch.randn_like(x) * sigma_sample
            return t * x + eps * (1 - t), eps
        elif self.args.interpolation_mode == 'zero':
            return t * x, 0
        elif self.args.interpolation_mode == 'fixed':
            return t * x + (1 - t) * eps, eps
        elif self.args.interpolation_mode == 'id':
            return x, 0
        else:
            raise ValueError('Interpolation mode unknown')

    def denoiser(self, x, t):
        v = self.model_forward(x, t)
        return x + (1 - t.view(-1, 1, 1, 1)) * v

    def mmse(self, x, t, eps=0):
        out = torch.zeros_like(x)
        for _ in range(self.args.num_samples):
            x_tilde, _ = self.interpolation_step(x, t.view(-1, 1, 1, 1), eps=eps)
            out += self.denoiser(x_tilde, t)
        return out / self.args.num_samples

    def compute_probability(self, x_hat):
        def hutchinson(model, x, t, n_samples=3):
            bsz = x.shape[0]
            out = torch.zeros(bsz, device=self.device)

            for _ in range(n_samples):
                u = x.detach().requires_grad_(True)
                eps = torch.randint(0, 2, u.shape, device=u.device, dtype=u.dtype) * 2 - 1

                with torch.enable_grad():
                    v = model(u, t)
                    vprod = torch.sum(v * eps)
                    grad = torch.autograd.grad(vprod, u)[0]
                out += torch.sum(grad * eps, dim=list(range(1, u.dim())))
            return out / n_samples
    
        class AugmentedDynamics(torch.nn.Module):
            def __init__(self, model, n_hutchinson_samples=3):
                super().__init__()
                self.model = model
                self.n_samples = n_hutchinson_samples

            def forward(self, t, state):
                x, logp = state
                t = torch.ones(len(x), device=x.device) * t

                with torch.set_grad_enabled(True):
                    x = x.detach().requires_grad_(True)
                    v = self.model(x, t)

                    trace_est = hutchinson(
                        self.model, x, t, n_samples=self.n_samples
                    )

                # d(logp)/dt = -tr(dv/dx)
                dlogp_dt = -trace_est

                return v.detach(), dlogp_dt.detach()

        def compute_log_likelihood(model, x1, n_hutchinson_samples=3):
            B = x1.shape[0]
            D = x1[0].numel()

            dynamics = AugmentedDynamics(model, n_hutchinson_samples)

            logp0 = torch.zeros(B, device=x1.device, dtype=x1.dtype)
            t_span = torch.tensor([1.0, 0.0], device=x1.device)  # integrate backward

            x0, delta_logp = odeint(
                dynamics, (x1, logp0), t_span,
                method='dopri5', atol=1e-4, rtol=1e-4
            )

            x0 = x0[-1]           # state at t=0
            delta_logp = delta_logp[-1]  # accumulated -tr integral

            # log p0(x0) under standard Gaussian base
            log_p0 = -0.5 * D * torch.log(torch.tensor(2 * torch.pi)) \
                    - 0.5 * torch.sum(x0.view(B, -1) ** 2, dim=1)

            log_p1 = log_p0 + delta_logp
            return log_p1
        
        return compute_log_likelihood(self.model, x_hat)


    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj
        self.args.sigma_noise = sigma_noise
        self.num_samples = self.args.num_samples if self.args.interpolation_mode == 'random' else 1
        steps, delta = self.args.steps_pnp, 1 / self.args.steps_pnp
        tau = sigma_noise ** 2

        # if self.args.noise_type == 'gaussian':
        #     self.args.lr_pnp = sigma_noise**2 * self.args.lr_pnp
        #     lr = self.args.lr_pnp

        # elif self.args.noise_type == 'laplace':
        #     self.args.lr_pnp = sigma_noise * self.args.lr_pnp
        #     lr = self.args.lr_pnp
        # else:
        #     raise ValueError('Noise type not supported')

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):

            (clean_img, labels) = next(loader)
            self.args.batch = batch
            print(clean_img.shape)

            if self.args.noise_type == 'gaussian':
                noisy_img = H(clean_img.clone().to(self.device))
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise
            elif self.args.noise_type == 'laplace':
                noisy_img = H(clean_img.clone().to(self.device))
                noise = torch.distributions.laplace.Laplace(
                    torch.zeros_like(noisy_img), sigma_noise * torch.ones_like(noisy_img)).sample().to(self.device)
                noisy_img += noise
            else:
                raise ValueError('Noise type not supported')

            noisy_img, clean_img = noisy_img.to(
                self.device), clean_img.to('cpu')

            # intialize the image with the adjoint operator
            # x = H_adj(noisy_img).to(self.device)
            x = torch.randn_like(clean_img).to(self.device)

            # y_dagger = H_adj(noisy_img).to(self.device)
            n = torch.randn_like(clean_img).to(self.device)
            y_dagger = n

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            with torch.no_grad():
                for count, iteration in enumerate(range(1, int(steps))):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()

                    lmbda_0 = self.args.lmbda_0
                    p = self.args.pexp
                    gamma = 1.0

                    sigma = ((steps - iteration) / iteration) ** (1.0)
                    b_1 = self.args.b_1 #1.0
                    b_2 = self.args.b_2 #1.0
                    # f_ = (sigma ** 2)/b_1 * np.exp(b_2 * sigma ** (-2))
                    f_ = b_1 * sigma ** (2 * b_2)
                    alpha = sigma ** 2 / (tau + sigma ** 2 + f_)
                    lmbda = lmbda_0 * sigma ** gamma
                    t = 1 / (1 + sigma)

                    print(f'{t:.6f}, {sigma:.6f}, {alpha:.6f}')

                    t1 = torch.ones(len(x), device=self.device) * t

                    # Generic-MMSE-Average
                    grad_x = self.grad_datafit(x, noisy_img, H, H_adj, l=lmbda, x_ref=y_dagger)
                    mmse = self.mmse(x, t1)
                    x = alpha * (x - grad_x) + (1 - alpha) * mmse

                    p_x = np.exp(self.compute_probability(x).detach().cpu().numpy())
                    print(p_x)

                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1

                    if True and self.args.save_results: #!! change this after hyperparam search
                        restored_img = x.detach().clone()
                        utils.compute_psnr(clean_img, noisy_img,
                                       restored_img, self.args, H_adj, iter=iteration)
                        utils.compute_ssim(
                                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                        if self.should_save_image(iteration, int(steps)):
                            utils.save_images(
                                        clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)

            if self.args.compute_memory:
                dict_memory = {}
                dict_memory["batch"] = batch
                dict_memory["max_allocated"] = torch.cuda.max_memory_allocated(
                    self.device)
                utils.save_memory_use(dict_memory, self.args)

            if self.args.compute_time:
                dict_time = {}
                dict_time["batch"] = batch
                dict_time["time_per_batch"] = time_per_batch
                utils.save_time_use(dict_time, self.args)

            if self.args.save_results:
                restored_img = x.detach().clone()
                utils.save_images(clean_img, noisy_img, restored_img,
                                  self.args, H_adj, iter='final')
                utils.compute_psnr(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=iteration)
                utils.compute_ssim(
                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                utils.compute_lpips(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=iteration)

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        return (steps >= 10) and iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):

        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)
