import torch
import torch
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils


class MAP_ESTIMATION(object):

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
            'alpha_1_minus_t': lambda lr, t: lr * (1 - t)**self.args.alpha,
        }
        return gamma_styles.get(self.args.gamma_style, lambda lr, t: lr)(lr, t)

    def inner_steps(self, k, eta):
        if self.args.base_steps_pnp > 1:
            return int(self.args.base_steps_pnp * (k + 1) ** (1.0 + eta))
        else:
            return self.args.base_steps_pnp
    
    def get_sigma_schedule(self, k):
        c = self.args.sigma_noise ** 2 / 2
        alpha = self.args.alpha
        print(c * ((1 + 1 / (k + 1) ** alpha) - 1))
        return np.sqrt(c * ((1 + 1 / (k + 1) ** alpha) - 1))

    def grad_datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) / (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1)/self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')

    def interpolation_step(self, x, t, eps=None):
        if self.args.interpolation_mode == 'random':
            eps = torch.randn_like(x)
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

    def mmse(self, x, t):
        noise = 0 * (1 - t.view(-1, 1, 1, 1)) * torch.randn_like(x)
        return self.denoiser(t.view(-1, 1, 1, 1) * x + noise, t)

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj
        self.args.sigma_noise = sigma_noise
        tau = self.args.step_size
        eta = self.args.eta
        lmbda = self.args.lmbda

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
            # x = H_adj(torch.ones_like(noisy_img)).to(self.device)
            x = H_adj(noisy_img)

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            with torch.no_grad():
                max_iter = self.args.max_iter
                for k in range(max_iter):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()

                    #tau = self.args.step_size / ((k + 1) ** (0.1))

                    tau = sigma_noise ** 2
                    print(tau)

                    x = x - tau * lmbda * self.grad_datafit(x, noisy_img, H, H_adj) * sigma_noise ** 2

                    if k % 10 == 0: 
                        utils.save_images(clean_img, noisy_img, x.clone(),
                            self.args, H_adj, iter=f'_grad_{k}')

                    x_ref = x.clone()
                    steps = self.inner_steps(k, eta)
                    if steps > 1:
                        for iteration in range(int(steps)):
                            if self.args.compute_time:
                                time_counter_1 = perf_counter()

                            # PnP-Flow
                            # t_k = iteration / steps
                            # sigma_k = (1 - t_k) / t_k
                            # alpha_k = (1 - t_k + 1/steps) ** self.args.alpha

                            # MMSE Average
                            # sigma_k = np.sqrt(tau / (iteration + 2))
                            # alpha_k = 1 / (iteration + 3)
                            # t_k = 1 / (1 + sigma_k)

                            # New Method
                            sigma_k = self.get_sigma_schedule(iteration)
                            alpha_k = sigma_k ** 2 / (self.args.sigma_noise ** 2 + sigma_k ** 2)
                            t_k = 1 / (1 + sigma_k)

                            t1 = torch.ones(len(x), device=self.device) * t_k
                            
                            x = (1 - alpha_k) * self.mmse(x, t1) + alpha_k * x_ref
                    else:
                        sigma_k = np.sqrt(tau / (k + 2))
                        alpha_k = 1 / (k + 3)
                        t_k = 1 / (1 + sigma_k)
                        t1 = torch.ones(len(x), device=self.device) * t_k

                        x = (1 - alpha_k) * self.mmse(x, t1) + alpha_k * x_ref

                    if self.args.save_results and k % 10 == 0:
                        restored_img = x.detach().clone()
                        utils.save_images(clean_img, noisy_img, restored_img,
                                    self.args, H_adj, iter=k)
                        utils.compute_psnr(clean_img, noisy_img,
                                                restored_img, self.args, H_adj, iter=k)
                        utils.compute_ssim(
                                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=k)
                        utils.compute_lpips(clean_img, noisy_img,
                                                    restored_img, self.args, H_adj, iter=k)

                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1
            
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
                                   restored_img, self.args, H_adj, iter=k)
                utils.compute_ssim(
                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=k)
                utils.compute_lpips(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=k)

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):

        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)
