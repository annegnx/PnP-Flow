import torch
import torch
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils


class PNP_FLOW_GRAD(object):

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

    def grad_datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) #/ (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1)#/self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')

    def interpolation_step(self, x, t, eps = 0):
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

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj
        self.args.sigma_noise = sigma_noise
        num_samples = self.args.num_samples if self.args.interpolation_mode == 'random' else 1
        steps, delta = self.args.steps_pnp, 1 / self.args.steps_pnp

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
            x = H_adj(noisy_img).to(self.device)

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            with torch.no_grad():
                for count, iteration in enumerate(range(1, int(steps))):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()

                    # if self.args.schedule == 'linear':
                    #     tau = 1.0
                    #     t = delta * iteration
                    #     sigma_t = (1 - t) / t
                    #     alpha_k = self.args.lr_pnp * (1 - t) ** self.args.alpha
                    #     beta = alpha_k ** (1.0)

                    # elif self.args.schedule == 'poly':
                    #     C = 1.0
                    #     tau = 1.0
                    #     sigma_t = np.sqrt(C / (iteration + 2))
                    #     t = 1 / (1 + sigma_t)
                    #     alpha_k = 1 / (iteration + 2)
                    #     beta = tau / C

                    t_k = delta * iteration
                    sigma_k = (1 - t_k) / t_k
                    alpha_k = (t_k / sigma_noise) ** 2 if iteration < int(steps) // 2 else (sigma_k / sigma_noise) ** 2
                    beta = 1.0

                    t_k = 1 / (1 + sigma_k)
                    t1 = torch.ones(len(x), device=self.device) * t_k
                    
                    print(f'{t_k:.6f}, {sigma_k:.6f}, {alpha_k:.6f}')

                    x_data = x.clone()
                    x_new = torch.zeros_like(x)
                    for _ in range(num_samples):
                        x_tilde, _ = self.interpolation_step(x, t1.view(-1, 1, 1, 1))
                        x_new += (1 - beta) * x + beta * self.denoiser(x_tilde, t1)

                        x_data_tilde = x_data
                        x_new += - alpha_k * self.grad_datafit(x_data_tilde, noisy_img, H, H_adj)
                    x = x_new / num_samples

                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1

                    if self.args.save_results:
                        restored_img = x.detach().clone()
                        utils.compute_psnr(clean_img, noisy_img,
                                       restored_img, self.args, H_adj, iter=iteration)
                        utils.compute_ssim(
                                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
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
