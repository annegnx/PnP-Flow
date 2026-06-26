import torch
import torch
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils


class MMSE_AVERAGE(object):

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

    def mmse(self, x, t, num_samples=5):
        # TODO: parametrize num_samples
        out = torch.zeros_like(x)
        tt = t.view(-1, 1, 1, 1)
        for _ in range(num_samples):
            x_tilde = tt * x + (1 - tt) * torch.randn_like(x)
            out += self.denoiser(x_tilde, t)
        return out / num_samples

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj
        self.args.sigma_noise = sigma_noise

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
            # x = torch.randn_like(clean_img).to(self.device)

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

                    # print(f'{k}, {sigma_noise:.6f}, {lmbda * tau:.6f}')
                    sigma_k = np.sqrt(sigma_noise ** 2 / (k + 1))
                    alpha_k = 1 / (k + 2)
                    t_k = 1 / (1 + sigma_k)
                    t1 = torch.ones(len(x), device=self.device) * t_k

                    x = (1 - alpha_k) * self.mmse(x, t1) + alpha_k * x

                    if self.args.save_results:
                        pass
                        # restored_img = x.detach().clone()
                        # if self.should_save_image(k, max_iter):
                        #     utils.save_images(clean_img, noisy_img, restored_img,
                        #                 self.args, H_adj, iter=k)
                        # utils.compute_psnr(clean_img, noisy_img,
                        #                         restored_img, self.args, H_adj, iter=k)
                        # utils.compute_ssim(
                        #             clean_img, noisy_img, restored_img, self.args, H_adj, iter=k)
                        # utils.compute_lpips(clean_img, noisy_img,
                        #                             restored_img, self.args, H_adj, iter=k)

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
