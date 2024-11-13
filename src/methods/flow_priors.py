import torch
import numpy as np
import utils as utils
import os
import ImageGeneration.models.utils as mutils
from time import perf_counter


class FLOW_PRIORS(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method
        self.N = args.N

    def model_forward(self, x, t):
        if self.args.model == "ot":
            return self.model(x, t)

        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            v = model_fn(x.type(torch.float), t * 999)
            return v

    def solve_ip(self, test_loader, degradation, sigma_noise):
        torch.cuda.empty_cache()
        self.args.sigma_noise = sigma_noise
        N = self.args.N
        K = self.args.K
        lmbda = self.args.lmbda
        eta = self.args.eta
        H = degradation.H
        H_adj = degradation.H_adj

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch

            if batch < 23:
                continue

            # noisy_img = H(clean_img.clone().to(self.device))
            # torch.manual_seed(batch)
            # noisy_img += torch.randn_like(noisy_img) * sigma_noise
            # noisy_img = noisy_img.to(self.device)

            if self.args.noise_type == 'gaussian':
                noisy_img = H(clean_img.clone().to(self.device))
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise
            elif self.args.noise_type == 'laplace':
                noise = torch.distributions.laplace.Laplace(torch.zeros(
                    self.args.num_channels*self.args.dim_image**2), sigma_noise*torch.ones(self.args.num_channels*self.args.dim_image**2)).sample([1]).view(self.args.num_channels, self.args.dim_image, self.args.dim_image).to(self.device)
                noisy_img = H(clean_img.clone().to(self.device)) + noise
            else:
                raise ValueError('Noise type not supported')

            clean_img = clean_img.to('cpu')

            # intialize the image with the adjoint operator
            x_init = torch.randn(clean_img.shape).to(
                self.device)

            x = x_init.clone()
            x.requires_grad_(True)

            if self.args.start_time > 0.0:
                eps = 1 * self.args.start_time
                dt = (1 - eps) / N
            else:
                # Uniform
                dt = 1./N
                eps = 1e-3  # default: 1e-3

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            for iteration in range(N):

                if self.args.compute_time:
                    time_counter_1 = perf_counter()

                num_t = iteration / N * (1 - eps) + eps
                t1 = torch.ones(len(x), device=self.device) * num_t
                t = t1.view(-1, 1, 1, 1)

                x = x.detach().clone()
                x.requires_grad = True
                optim_img = torch.optim.Adam([x], lr=eta)

                if iteration == 0:

                    for k in range(K):
                        x_next = x + self.model_forward(x, t1) * dt

                        y_next = (t + dt) * noisy_img + (1-(t+dt)) * H(x_init)
                        trace_term = utils.hut_estimator(
                            1, self.model_forward, x,  num_t)
                        if self.args.noise_type == 'gaussian':
                            loss = lmbda * torch.sum((H(x_next) - y_next) ** 2, dim=(
                                1, 2, 3))
                        elif self.args.noise_type == 'laplace':
                            loss = lmbda * torch.sum(torch.abs(H(x_next) - y_next), dim=(
                                1, 2, 3))
                        loss += 0.5 * \
                            torch.sum(x ** 2, dim=(1, 2, 3)) + trace_term * dt
                        loss = loss.sum()
                        optim_img.zero_grad()
                        grad = torch.autograd.grad(
                            loss, x, create_graph=False)[0]
                        x.grad = grad
                        optim_img.step()

                else:

                    for k in range(K):

                        pred = self.model_forward(x, t1)
                        x_next = x + pred * dt
                        y_next = (t + dt) * noisy_img + (1-(t+dt)) * H(x_init)

                        trace_term = utils.hut_estimator(
                            1,  self.model_forward, x, num_t)
                        if self.args.noise_type == 'gaussian':
                            loss = lmbda * torch.sum((H(x_next) - y_next) ** 2, dim=(
                                1, 2, 3))
                        elif self.args.noise_type == 'laplace':
                            loss = lmbda * torch.sum(torch.abs(H(x_next) - y_next), dim=(
                                1, 2, 3))
                        loss += trace_term * dt
                        loss = loss.sum()

                        optim_img.zero_grad()
                        grad = torch.autograd.grad(
                            loss, x, create_graph=False)[0]
                        grad_xt_lik = - 1 / (1-num_t) * (-x + num_t * pred)
                        x.grad = grad + grad_xt_lik
                        optim_img.step()

                x = x + self.model_forward(x, t1) * dt

                if self.args.compute_time:
                    torch.cuda.synchronize()
                    time_counter_2 = perf_counter()
                    time_per_batch += time_counter_2 - time_counter_1

                if iteration % 20 == 0 and self.args.save_results:
                    restored_img = x.detach().clone()  # / (delta * iteration)
                    # utils.save_images(
                    #         clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                    utils.compute_psnr(clean_img, noisy_img,
                                       restored_img, self.args, H_adj, iter=iteration)
                    utils.compute_ssim(clean_img, noisy_img,
                                       restored_img, self.args, H_adj, iter=iteration)
                    utils.compute_lpips(clean_img, noisy_img,
                                        restored_img, self.args, H_adj, iter=iteration)
                    del restored_img
                torch.cuda.empty_cache()

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
                utils.compute_ssim(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=iteration)
                utils.compute_lpips(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=iteration)
                del restored_img

            del noisy_img, clean_img, x
            torch.cuda.empty_cache()

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def run_method(self, data_loaders, degradation, sigma_noise):
        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        if self.args.noise_type == 'laplace':
            print('Laplace noise')
            self.args.save_path_ip = os.path.join(
                'results_laplace', self.args.save_path_ip)
        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise)
