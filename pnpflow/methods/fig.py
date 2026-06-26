import torch
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils



class FIG(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.method = args.method


    def model_forward(self, x, t):
        if self.args.model == 'ot':
            return self.model(x, t)
        
        elif self.args.model == 'rectified':
            model_fn = mutils.get_model_fn(self.model, train=False)
            v = model_fn(x.type(torch.float), t * 999)
            return v

    def grad_datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) / (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1)/self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')
    
    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        torch.cuda.empty_cache()
        self.args.sigma_noise = sigma_noise
        N = self.args.N
        K = self.args.K
        eta = self.args.eta
        H = degradation.H
        H_adj = degradation.H_adj

        # TODO: add hyperparameters configs
        w = 0.0
        c = 20

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch


            if self.args.noise_type == 'gaussian':
                noisy_img = H(clean_img.clone().to(self.device))
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise
            elif self.args.noise_type == 'laplace':
                pass
        
            clean_img = clean_img.to('cpu')

            x_0 = torch.randn_like(clean_img).to(self.device)
            y_0 = H(x_0.clone())
            x = x_0
            delta_t = 1 / N 
            
            # TODO: add late initialisation with t_start < 1
            with torch.no_grad():
                for i in range(N):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()
                    t = i * delta_t * torch.ones(len(x), device=self.device).view(-1, 1, 1, 1)
                    t_next = (i + 1) * delta_t * torch.ones(len(x), device=self.device).view(-1, 1, 1, 1)

                    # TODO: add other type of probability path (here is AGPP)
                    alpha_t_next = t_next
                    sigma_t_next = 1 - t_next

                    # print(y_0.shape)
                    # print(noisy_img.shape)
                    y_next = alpha_t_next * noisy_img.clone() + w * sigma_t_next * y_0
                    x_next = x + self.model_forward(x, t.squeeze()) * delta_t
                    # print(y_next.abs().max().detach().item())

                    alpha_t = t
                    sigma_t = 1 - t
                    if i > 0:
                        for k in range(K):
                            # kappa = c * lambda_t * sigma_t * delta_t / (2 * alpha_t ** 2)
                            print("interpolant grad")
                            print(self.grad_datafit(x_next, y_next, H, H_adj).abs().max().item())
                            print("usual grad")
                            print(self.grad_datafit(x_next, noisy_img, H, H_adj).abs().max().item())
                            x_next -= c * sigma_noise ** 2 * (1 - t) / t * self.grad_datafit(x_next, y_next, H, H_adj)


                    x = x_next
                    # print(x.abs().max().detach().item())
                
                    if self.args.compute_time:
                            torch.cuda.synchronize()
                            time_counter_2 = perf_counter()
                            time_per_batch += time_counter_2 - time_counter_1

                            if self.args.save_results:
                                if i % 50 == 0 or self.should_save_image(i, N):

                                    restored_img = x.detach().clone()
                                    # utils.save_images(
                                    #     clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                                    utils.compute_psnr(clean_img, noisy_img,
                                                    restored_img, self.args, H_adj, iter=i)
                                    utils.compute_ssim(
                                        clean_img, noisy_img, restored_img, self.args, H_adj, iter=i)
                                    utils.compute_lpips(clean_img, noisy_img,
                                                        restored_img, self.args, H_adj, iter=i)
        
            if self.args.save_results:
                restored_img = x.detach().clone()
                utils.save_images(clean_img, noisy_img, restored_img,
                                  self.args, H_adj, iter='final')
                utils.compute_psnr(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=N)
                utils.compute_ssim(
                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=N)
                utils.compute_lpips(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=N)

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