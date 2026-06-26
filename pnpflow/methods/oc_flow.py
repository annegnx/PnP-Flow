import torch
from torch.nn.utils import clip_grad_norm_
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import os
from time import perf_counter
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils

#TODO: only works with 'ot' unconditional model type
class ConditionalVectorField(torch.nn.Module):
    def __init__(self, unconditional_vf, thetas):
        super().__init__()
        self.unc_vf = unconditional_vf
        self.thetas = thetas
        self.ode_steps = len(thetas)
    
    def forward(self, t, x):
        return self.unc_vf(x, t.expand(len(x))) + self.thetas[int(t * self.ode_steps)]

class RegularizationField(torch.nn.Module):
    def __init__(self, thetas):
        super().__init__()
        self.thetas = thetas
        self.ode_steps = len(thetas)
    
    def forward(self, t, x):
        return (self.thetas[int(t * self.ode_steps)] ** 2).sum([-3, -2, -1])


class OC_FLOW(object):

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

    def datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return ((H(x) - y) ** 2).sum([-3, -2, -1]) / (self.args.sigma_noise ** 2)
        elif self.args.noise_type == 'laplace':
            return ((H(x) - y) ** 2).sum([-3, -2, -1]) / self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')

    def grad_datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) / (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1)/self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')
    
    def solve_ode(self, f, x_init, timesteps):
        x = x_init
        dt = 1. / len(timesteps)
        for iteration, t in enumerate(timesteps[:-1]):
            x = x + f(x, t) * dt
        return x
    
    def loss_fn(self, x0, thetas, timesteps, y, H, H_adj):
        # regularized l2 loss
        
        # initialize
        norm2 = torch.zeros(x0.size(0), device=self.device, dtype=torch.float)
        new_x0 = x0 + thetas[-1]

        # run ode twice (one for datafit loss, one for regularization loss)
        x1 = odeint(self.conditional_vector_field, new_x0, timesteps[:-1], method='euler', adjoint_params=self.reg.parameters())[-1]
        norm2 = odeint(self.reg, norm2, timesteps[:-1], method='euler', adjoint_params=self.reg.parameters())[-1]
        
        batch_loss = self.datafit(x1, y, H, H_adj) + self.args.gamma * norm2 / 2
        
        # optional latent save
        utils.save_images(0*x1, 0*x1, x1, self.args, H_adj, iter=self.cnt)
        return batch_loss.mean()

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        torch.cuda.empty_cache()
        device = self.device
        self.args.sigma_noise = sigma_noise
        max_iter = self.args.max_iter
        ode_steps = self.args.ode_steps
        optim = self.args.optim
        max_optim_iter = self.args.max_optim_iter
        lr = self.args.optim_lr
        H = degradation.H
        H_adj = degradation.H_adj
        self.cnt = 0

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

            x = torch.randn_like(clean_img).to(self.device)
            thetas = torch.randn(ode_steps + 1, *clean_img.shape).to(self.device)
            thetas.requires_grad_(True)
            timesteps = torch.linspace(0, 1, ode_steps + 1, device=device, dtype=torch.float)

            self.conditional_vector_field = ConditionalVectorField(self.model, thetas)
            self.reg = RegularizationField(thetas)

            if optim == 'sgd':
                optimizer = torch.optim.SGD([thetas], lr=lr)
            else:
                optimizer = torch.optim.LBFGS([thetas], max_iter=max_optim_iter, lr=lr, line_search_fn='strong_wolfe')
                
            def closure():
                optimizer.zero_grad()
                loss = self.loss_fn(x, thetas, timesteps, noisy_img, H, H_adj)
                loss.backward()
                clip_grad_norm_([thetas], self.args.max_grad_norm)
            
                self.cnt += 1
                print(f'Iter {self.cnt}: Loss {loss.item():.4f}')
                return loss

            for itr in range(max_iter):
                loss = optimizer.step(closure)
                print(f'Step {itr}: Loss {loss:.4f}')

            thetas = thetas.detach()
            with torch.no_grad():
                # replace with torchdiffeq ?
                x1_opt = odeint(self.conditional_vector_field, x + thetas[-1], timesteps, method='euler')
                # x1_opt = self.solve_ode(
                #     (lambda x, t: self.model_forward(x, t.expand(len(x))) + thetas[int(t * len(timesteps))]),
                #     x + thetas[-1],
                #     timesteps
                # ).detach()


                if self.args.save_results:
                    restored_img = x1_opt.detach().clone()
                    utils.save_images(clean_img, noisy_img, restored_img,
                                    self.args, H_adj, iter='final')
                    utils.compute_psnr(clean_img, noisy_img,
                                    restored_img, self.args, H_adj, iter=max_iter)
                    utils.compute_ssim(
                        clean_img, noisy_img, restored_img, self.args, H_adj, iter=max_iter)
                    utils.compute_lpips(clean_img, noisy_img,
                                        restored_img, self.args, H_adj, iter=max_iter)

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