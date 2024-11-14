import torch
import numpy as np
import torch
import numpy as np
import utils as utils
from matplotlib import pyplot as plt
import os
from degradations import Superresolution


class PROX_PNP(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.model = model
        self.method = args.method

    def model_forward(self, x):
        sigma = torch.ones(len(x), device=self.device) * self.args.sigma_noise
        if self.args.model == "gradient_step":
            return self.model(x, sigma)

    def grad_datafit(self, x, y, H, H_adj):
        if self.args.noise_type == 'gaussian':
            return H_adj(H(x) - y) / (self.args.sigma_noise**2)
        elif self.args.noise_type == 'laplace':
            return H_adj(2*torch.heaviside(H(x)-y, torch.zeros_like(H(x)))-1)/self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')

    def prox_datafit(self, x, y, H, H_adj, degradation=None, alpha=None):
        if self.args.noise_type == 'gaussian' and self.args.problem == "random_inpainting":
            return H(y) - H(x) + x
        if self.args.noise_type == 'gaussian' and self.args.problem == "gaussian_deblurring_FFT":
            fft_d = torch.fft.fft2(alpha * H_adj(y) + x)
            kernel = degradation.filter
            kernel_size = kernel.shape[2]
            kernel_id = torch.zeros_like(kernel)
            kernel_id[:, :, kernel_size//2, kernel_size//2] = 1
            fft_kernel = torch.fft.fft2(kernel)
            inv = alpha * torch.conj(fft_kernel) * fft_kernel + 1.
            sol = torch.real(torch.fft.ifft2(fft_d / inv))
            return sol
        if self.args.noise_type == 'gaussian' and self.args.problem == "superresolution_bicubic":
            if x.shape[-1] == 128:
                sf = 2
            else:
                sf = 4

            def splits(a, sf):
                """split a into sfxsf distinct blocks
                Args:
                    a: NxCxWxH
                    sf: split factor
                Returns:
                    b: NxCx(W/sf)x(H/sf)x(sf^2)
                """
                b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
                b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
                return b

            hat_z = H_adj(y) + x / alpha
            fft_hat_z = torch.fft.fft2(hat_z)
            kernel = degradation.filter
            kernel_size = kernel.shape[2]
            kernel_id = torch.zeros_like(kernel)
            kernel_id[:, :, kernel_size//2, kernel_size//2] = 1
            fft_kernel = torch.fft.fft2(kernel)
            top = torch.mean(splits(fft_kernel * fft_hat_z, sf), dim=-1)
            below = torch.mean(
                splits(torch.conj(fft_kernel) * fft_kernel * fft_hat_z, sf), dim=-1) + 1 / alpha
            rc = torch.conj(fft_kernel) * (top / below).repeat(1, 1, sf, sf)

            sol = torch.real(torch.fft.ifft2(rc))
            return (hat_z-sol) * alpha

    def objective(self, x, y, H, H_adj, lmbda,  g):
        if self.args.noise_type == 'gaussian':
            datafit = 0.5 * torch.linalg.norm(H(x)-y)**2
            return datafit + lmbda * g
        elif self.args.noise_type == 'laplace':
            datafit = torch.mean(torch.abs(H(x)-y))
            return datafit + lmbda * g

    def solve_ip(self, test_loader, degradation, sigma_noise):
        H = degradation.H
        H_adj = degradation.H_adj
        self.args.sigma_noise = sigma_noise
        self.args.lr_pnp = sigma_noise**2 * self.args.lr_pnp
        lr = self.args.lr_pnp
        sigma_factor = self.args.sigma_factor

        max_batch = self.args.max_batch
        max_iter = self.args.max_iter
        alpha = self.args.alpha

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch

            noisy_img = H(clean_img.clone().to(self.device))
            if self.args.noise_type == 'gaussian':
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise
            elif self.args.noise_type == 'laplace':
                noise = torch.distributions.laplace.Laplace(
                    torch.zeros_like(noisy_img), sigma_noise * torch.ones_like(noisy_img)).sample().to(self.device)
                noisy_img += noise
            else:
                raise ValueError('Noise type not supported')

            noisy_img, clean_img = noisy_img.to(
                self.device), clean_img.to('cpu')

            # intialize the image with the adjoint operator

            if self.args.problem == "random_inpainting":
                x = 1.5 * noisy_img.clone() - H(noisy_img)
            elif self.args.problem == "superresolution":
                if clean_img.shape[-1] == 128:
                    sf = 2
                else:
                    sf = 4
                superres_bicubic = Superresolution(
                    sf, clean_img.shape[-1], mode="bicubic", device=self.device)
                x = superres_bicubic.H_adj(noisy_img.clone())
            else:
                x = H_adj(noisy_img.clone()).to(self.device)

            with torch.no_grad():
                for count, iteration in enumerate(range(int(max_iter))):
                    list_dist = []

                    x_old = x

                    if self.args.algo == "hqs" and self.args.problem == "random_inpainting":
                        # Douglas Rachford

                        sigma_ = (
                            sigma_noise * torch.ones(x.shape[0], 1, 1, 1, device=self.device)).squeeze()
                        if iteration < 20:
                            sigma_ = (
                                0.2 * torch.ones(x.shape[0], 1, 1, 1, device=self.device)).squeeze()
                        torch.set_grad_enabled(True)
                        Dg, N = self.model.calculate_grad(
                            x_old, sigma_)
                        torch.set_grad_enabled(False)
                        Dx = x_old - Dg
                        y = Dx

                        if iteration < max_iter - 1:
                            # Proximal step
                            z = self.prox_datafit(y, noisy_img, H, H_adj)
                            x = z

                    elif self.args.algo == "hqs" and self.args.problem == "gaussian_deblurring_FFT":
                        sigma_ = (
                            1.8 * sigma_noise * torch.ones(len(x), device=self.device)).squeeze()

                        torch.set_grad_enabled(True)
                        Dg, N, g = self.model.calculate_grad(
                            x_old, sigma_, compute_g=True)
                        torch.set_grad_enabled(False)
                        Dx = x_old - Dg
                        y = 0.1 * alpha * Dx + alpha * \
                            (1 - alpha * 0.1) * x_old

                        # Proximal step
                        z = self.prox_datafit(
                            y, noisy_img, H, H_adj, degradation, alpha)
                        x = z
                        gap_objective = self.objective(
                            x, noisy_img, H, H_adj, 0.1,  g)-self.objective(x_old, noisy_img, H, H_adj, 0.1,  g)
                        if gap_objective < 0.1 / alpha * torch.linalg.norm(x-x_old)**2:
                            print(gap_objective)
                            alpha = 0.9 * alpha

                    elif self.args.algo == "hqs" and self.args.problem == "superresolution_bicubic":

                        sigma_ = (
                            2.0 * sigma_noise * torch.ones(len(x), device=self.device)).squeeze()

                        torch.set_grad_enabled(True)
                        Dg, N, g = self.model.calculate_grad(
                            x_old, sigma_, compute_g=True)
                        torch.set_grad_enabled(False)
                        Dx = x_old - Dg
                        y = 0.065 * alpha * Dx + alpha * \
                            (1 - alpha * 0.065) * x_old

                        # Proximal step
                        z = self.prox_datafit(
                            y, noisy_img, H, H_adj, degradation, alpha)
                        x = z
                        gap_objective = self.objective(
                            x, noisy_img, H, H_adj, 0.065,  g)-self.objective(x_old, noisy_img, H, H_adj, 0.065,  g)
                        # if gap_objective < 0.1 / alpha * torch.linalg.norm(x-x_old)**2:
                        #     alpha = 0.9 * alpha

                    elif self.args.algo == "pgd":
                        # Proximal Gradient Descent
                        if self.args.problem != "denoising" or self.args.noise_type == "laplace":
                            # Gradient step
                            gradx = self.grad_datafit(
                                x_old, noisy_img, H, H_adj)
                            z = x_old - lr * gradx
                        else:
                            z = x_old

                        # Denoising step
                        sigma_ = sigma_factor * self.args.sigma_noise * \
                            torch.ones(len(x), device=self.device)
                        torch.set_grad_enabled(True)
                        Dg, N = self.model.calculate_grad(
                            z, sigma_)
                        torch.set_grad_enabled(False)

                        Dg = Dg.detach()
                        Dz = z - Dg
                        x = (1 - alpha) * z + alpha * Dz

                    if iteration % 10 == 0:

                        restored_img = x.detach().clone()
                        # utils.save_images(
                        #     clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                        utils.compute_psnr(clean_img, noisy_img,
                                           restored_img, self.args, H_adj, iter=iteration)
                        utils.compute_ssim(clean_img, noisy_img,
                                           restored_img, self.args, H_adj, iter=iteration)
                        utils.compute_lpips(clean_img, noisy_img,
                                            restored_img, self.args, H_adj, iter=iteration)

            restored_img = x.detach().clone()
            utils.save_images(clean_img, noisy_img, restored_img,
                              self.args, H_adj, iter='final')
            utils.compute_psnr(clean_img, noisy_img,
                               restored_img, self.args, H_adj, iter=iteration)
            utils.compute_ssim(clean_img, noisy_img,
                               restored_img, self.args, H_adj, iter=iteration)
            utils.compute_lpips(clean_img, noisy_img,
                                restored_img, self.args, H_adj, iter=iteration)

        utils.compute_average_psnr(self.args)
        utils.compute_average_ssim(self.args)
        utils.compute_average_lpips(self.args)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 5) == 0

    def run_method(self, data_loaders, degradation, sigma_noise):

        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise)
