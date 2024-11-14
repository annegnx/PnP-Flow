import torch
import torch
import utils as utils
import os
import torch.nn.functional as F
import deepinv as dinv
from deepinv.physics import GaussianNoise
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.physics.forward import DecomposablePhysics
from deepinv.physics.noise import NoiseModel


class PNP_DIFF(object):

    def __init__(self, model, device, args):
        self.device = device
        self.args = args
        self.method = args.method
        self.lmbda = self.args.lmbda
        self.zeta = self.args.zeta
        self.max_iter = self.args.max_iter
        self.model = model.to(device)

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        H = degradation.H
        H_adj = degradation.H_adj

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):

            (clean_img, labels) = next(loader)
            self.args.batch = batch

            if self.args.noise_type == 'gaussian':
                physics = ForwardOperator(
                    GaussianNoise(sigma=sigma_noise), H, H_adj)
            elif self.args.noise_type == 'laplace':
                physics = ForwardOperator(
                    LaplaceNoise(sigma=sigma_noise), H, H_adj)
            else:
                raise ValueError('Noise type not supported')

            noisy_img = ((physics(clean_img) + 1) / 2).to(self.device)

            if self.args.noise_type == 'laplace':
                data_fidelity = dinv.optim.L1()
            elif self.args.problem == 'denoising':
                data_fidelity = dinv.optim.L2()
            elif self.args.problem == 'inpainting' or self.args.problem == 'random_inpainting' or self.args.problem == 'paintbrush_inpainting':
                data_fidelity = DataFidelity_Inpainting(
                    sigma_noise, H, H_adj, degradation)
            elif self.args.problem == 'gaussian_deblurring_FFT':
                data_fidelity = DataFidelity_GaussianDeblurring(
                    sigma_noise, H, H_adj, degradation)
            elif self.args.problem == 'superresolution':
                data_fidelity = DataFidelity_SuperResolution(
                    sigma_noise, H, H_adj, degradation)

            model = dinv.sampling.DiffPIR(
                self.model, data_fidelity=data_fidelity, sigma=sigma_noise, zeta=self.zeta, lambda_=self.lmbda, device=self.device)
            x = 2 * model(noisy_img, physics) - 1

            restored_img = x.detach()
            noisy_img = 2 * noisy_img - 1
            utils.save_images(clean_img, noisy_img, restored_img,
                              self.args, H_adj, iter='final')
            utils.compute_psnr(clean_img, noisy_img,
                               restored_img, self.args, H_adj, iter=100)
            utils.compute_ssim(clean_img, noisy_img,
                               restored_img, self.args, H_adj, iter=100)
            utils.compute_lpips(clean_img, noisy_img,
                                restored_img, self.args, H_adj, iter=100)

        utils.compute_average_psnr(self.args)
        utils.compute_average_ssim(self.args)
        utils.compute_average_lpips(self.args)

    def run_method(self, data_loaders, degradation, sigma_noise, H_funcs=None):

        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        if self.args.noise_type == 'laplace':
            self.args.save_path_ip = os.path.join(
                'results_laplace', self.args.save_path_ip)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise, H_funcs)


class DataFidelity_SuperResolution(DataFidelity):

    def __init__(self, sigma, H, H_adj, degradation):
        super().__init__()
        self.norm = 1 / sigma**2
        self.sigma = sigma
        self.H = H
        self.H_adj = H_adj
        self.degradation = degradation

    def prox(self, x, y, physics, gamma=1.0):

        d = self.H_adj(y) * self.norm + 1 / gamma * x

        diag = self.degradation.downsampling_matrix
        h_adj_h = torch.matmul(diag.T, diag)
        sol_tmp = 1 / (torch.diag(h_adj_h) * self.norm + 1 / gamma)

        sol_tmp = sol_tmp[None, None, :] * \
            d.flatten(start_dim=2)
        sol = sol_tmp.reshape(d.shape)

        return sol


class DataFidelity_GaussianDeblurring(DataFidelity):

    def __init__(self, sigma, H, H_adj, degradation):
        super().__init__()
        self.norm = 1 / sigma**2
        self.sigma = sigma
        self.degradation = degradation
        self.H_adj = H_adj
        self.device = 'cuda'

    def prox(self, x, y, physics, gamma=1.0):

        d = self.H_adj(y) * self.norm + 1 / gamma * x
        sol = torch.zeros_like(d)

        fft_d = torch.fft.fft2(d)
        kernel = self.degradation.filter
        kernel_size = kernel.shape[2]
        kernel_id = torch.zeros_like(kernel)
        kernel_id[:, :, kernel_size//2, kernel_size//2] = 1
        fft_kernel = torch.fft.fft2(kernel)
        inv = self.norm * fft_kernel * \
            torch.conj(fft_kernel) + 1 / gamma
        sol = torch.real(torch.fft.ifft2(fft_d / inv))
        return sol


class DataFidelity_Inpainting(DataFidelity):

    def __init__(self, sigma, H, H_adj, degradation):
        super().__init__()
        self.norm = 1 / sigma**2
        self.sigma = sigma
        self.H = H
        self.H_adj = H_adj

    def prox(self, x, y, physics, gamma=1.0):

        d = self.H_adj(y) * self.norm + 1 / gamma * x
        sol = torch.zeros_like(d)

        for i in range(d.shape[0]):
            sol_tmp = (1 / (self.H(torch.ones_like(x))
                       [i] * self.norm + 1 / gamma)) * d[i]
            sol[i] = sol_tmp.reshape(d[i].shape)
        return sol


class ForwardOperator(DecomposablePhysics):

    def __init__(self, noise_model, H, H_adj):
        super().__init__(noise_model=noise_model)
        self.H = H
        self.H_adj = H_adj

    def A(self, x, filter=None, **kwargs):
        return self.H(x)

    def A_adjoint(self, y, filter=None, **kwargs):
        return self.H_adj(y)


class LaplaceNoise(NoiseModel):

    def __init__(self, sigma=0.1, rng: torch.Generator = None):
        super().__init__(rng=rng)
        self.update_parameters(sigma=sigma)

    def forward(self, x, sigma=None, seed=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param float, torch.Tensor sigma: standard deviation of the noise.
            If not None, it will overwrite the current noise level.
        :param int seed: the seed for the random number generator, if `rng` is provided.

        :returns: noisy measurements
        """
        self.update_parameters(sigma=sigma)
        sigma = self.sigma
        noise = torch.distributions.laplace.Laplace(
            torch.zeros_like(x), sigma * torch.ones_like(x)).sample().to(x.device)
        return x + noise
