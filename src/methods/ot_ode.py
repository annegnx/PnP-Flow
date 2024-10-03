import torch
import utils as utils
from matplotlib import pyplot as plt
import os
import ImageGeneration.models.utils as mutils
from time import perf_counter


class OT_ODE(object):

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

    def initialization(self, noisy_img, t0):
        return t0 * noisy_img + (1-t0) * torch.randn_like(noisy_img)
        # return noisy_img

    def solve_ip(self, test_loader, degradation, sigma_noise, H_funcs=None):
        self.args.sigma_noise = sigma_noise

        H = degradation.H
        H_adj = degradation.H_adj

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch

            noisy_img = H(clean_img.clone().to(
                self.device))  # .reshape(clean_img.shape)
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * sigma_noise
            noisy_img = noisy_img.to(self.device)
            clean_img = clean_img.to('cpu')

            # intialize the image with the adjoint operator
            x = H_adj(noisy_img.clone()).to(self.device)
            x = self.initialization(
                H_adj(noisy_img.clone()), self.args.start_time)

            steps, delta = self.args.steps_ode, 1 / self.args.steps_ode

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            for count, iteration in enumerate(range(int(steps * self.args.start_time), int(steps))):

                if self.args.compute_time:
                    time_counter_1 = perf_counter()

                with torch.no_grad():
                    t1 = torch.ones(len(x), device=self.device) * \
                        delta * iteration
                    vt = self.model_forward(x, t1)
                    rt_squared = ((1-t1)**2 / ((1-t1)**2 + t1**2)
                                  ).view(-1, 1, 1, 1)
                    x1_hat = x + (1-t1.view(-1, 1, 1, 1)) * vt

                    # sovle linear problem Cx=d
                    d = noisy_img - H(x1_hat)

                    sol = torch.zeros_like(d)

                    if self.args.problem == "inpainting" or self.args.problem == "random_inpainting" or self.args.problem == "paintbrush_inpainting":
                        for i in range(d.shape[0]):

                            sol_tmp = 1 / \
                                (H(torch.ones_like(x))[
                                    i] * rt_squared[i] + sigma_noise**2) * d[i]
                            sol[i] = sol_tmp.reshape(d[i].shape)

                    elif self.args.problem == "denoising":
                        for i in range(d.shape[0]):
                            sol_tmp = d[i] / \
                                (rt_squared[i] + sigma_noise**2)
                            sol[i] = sol_tmp.reshape(d[i].shape)

                    elif self.args.problem == "superresolution":
                        rt_squared = torch.tensor((1-delta * iteration)**2 / ((1-delta * iteration)**2 + delta * iteration**2)
                                                  )

                        diag = degradation.downsampling_matrix
                        h_h_adj = torch.matmul(diag, diag.T)
                        sol_tmp = 1 / (torch.diag(h_h_adj) * rt_squared +
                                       torch.tensor(sigma_noise)**2)

                        sol_tmp = sol_tmp[None, None, :] * \
                            d.flatten(start_dim=2)
                        sol = sol_tmp.reshape(d.shape)

                    elif self.args.problem == "gaussian_deblurring_FFT":
                        fft_d = torch.fft.fft2(d)
                        kernel = degradation.filter
                        kernel_size = kernel.shape[2]
                        kernel_id = torch.zeros_like(kernel)
                        kernel_id[:, :, kernel_size//2, kernel_size//2] = 1
                        fft_kernel = torch.fft.fft2(kernel)
                        inv = rt_squared * fft_kernel * \
                            torch.conj(fft_kernel) + \
                            sigma_noise**2
                        sol = torch.fft.ifft2(fft_d / inv)
                    else:
                        for i in range(d.shape[0]):
                            def C_ope(z):
                                z = z.reshape(
                                    noisy_img.shape[1:]).unsqueeze(0)
                                return (rt_squared[i].unsqueeze(0) * H(H_adj(z)) + sigma_noise**2 * z).reshape(-1)

                            sol_tmp, _ = utils.GMRES(
                                C_ope, d[i].reshape(-1), max_iter=100)
                            sol[i] = sol_tmp.reshape(d[i].shape)

                    vec = H_adj(sol)
                # do vector jacobian product
                t = t1.view(-1, 1, 1, 1)
                if self.args.gamma == "constant":
                    gamma = 1
                elif self.args.gamma == "gamma_t":
                    gamma = torch.sqrt(t / (t**2 + (1 - t)**2))
                g = torch.autograd.functional.vjp(
                    lambda z:  self.model_forward(z, t1), inputs=x, v=vec)[1]

                with torch.no_grad():
                    g = vec + (1-t1.view(-1, 1, 1, 1)) * g

                    ratio = (1-t1.view(-1, 1, 1, 1)) / t1.view(-1, 1, 1, 1)
                    v_adapted = vt + ratio * gamma * g
                    x_new = x + delta * v_adapted

                    x = x_new

                    if self.args.save_results:
                        if iteration % 10 == 0 or self.should_save_image(iteration, steps):
                            restored_img = x.detach().clone()
                    # utils.save_images(
                    #     clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                    utils.compute_psnr(clean_img, noisy_img,
                                       restored_img, self.args, H_adj, iter=iteration)
                    utils.compute_ssim(clean_img, noisy_img,
                                       restored_img, self.args, H_adj, iter=iteration)

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
                                   restored_img, self.args, H_adj, iter=iteration)
                utils.compute_ssim(
                    clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def should_save_image(self, iteration, steps):
        return iteration % (steps // 10) == 0

    def run_method(self, data_loaders, degradation, sigma_noise):
        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise)
