import torch
import numpy as np
import numpy as np
import utils as utils
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint
import ImageGeneration.models.utils as mutils
from time import perf_counter


class D_FLOW(object):

    """This class implements the D-Flow method for solving inverse problems, from the paper Ben-Hamu et al, "D-Flow: Differentiating through flows for controlled generation", 2024.
    It consists in minimizing over the latent space the loss function norm(H(Tz) - y \\Vert)**2 using the implicit prior given by the transport map T=f(z,1). The minimization is performed using gradient descent.
    """

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

    def gaussian(self, img):
        if img.ndim != 4:
            raise RuntimeError(
                f"Expected input `img` to be an 4D tensor, but got {img.shape}")
        return (img**2).sum([1, 2, 3]) * 0.5

    def forward_flow_matching(self, z):
        steps = self.args.steps_euler
        delta = (1 - self.args.start_time) / (steps - 1)
        for i in range(steps - 1):
            t1 = torch.ones(len(z), device=self.device) * \
                delta * i + self.args.start_time
            z = z + delta * self.model_forward(z + delta /
                                               2 * self.model_forward(z, t1), t1 + delta / 2)
        return z

    def inverse_flow_matching(self, z):
        flow_class = cnf(self.model, self.args.model)
        z_t = odeint(flow_class, z,
                     torch.tensor([1.0, 0.0]).to(self.device),
                     atol=1e-2,
                     rtol=1e-2,
                     method='dopri5',
                     )
        x = z_t[-1].detach()
        return x

    def gaussian(self, img):
        if img.ndim != 4:
            raise RuntimeError(
                f"Expected input `img` to be an 4D tensor, but got {img.shape}")
        return (img**2).sum([1, 2, 3]) * 0.5

    def solve_ip(self, test_loader, degradation, sigma_noise):
        H = degradation.H
        H_adj = degradation.H_adj

        self.args.sigma_noise = sigma_noise

        loader = iter(test_loader)
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch

            noisy_img = H(clean_img.clone().to(self.device))
            torch.manual_seed(batch)
            noisy_img += torch.randn_like(noisy_img) * sigma_noise
            noisy_img = noisy_img.to(self.device)
            clean_img = clean_img.to('cpu')

            x = H_adj(noisy_img.clone()).to(self.device)
            z = self.inverse_flow_matching(x).to(self.device)

            # blend intialization as in the d-flow paper
            z = np.sqrt(self.args.alpha) * z + np.sqrt(1 -
                                                       self.args.alpha) * torch.randn_like(z)
            z = z.detach().requires_grad_(True)

            # start the gradient descent
            optim_img = torch.optim.LBFGS(
                [z], max_iter=self.args.LBFGS_iter, history_size=100, line_search_fn='strong_wolfe')
            d = z.shape[1] * z.shape[2] * z.shape[3]

            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0

            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)

            # tq = tqdm(range(self.args.max_iter), desc='psnr')
            for iteration in range(self.args.max_iter):
                if self.args.compute_time:
                    time_counter_1 = perf_counter()

                def closure():
                    optim_img.zero_grad()  # Reset gradients
                    reg = - torch.clamp(self.gaussian(z), min=-1e6, max=1e6) + (
                        d - 1) * torch.log(torch.sqrt(torch.sum(z**2, dim=(1, 2, 3))) + 1e-5)

                    loss = (torch.sum((H(self.forward_flow_matching(z)) -
                            noisy_img)**2, dim=(1, 2, 3)) + self.args.lmbda * reg).sum()
                    loss.backward()  # Compute gradients
                    return loss

                optim_img.step(closure)

                restored_img = self.forward_flow_matching(z.detach())
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
                if self.args.save_results:
                    utils.save_images(
                        clean_img, noisy_img, restored_img, self.args, H_adj, iter=iteration)
                    utils.compute_psnr(clean_img, noisy_img,
                                       restored_img, self.args, H_adj, iter=iteration)
                    utils.compute_ssim(clean_img, noisy_img,
                                       restored_img, self.args, H_adj, iter=iteration)

                del restored_img

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
                # Compute PSNR values
                z = z.detach().requires_grad_(False)
                restored_img = self.forward_flow_matching(z.detach())
                utils.save_images(clean_img, noisy_img, restored_img,
                                  self.args, H_adj, iter='final')
                utils.compute_psnr(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=iteration)
                utils.compute_ssim(clean_img, noisy_img,
                                   restored_img, self.args, H_adj, iter=iteration)
                del restored_img

            del noisy_img, clean_img, x, z

        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)

    def run_method(self, data_loaders, degradation, sigma_noise):

        # Construct the save path for results
        folder = utils.get_save_path_ip(self.args.dict_cfg_method)
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)

        # Create the directory if it doesn't exist
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # Solve the inverse problem
        self.solve_ip(
            data_loaders[self.args.eval_split], degradation, sigma_noise)


class cnf(torch.nn.Module):

    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name

    def model_forward(self, x, t):
        if self.model_name == "ot":
            return self.model(x, t)

        elif self.model_name == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            t_ = t[:, None, None, None]
            v = model_fn(x.type(torch.float), t * 999)
            return v

    def forward(self, t, x):
        with torch.no_grad():
            z = self.model_forward(x, t.repeat(x.shape[0]))
        return z
