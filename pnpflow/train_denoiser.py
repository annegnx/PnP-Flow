import torch
import skimage.io as io
import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import os
from torch.optim import Adam
from torch.optim import lr_scheduler
from torchmetrics.image import PeakSignalNoiseRatio


class GRADIENT_STEP_DENOISER(object):

    def __init__(self, model, device, args):
        self.d = args.dim_image
        self.num_channels = args.num_channels
        self.device = device
        self.args = args
        self.lr = args.lr
        self.model = model.to(device)
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.scheduler_milestones = [300, 600, 900, 1200]
        self.scheduler_gamma = 0.5
        self.jacobian_loss_weight = -1
        self.jacobian_loss_type = "max"
        self.eps_jacobian_loss = 0.1
        self.power_method_nb_step = 50
        self.power_method_error_threshold = 1e-2
        self.power_method_error_momentum = 0.
        self.power_method_mean_correction = False
        self.sigma_step = False
        self.weight_Ds = 1.
        self.grad_matching = True

    def calculate_grad(self, x, sigma, compute_g=False):
        '''
        Calculate Dg(x) the gradient of the regularizer g at input x
        :param x: torch.tensor Input image
        :param sigma: Denoiser level (std)
        :return: Dg(x), DRUNet output N(x)
        '''
        x = x.float()
        x = x.requires_grad_()

        N = self.model.forward(x, sigma)
        JN = torch.autograd.grad(
            N, x, grad_outputs=x - N, create_graph=True, only_inputs=True)[0]
        Dg = x - N - JN
        if compute_g:
            g = 0.5 * torch.sum((x - N).reshape((x.shape[0], -1)) ** 2)
            return Dg, N, g
        else:
            return Dg, N

    def forward(self, x, sigma):
        '''
        Denoising with Gradient Step Denoiser
        :param x:  torch.tensor input image
        :param sigma: Denoiser level (std)
        :return: Denoised image x_hat, Dg(x) gradient of the regularizer g at x
        '''
        if self.grad_matching:  # If gradient step denoising
            Dg, _ = self.calculate_grad(x, sigma)
            if self.sigma_step:  # possibility to multiply Dg by sigma
                x_hat = x - self.weight_Ds * sigma * Dg
            else:  # as realized in the paper (with weight_Ds=1)
                x_hat = x - self.weight_Ds * Dg
            return x_hat, Dg
        else:  # If denoising with standard forward CNN
            x_hat = self.model.forward(x, sigma)
            Dg = x - x_hat
            return x_hat, Dg

    def configure_optimizers(self):
        optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        optimizer = Adam(
            optim_params, lr=self.lr, weight_decay=0)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             self.scheduler_milestones,
                                             self.scheduler_gamma)
        return [optimizer], [scheduler]

    def power_iteration(self, operator, vector_size, steps=100, momentum=0.0, eps=1e-3,
                        init_vec=None, verbose=False):
        '''
        Power iteration algorithm for spectral norm calculation
        '''
        with torch.no_grad():
            if init_vec is None:
                vec = torch.rand(vector_size).to(self.device)
            else:
                vec = init_vec.to(self.device)
            vec /= torch.norm(vec.view(vector_size[0], -1),
                              dim=1, p=2).view(vector_size[0], 1, 1, 1)

            for i in range(steps):

                new_vec = operator(vec)
                new_vec = new_vec / \
                    torch.norm(new_vec.view(
                        vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1, 1)
                if momentum > 0 and i > 1:
                    new_vec -= momentum * old_vec
                old_vec = vec
                vec = new_vec
                diff_vec = torch.norm(new_vec - old_vec, p=2)
                if diff_vec < eps:
                    if verbose:
                        print("Power iteration converged at iteration: ", i)
                    break

        new_vec = operator(vec)
        div = torch.norm(
            vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0])
        lambda_estimate = torch.abs(
            torch.sum(vec.view(vector_size[0], -1) * new_vec.view(vector_size[0], -1), dim=1)) / div

        return lambda_estimate

    def jacobian_spectral_norm(self, y, x_hat, sigma, interpolation=False, training=False):
        '''
        Get spectral norm of Dg^2 the hessian of g
        :param y:
        :param x_hat:
        :param sigma:
        :param interpolation:
        :return:
        '''
        torch.set_grad_enabled(True)
        if interpolation:
            eta = torch.rand(y.size(0), 1, 1, 1,
                             requires_grad=True).to(self.device)
            x = eta * y.detach() + (1 - eta) * x_hat.detach()
            x = x.to(self.device)
        else:
            x = y

        x.requires_grad_()
        x_hat, Dg = self.forward(x, sigma)
        if self.grad_matching:
            # we calculate the lipschitz constant of the gradient operator Dg=Id-D
            def operator(vec): return torch.autograd.grad(
                Dg, x, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            # we calculate the lipschitz constant of the denoiser operator D
            f = x_hat
            def operator(vec): return torch.autograd.grad(
                f, x, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
        lambda_estimate = self.power_iteration(operator, x.size(), steps=self.power_method_nb_step, momentum=self.power_method_error_momentum,
                                               eps=self.power_method_error_threshold)
        return lambda_estimate

    def train_denoiser(self, train_loader, opt, num_epoch):
        tq = tqdm(range(num_epoch), desc='loss')
        for ep in tq:
            for iteration, (y, labels) in enumerate(train_loader):

                if y.size(0) == 0:
                    continue
                y = y.to(self.device)

                sigma = random.uniform(0, 0.25)
                u = torch.randn(y.size(), device=self.device)
                noise_in = u * sigma
                x = y + noise_in

                sigma_chan = sigma * \
                    torch.ones(y.shape[0], 1, 1, 1, device=self.device)

                x_hat, Dg = self.forward(x, sigma_chan.squeeze())
                self.psnr.update(x_hat, y)

                criterion = nn.MSELoss(reduction='none')
                loss = criterion(
                    x_hat.view(x.size()[0], -1), y.view(y.size()[0], -1)).mean(dim=1)

                if self.jacobian_loss_weight > 0:
                    jacobian_norm = self.jacobian_spectral_norm(
                        x, x_hat, sigma, interpolation=False, training=True)
                    if self.jacobian_loss_type == 'max':
                        jacobian_loss = torch.maximum(jacobian_norm, torch.ones_like(
                            jacobian_norm)-self.eps_jacobian_loss)
                    elif self.jacobian_loss_type == 'exp':
                        jacobian_loss = self.eps_jacobian_loss * torch.exp(jacobian_norm - torch.ones_like(
                            jacobian_norm)*(1+self.eps_jacobian_loss)) / self.eps_jacobian_loss
                    else:
                        print("jacobian loss not available")
                    jacobian_loss = torch.clip(jacobian_loss, 0, 1e3)
                    loss = (
                        loss + self.jacobian_loss_weight * jacobian_loss)

                loss = loss.mean()
                psnr = self.psnr.compute()

                opt.zero_grad()
                loss.backward()
                opt.step()

                # save loss in txt file
                with open(self.save_path + 'loss_training.txt', 'a') as file:
                    file.write(
                        f'Epoch: {ep}, iter: {iteration}, Loss: {loss.item()}\n')

            if ep % 1 == 0:

                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, self.model_path + 'gradient_step_denoiser_{}.pt'.format(ep))

                with open(self.save_path + 'losses_gradient_step.txt', 'a') as file:
                    file.write(
                        f'Epoch: {ep}, Loss: {loss.item()}, PSNR: {psnr.item()}\n')

    def train(self, data_loaders):

        self.save_path = self.args.root + \
            'results/{}/{}'.format(
                self.args.dataset, self.args.model)
        try:
            os.makedirs(self.save_path)
        except BaseException:
            pass

        self.model_path = self.args.root + \
            'model/{}/{}'.format(
                self.args.dataset, self.args.model)
        try:
            os.makedirs(self.model_path)
        except BaseException:
            pass

        train_loader = data_loaders['train']

        # create txt file for storing all information about model
        with open(self.save_path + 'model_info.txt', 'w') as file:
            file.write(f'PARAMETERS\n')
            file.write(
                f'Number of parameters: {sum(p.numel() for p in self.model.parameters())}\n')
            file.write(f'Number of epochs: {self.args.num_epoch}\n')
            file.write(f'Batch size: {self.args.batch_size_train}\n')
            file.write(f'Learning rate: {self.lr}\n')

        [opt], [scheduler] = self.configure_optimizers()
        self.train_denoiser(train_loader, opt, num_epoch=self.args.num_epoch)
        torch.save(self.model.state_dict(),
                   self.model_path + 'gradient_step_denoiser_final.pt')
