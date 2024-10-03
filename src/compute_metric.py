import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio

from src.dataloaders import DataLoaders
from src.models import InceptionV3
import src.fid_score as fs
import utils


class ComputeMetric:
    def __init__(self, full_data, generative_method, device, args):
        self.train_set = full_data['train']
        self.test_set = full_data['test']
        self.device = device
        self.args = args
        # TODO: call it model (because of Hurault example)
        self.generative_method = generative_method
        self.save_path = self.args.root + \
            'results/{}/{}/'.format(
                self.args.dataset, self.args.latent)

    def compute_psnr_test_set(self):
        y = next(iter(self.test_set))
        psnr_ope = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        sigma_list = torch.linspace(0.001, 0.5, 10)
        for i, sigma in enumerate(sigma_list):
            x = y + torch.randn(y.size(), device=self.device) * sigma
            torch.set_grad_enabled(True)
            sigma_chan = sigma * \
                torch.ones(y.shape[0], 1, 1, 1, device=self.device)
            x_hat = self.generative_method.forward(x, sigma_chan)[0]
            psnr_ope.reset()
            psnr_noisy_value = psnr_ope(x, y)
            psnr_ope.reset()
            psnr_clean_value = psnr_ope(x_hat, y)
            with open(self.save_path + 'metrics.txt', 'a') as file:
                file.write(
                    f'Model: Gradient Step, Epoch: final, PSNR noisy at sigma {sigma}: {psnr_noisy_value.item()}\n')
                file.write(
                    f'Model: Gradient Step, Epoch: final, PSNR clean at sigma {sigma}: {psnr_clean_value.item()}\n')

    def compute_fid_test_set(self):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        incep_model = InceptionV3([block_idx]).to(self.device)
        data_v = next(iter(self.train_set))
        gt = data_v[0].to(self.device)
        gt = gt.permute(0, 2, 3, 1).cpu().numpy()
        if gt.shape[-1] == 1:
            gt = np.concatenate([gt, gt, gt], axis=-1)
        gt = np.transpose(gt, axes=(0, 3, 1, 2))
        batch_size = 50
        m1, s1 = fs.calculate_activation_statistics(
            gt, incep_model, batch_size, 2048, self.device)

        data_v = next(iter(self.test_set))
        gt = data_v[0].to(self.device)
        gt = gt.permute(0, 2, 3, 1).cpu().numpy()
        if gt.shape[-1] == 1:
            gt = np.concatenate([gt, gt, gt], axis=-1)
        gt = np.transpose(gt, axes=(0, 3, 1, 2))
        m2, s2 = fs.calculate_activation_statistics(
            gt, incep_model, batch_size, 2048, self.device)
        fid_value = fs.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def compute_metrics(self, num_samples):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        incep_model = InceptionV3([block_idx]).to(self.device)
        data_v = next(iter(self.test_set))
        gt = data_v[0].to(self.device)[:num_samples]
        gt = gt.permute(0, 2, 3, 1).cpu().numpy()
        if gt.shape[-1] == 1:
            gt = np.concatenate([gt, gt, gt], axis=-1)
        gt = np.transpose(gt, axes=(0, 3, 1, 2))
        batch_size = 50
        m1, s1 = fs.calculate_activation_statistics(
            gt, incep_model, batch_size, 2048, self.device)

        samples = torch.empty(0).to(self.device)
        n_iter = 50
        for i in range(n_iter):
            print('Iteration: ', i)
            samples = torch.cat([samples, self.generative_method.apply_flow_matching(
                num_samples // n_iter)], dim=0)
        gen = torch.clip(samples.permute(0, 2, 3, 1), 0, 1).cpu().numpy()
        if gen.shape[-1] == 1:
            gen = np.concatenate([gen, gen, gen], axis=-1)
        gen = np.transpose(gen, axes=(0, 3, 1, 2))
        m2, s2 = fs.calculate_activation_statistics(
            gen, incep_model, batch_size, 2048, self.device)
        fid_value = fs.calculate_frechet_distance(m1, s1, m2, s2)

        test_samples = data_v[0].to(self.device)[:num_samples].squeeze(1).cpu()
        gen_samples = samples.squeeze(1).cpu()

        # sw_approx
        sw_score = utils.sw_approx(test_samples, gen_samples)

        # compute_vendi_score
        vendi_score_true = utils.compute_vendi_score(test_samples.numpy())
        vendi_score_gen = utils.compute_vendi_score(gen_samples.numpy())
        fid_value_test = self.compute_fid_test_set()

        with open(self.save_path + 'metrics.txt', 'a') as file:
            file.write(f'Epoch: final, FID: {fid_value}\n')
            file.write(f'Epoch: final, SW: {sw_score}\n')
            file.write(f'Epoch: final, VENDI_test: {vendi_score_true}\n')
            file.write(f'Epoch: final, VENDI_gen: {vendi_score_gen}\n')
            file.write(f'FID test set: {fid_value_test}\n')
