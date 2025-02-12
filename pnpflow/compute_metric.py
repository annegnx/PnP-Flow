import torch
import numpy as np
from pnpflow.models import InceptionV3
import pnpflow.fid_score as fs


class ComputeMetric:
    def __init__(self, full_data, generative_method, device, args):
        self.test_set = full_data['test']
        self.device = device
        self.args = args
        self.generative_method = generative_method
        self.save_path = self.args.root + \
            'results/{}/{}/'.format(
                self.args.dataset, self.args.latent)

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

        with open(self.save_path + 'metrics.txt', 'a') as file:
            file.write(f'Epoch: final, FID: {fid_value}\n')
