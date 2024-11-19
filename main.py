import os
import random
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from utils import load_cfg_from_cfg_file, merge_cfg_from_list
from degradations import *
from src.dataloaders import DataLoaders
from src.train_flow_matching import FLOW_MATCHING
from src.train_denoiser import GRADIENT_STEP_DENOISER
from src.compute_metric import ComputeMetric
from src.methods.pnp_flow import PNP_FLOW
from src.methods.d_flow import D_FLOW
from src.methods.ot_ode import OT_ODE
from src.methods.flow_priors import FLOW_PRIORS
from src.methods.pnp_gs import PROX_PNP
from src.methods.pnp_diff import PNP_DIFF
from utils import gaussian_blur, define_model, load_model
import warnings
warnings.filterwarnings("ignore", module="matplotlib\\..*")

torch.cuda.empty_cache()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('./' + 'config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    dataset_config = cfg.root + \
        'config/dataset_config/{}.yaml'.format(
            cfg.dataset)
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    method_config_file = cfg.root + \
        'config/method_config/{}.yaml'.format(
            cfg.method)
    cfg.update(load_cfg_from_cfg_file(method_config_file))

    if args.opts is not None:
        # override config with command line input
        cfg = merge_cfg_from_list(cfg, args.opts)

    # for all keys in the method config file, create a dictionary {key: value} in the cfg object cfg.dict_cfg_method
    method_cfg = load_cfg_from_cfg_file(method_config_file)
    cfg.dict_cfg_method = {}
    for key in method_cfg.keys():
        cfg.dict_cfg_method[key] = cfg[key]
    return cfg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    (model, state) = define_model(args)

    if args.train:
        args.batch_size = args.batch_size_train
        print('Training...')
        data_loaders = DataLoaders(
            args.dataset, args.batch_size_train, args.batch_size_train).load_data()
        if args.model == "ot":
            generative_method = FLOW_MATCHING(model, device, args)
        elif args.model == "gradient_step":
            generative_method = GRADIENT_STEP_DENOISER(model, device, args)
        else:
            raise ValueError(
                "Model not implemented yet: you can choose between 'ot' and 'gradient_step'")
        generative_method.train(data_loaders)
        print('Training done!')

    if args.eval:

        if args.model == "ot" or args.model == "gradient_step":
            model_path = args.root + \
                'model/{}/{}/{}/model_final.pt'.format(
                    args.dataset, args.latent, args.model)
            load_model(args.model, model, state, model_path, device)
            model.eval()

        elif args.model == "rectified":
            model_path = args.root + 'model/{}/{}/{}/model_final.pth'.format(
                args.dataset, args.latent, args.model)
            load_model(args.model, model, state, model_path, device)
            model.eval()

        elif args.model == "diffusion":
            model.eval()

        if args.model == "gradient_step":
            generative_method = GRADIENT_STEP_DENOISER(model, device, args)
        else:
            generative_method = FLOW_MATCHING(model, device, args)

        if args.compute_metrics:
            print('Computing metrics...')
            data_loaders = DataLoaders(args.dataset, 5000, 5000).load_data()
            metric = ComputeMetric(
                data_loaders, generative_method, device, args)
            metric.compute_metrics(5000)
            print('Computing metrics done!')

        if args.problem == "denoising":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.2
            degradation = Denoising()

        elif args.problem == "inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            if args.dim_image == 128:
                half_size_mask = 20
            elif args.dim_image == 256:
                half_size_mask = 40
            degradation = BoxInpainting(half_size_mask)

        elif args.problem == "paintbrush_inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            degradation = PaintbrushInpainting()

        elif args.problem == "random_inpainting":
            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.01
            p = 0.7
            degradation = RandomInpainting(p)

        elif args.problem == "superresolution":
            if args.dim_image == 128:
                print('Superresolution with scale factor 2')
                sf = 2
            elif args.dim_image == 256:
                print('Superresolution with scale factor 4')
                sf = 4
            if args.noise_type == 'laplace':
                sigma_noise = 0.3

            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            degradation = Superresolution(sf, args.dim_image)

        elif args.problem == "gaussian_deblurring_FFT":
            if args.dim_image == 128:
                sigma_blur = 1.0
            elif args.dim_image == 256:
                sigma_blur = 3.0

            if args.noise_type == 'laplace':
                sigma_noise = 0.3
            elif args.noise_type == 'gaussian':
                sigma_noise = 0.05
            kernel_size = 61
            degradation = GaussianDeblurring(
                sigma_blur, kernel_size, "fft", args.num_channels, args.dim_image, device)

        print('Solving the {} inverse problem with the method {}...'.format(
            args.problem, args.method))
        print('sigma_noise', sigma_noise)
        data_loaders = DataLoaders(
            args.dataset, args.batch_size_ip, args.batch_size_ip).load_data()
        if args.noise_type == 'laplace':
            args.save_path = os.path.join(
                args.root, 'results_laplace', args.dataset, args.model, args.problem, args.method, args.eval_split)
        elif args.noise_type == 'gaussian':
            args.save_path = os.path.join(
                args.root, 'results', args.dataset, args.model, args.problem, args.method, args.eval_split)
        try:
            os.makedirs(args.save_path)
        except FileExistsError:
            pass

        if args.method == 'pnp_flow':
            method = PNP_FLOW(model, device, args)
        elif args.method == 'd_flow':
            method = D_FLOW(model, device, args)
        elif args.method == 'ot_ode':
            method = OT_ODE(model, device, args)
        elif args.method == 'flow_priors':
            method = FLOW_PRIORS(model, device, args)
        elif args.method == 'pnp_gs':
            method = PROX_PNP(generative_method, device, args)
        elif args.method == 'pnp_diff':
            method = PNP_DIFF(model, device, args)
        else:
            raise ValueError("The method your entered does not exist")

        method.run_method(data_loaders, degradation, sigma_noise)


if __name__ == "__main__":
    main()
