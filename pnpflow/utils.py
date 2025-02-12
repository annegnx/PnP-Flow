from functools import partial
import torch
from numpy import linalg
import numpy as np
from typing import List
import os
import yaml
import ast
import torch
import copy
from ast import literal_eval
# from pykeops.torch import LazyTensor
import scipy
import scipy.linalg
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as v2
import argparse
from skimage.metrics import peak_signal_noise_ratio as PSNR
from ignite.metrics import SSIM
from collections import defaultdict
import math
from random import randint
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import deepinv as dinv
from random import randint, seed
from pnpflow.models import UNet
import pnpflow.image_generation.losses as losses
import pnpflow.image_generation.models.utils as mutils
import pnpflow.image_generation.datasets as datasets
import pnpflow.image_generation.models.ema as mema
from pnpflow.image_generation.utils import restore_checkpoint
from pnpflow.image_generation.configs.rectified_flow.celeba_hq_pytorch_rf_gaussian import get_config as get_config_celebahq
from pnpflow.image_generation.configs.rectified_flow.afhq_cat_pytorch_rf_gaussian import get_config as get_config_afhq_cat
import warnings
import lpips
warnings.filterwarnings("ignore", module="matplotlib\..*")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn_alex = lpips.LPIPS(net='alex').to(
    DEVICE)  # best forward scores


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, super(
                CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)

    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg: CfgNode,
                        cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        # assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        if subkey in cfg:
            value = _decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(
                value, cfg[subkey], subkey, full_key
            )
            setattr(new_cfg, subkey, value)
        else:
            value = _decode_cfg_value(v)
            setattr(new_cfg, subkey, value)
    return new_cfg


def define_model(args):
    if args.model == "ot" or args.model == "gradient_step":
        model = UNet(input_channels=args.num_channels,
                     input_height=args.dim_image,
                     ch=32,
                     ch_mult=(1, 2, 4, 8),
                     num_res_blocks=6,
                     attn_resolutions=(16, 8),
                     resamp_with_conv=True,
                     )
        return (model, None)

    elif args.model == "diffusion":
        model = dinv.models.DiffUNet()
        return (model, None)

    elif args.model == "rectified":
        if args.dataset == "celebahq":
            config = get_config_celebahq()
        elif args.dataset == "afhq_cat":
            config = get_config_afhq_cat()
        # Initialize model
        score_model = mutils.create_model(config)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = mema.ExponentialMovingAverage(
            score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
        return score_model, state
    else:
        raise Exception("Unknown model!")


def load_model(name_model, model, state, checkpoint_path, device):

    if name_model == "ot":
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)

    elif name_model == "rectified":
        ckpt_path = checkpoint_path  # 'model/celebahq/gaussian/model_final.pth'
        state = restore_checkpoint(ckpt_path, state, device=device)

    elif name_model == "gradient_step":
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)


def hut_estimator(NO_test, v, inp, t):
    batch_size = inp.shape[0]

    def fn(x):
        x = x.reshape(batch_size * NO_test, inp.shape[1], inp.shape[2],
                      inp.shape[3])
        return v(
            x,
            torch.tensor(
                [t]).repeat(
                x.shape[0]).to('cuda')).reshape(
            NO_test,
            batch_size,
            inp.shape[1],
            inp.shape[2],
            inp.shape[3])

    inp_new = inp.repeat(NO_test, 1, 1, 1, 1).clone()
    # eps = torch.randn(NO_test, batch_size,
    #                   inp.shape[1], inp.shape[2], inp.shape[3], device='cuda')
    eps = ((torch.rand(NO_test, batch_size,
                       inp.shape[1], inp.shape[2], inp.shape[3], device='cuda') < 0.5)) * 2 - 1
    # t0_hut = time.time()
    prod = torch.autograd.functional.jvp(
        fn, inp_new, eps, create_graph=True)[1]

    prod = (prod * eps).sum(dim=(2, 3, 4)).mean(0)
    return prod


def gaussian_2d_kernel(sigma, size):
    """Generate a 2D Gaussian kernel."""
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    y = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x, y = torch.meshgrid(x, y)
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def upsample(x, sf):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros(
        (x.shape[0],
         x.shape[1],
         x.shape[2] *
         sf,
         x.shape[3] *
         sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def gaussian_blur(x, sigma_blur, size_kernel):
    '''Blur a tensor image with Gaussian filter

    x: tensor image, NxCxWxH
    sigma: standard deviation of the Gaussian kernel
    '''
    kernel = gaussian_2d_kernel(sigma_blur, size_kernel).type_as(x)
    # uniform kernel
    kernel = kernel.view(1, 1, size_kernel, size_kernel)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    # kernel = kernel.flip(-1).flip(-2)
    return F.conv2d(x, kernel, stride=1, padding='same', groups=x.shape[1])


def square_mask(x, half_size_mask):
    """
    Black square mask of 20 x 20 pixels at the center of the image
    """
    d = x.shape[2] // 2

    mask = torch.ones_like(x)
    mask[:, :, d - half_size_mask:d + half_size_mask,
         d - half_size_mask:d + half_size_mask] = 0
    return mask * x


def paintbrush_mask(x):
    """
    Black mask that looks like paintbrush on the image. Make it random
    """
    mask_generator = MaskGenerator(x.shape[2], x.shape[3], 1, rand_seed=42)
    mask = torch.zeros_like(x)
    for i in range(x.shape[0]):
        mask_i = torch.from_numpy(
            mask_generator.sample().transpose((2, 0, 1))).to(x.device) - 1
        mask_i = (mask_i == 0).squeeze(0)
        mask[i] = mask_i
    return mask * x


def random_mask(x, p, seed=None):
    """
    Random mask on x
    """
    np.random.seed(42)
    mask = torch.from_numpy(np.random.binomial(n=1, p=1-p, size=(
        x.shape[0], x.shape[2], x.shape[3]))).to(x.device)

    return mask.unsqueeze(1) * x


# comes from deepinv
def bicubic_filter(factor=2):
    r"""
    Bicubic filter.

    It has size (4*factor, 4*factor) and is defined as

    .. math::

        \begin{equation*}
            w(x, y) = \begin{cases}
                (a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{if } |x| \leq 1 \\
                a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{if } 1 < |x| < 2 \\
                0 & \text{otherwise}
            \end{cases}
        \end{equation*}

    for :math:`x, y \in {-2\text{factor} + 0.5, -2\text{factor} + 0.5 + 1/\text{factor}, \ldots, 2\text{factor} - 0.5}`.

    :param int factor: downsampling factor
    """
    x = np.arange(start=-2 * factor + 0.5, stop=2 * factor, step=1) / factor
    a = -0.5
    x = np.abs(x)
    w = ((a + 2) * np.power(x, 3) - (a + 3) * np.power(x, 2) + 1) * (x <= 1)
    w += (
        (a * np.power(x, 3) - 5 * a * np.power(x, 2) + 8 * a * x - 4 * a)
        * (x > 1)
        * (x < 2)
    )
    w = np.outer(w, w)
    w = w / np.sum(w)
    return torch.Tensor(w).unsqueeze(0).unsqueeze(0)


def save_samples(samples, train_samples, path, args):

    samples = samples.clone().permute(0, 2, 3, 1).cpu().data.numpy()
    train_samples = train_samples.clone().permute(0, 2, 3, 1).cpu().data.numpy()
    batch_samples_size = samples.shape[0]
    cols = int(math.sqrt(batch_samples_size))  # Number of columns
    rows = int(batch_samples_size / cols)   # Number of rows
    fig, ax = plt.subplots(rows, 2 * cols, figsize=(20, 20))
    for i in range(rows):
        for j in range(cols):
            if args.num_channels == 1:
                ax[i, j].imshow(samples[i + j * rows].squeeze(-1),
                                cmap='gray', vmin=0, vmax=1)
            else:
                ax[i, j].imshow(samples[i + j * rows])
    for i in range(rows):
        for j in range(cols, 2*cols+1):
            if i+(j - cols)*rows < train_samples.shape[0]:
                if args.num_channels == 1:
                    ax[i, j].imshow(train_samples[i+(j - cols)*rows].squeeze(-1),
                                    cmap='gray', vmin=0, vmax=1)
                else:
                    ax[i, j].imshow(train_samples[i+(j - cols)*rows])
    ax[0, 0].set_title("Model samples")
    ax[0, cols].set_title("Training samples")

    for ax_ in ax.flatten():
        ax_.set_xticks([])
        ax_.set_yticks([])

    plt.savefig(path),
    plt.close(fig)


def save_images(clean_img, noisy_img, rec_img, args, H_adj, iter='final'):

    clean_img = postprocess(clean_img.clone(), args)
    noisy_img = postprocess(noisy_img.clone(), args)
    rec_img = postprocess(rec_img.clone(), args)
    H_adj_noisy_img = postprocess(H_adj(torch.ones_like(noisy_img)), args)

    # save images all together
    batch_size = clean_img.shape[0]

    cols = int(math.sqrt(batch_size))  # Number of columns
    rows = int(batch_size / cols)   # Number of rows

    clean_img = clean_img.permute(0, 2, 3, 1).cpu().data.numpy()
    noisy_img = noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()
    rec_img = rec_img.permute(0, 2, 3, 1).cpu().data.numpy()
    H_adj_noisy_img = H_adj_noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()

    if iter != 'final':
        if batch_size == 1:
            fig = plt.figure()
            plt.imshow(rec_img[0])
        elif batch_size == 2:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(rec_img[0])
            ax[1].imshow(rec_img[1])
            for ax_ in ax.flatten():
                ax_.set_xticks([])
                ax_.set_yticks([])
        else:
            fig, ax = plt.subplots(rows, cols, figsize=(20, 20))
            for i in range(rows):
                for j in range(cols):
                    if args.num_channels == 1:
                        ax[i, j].imshow(rec_img[i + j * rows].squeeze(-1),
                                        cmap='gray', vmin=0, vmax=1)
                    else:
                        ax[i, j].imshow(rec_img[i + j * rows])

            for ax_ in ax.flatten():
                ax_.set_xticks([])
                ax_.set_yticks([])

        plt.savefig(os.path.join(args.save_path_ip,
                    f"{args.problem}_{args.method}_batch{args.batch}_iter{iter}.png")),
        plt.close(fig)

    list_word = ['clean', 'noisy', args.method]
    if iter == 'final':
        for k, img in enumerate([clean_img, noisy_img, rec_img]):

            if batch_size == 1:
                fig = plt.figure()
                plt.imshow(img[0])
            elif batch_size == 2:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(img[0])
                ax[1].imshow(img[1])
            else:
                fig, ax = plt.subplots(rows, cols, figsize=(20, 20))
                for i in range(rows):
                    for j in range(cols):
                        if args.num_channels == 1:
                            ax[i, j].imshow(img[i + j * rows].squeeze(-1),
                                            cmap='gray', vmin=0, vmax=1)
                        else:
                            ax[i, j].imshow(img[i + j * rows])

                for ax_ in ax.flatten():
                    ax_.set_xticks([])
                    ax_.set_yticks([])

            plt.savefig(os.path.join(
                args.save_path_ip, f"{args.problem}_{list_word[k]}_batch{args.batch}_final.png")),
            plt.close(fig)

    # save images one by one, in .eps, adding the name of the method (args.method) and the PSNR value to the path
    list_batch = [0, 1, 2, 5]
    # if ((args.batch < 8 and args.method == 'd_flow') or args.batch < 4) and args.eval_split == 'test' and iter == 'final':
    # if ((args.batch < 8 and args.method == 'd_flow') or args.batch in list_batch) and args.eval_split == 'test' and iter == 'final':
    # if args.eval_split == 'test' and iter == 'final':
    if args.eval_split == 'test':
        if ((args.batch < 8 and args.method == 'd_flow') or args.batch < 4):
            print('Saving images one by one')
            for i in range(batch_size):

                if args.problem == 'superresolution' or args.problem == 'superresolution_bicubic':
                    psnr_noisy = PSNR(
                        clean_img[i], H_adj_noisy_img[i], data_range=1.)
                else:
                    psnr_noisy = PSNR(
                        clean_img[i], noisy_img[i], data_range=1.)
                psnr_rec = PSNR(clean_img[i], rec_img[i], data_range=1.)

                for k, img in enumerate([clean_img, noisy_img, rec_img]):

                    fig = plt.figure()
                    plt.imshow(img[i])
                    plt.axis('off')
                    if k == 0 and args.method == 'pnp_flow':
                        plt.savefig(os.path.join(args.save_path_ip, f"{args.problem}_{list_word[k]}_batch{args.batch}_im{i}.eps"),
                                    bbox_inches='tight', pad_inches=0)
                    if k == 1 and args.method == 'pnp_flow':
                        plt.savefig(os.path.join(args.save_path_ip, f"{args.problem}_{list_word[k]}_batch{args.batch}_im{i}_pnsr{psnr_noisy:4.2f}.eps"),
                                    bbox_inches='tight', pad_inches=0)
                    if k == 2:
                        print(os.path.join(
                            args.save_path_ip, f"{args.problem}_{list_word[k]}_batch{args.batch}_im{i}_pnsr{psnr_rec:4.2f}.eps"))
                        plt.savefig(os.path.join(args.save_path_ip, f"{args.problem}_{list_word[k]}_batch{args.batch}_im{i}_iter{iter}_pnsr{psnr_rec:4.2f}.eps"),
                                    bbox_inches='tight', pad_inches=0)
                    plt.close(fig)


def preprocess(img, args):
    if args.model == "rectified":
        if args.dataset == "celebahq":
            config = get_config_celebahq()
        elif args.dataset == "afhq_cat":
            config = get_config_afhq_cat()
        scaler = datasets.get_data_scaler(config)
        img = scaler(img)
    return img


def postprocess(img, args):
    if args.model == "rectified":
        if args.dataset == "celebahq":
            config = get_config_celebahq()
        elif args.dataset == "afhq_cat":
            config = get_config_afhq_cat()
            # inverse_scaler = datasets.get_data_inverse_scaler(config)
        img = (img + 1.) / 2.
    if args.model == "ot" or args.model == "gradient_step" or args.model == "diffusion":
        if args.dataset == "afhq_cat":
            img = (img + 1) / 2
        else:
            invTrans = v2.Normalize(
                mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5], std=[1./0.5, 1./0.5, 1./0.5])
            img = invTrans(img)
    return img


def save_memory_use(dict_mem,  args):
    memory_filename = os.path.join(
        args.save_path_ip, f'memory_stats.txt')
    with open(memory_filename, "a") as f:
        f.write(str(dict_mem) + '\n')


def save_time_use(dict_mem,  args):
    time_filename = os.path.join(
        args.save_path_ip, f'time_stats.txt')
    with open(time_filename, "a") as f:
        f.write(str(dict_mem) + '\n')


def compute_psnr(clean_img, noisy_img, rec_img, args, H_adj, iter='final'):

    # Ensure images are in the appropriate range and format for PSNR calculation
    clean_img = postprocess(clean_img.clone(), args)
    noisy_img = postprocess(noisy_img.clone(), args)
    rec_img = postprocess(rec_img.clone(), args)
    H_adj_noisy_img = postprocess(H_adj(noisy_img), args)

    clean_img = clean_img.permute(0, 2, 3, 1).cpu().data.numpy()
    if args.problem == 'superresolution' or args.problem == 'superresolution_bicubic':
        noisy_img = H_adj_noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()
    else:
        noisy_img = noisy_img.permute(0, 2, 3, 1).cpu().data.numpy()
    rec_img = rec_img.permute(0, 2, 3, 1).cpu().data.numpy()

    # Compute PSNR values
    psnr_rec = PSNR(clean_img, rec_img, data_range=1.0)
    psnr_noisy = PSNR(clean_img, noisy_img, data_range=1.0)

    # Save PSNR restored values
    rec_filename = os.path.join(
        args.save_path_ip, f'psnr_rec_batch{args.batch}.txt')

    with open(rec_filename, 'a') as file:
        file.write(f'{iter} {psnr_rec}\n')

    # Save PSNR noisy values
    noisy_filename = os.path.join(
        args.save_path_ip, f'psnr_noisy_batch{args.batch}.txt')

    with open(noisy_filename, 'a') as file:
        file.write(f'{iter} {psnr_noisy}\n')


def compute_average_psnr(args):
    # Compute the average PSNR values
    dict_pnsr = {}
    for word in ['rec', 'noisy']:
        psnr_by_iteration = defaultdict(list)

        for batch in range(args.max_batch):
            filename = os.path.join(
                args.save_path_ip, f'psnr_{word}_batch{batch}.txt')

            with open(filename, 'r') as f:
                for line in f:
                    iteration, psnr = map(float, line.strip().split())
                    psnr_by_iteration[int(iteration)].append(psnr)
        psnr_averages = {iteration: np.mean(
            psnrs) for iteration, psnrs in psnr_by_iteration.items()}

        avg_filename = os.path.join(
            args.save_path_ip, f'psnr_{word}_average.txt'
        )

        with open(avg_filename, 'a') as f:
            for iteration, avg_psnr in sorted(psnr_averages.items()):
                f.write(f'{iteration} {avg_psnr:.4f}\n')

        with open(avg_filename, 'r') as file:
            lines = file.readlines()
            psnr_values = [float(line.split()[1]) for line in lines]
            dict_pnsr[word] = psnr_values[-1]

    # Save final PSNR values for a given config
    filename = f'psnr_rec.png'
    with open(os.path.join(args.save_path, 'final_psnr.txt'), 'a') as file:

        # header if file is empty
        if os.stat(os.path.join(args.save_path, 'final_psnr.txt')).st_size == 0:
            file.write('psnr_rec ')
            file.write('psnr_noisy ')
            for key in args.dict_cfg_method.keys():
                file.write(f'{key} ')
            file.write('\n')

        file.write(f"{dict_pnsr['rec']} ")
        file.write(f"{dict_pnsr['noisy']} ")
        for value in args.dict_cfg_method.values():
            file.write(f'{value} ')
        file.write('\n')


def compute_lpips(clean_img, noisy_img, rec_img, args, H_adj, iter='final'):
    # Ensure images are in the appropriate range and format for LPIPS calculation
    clean_img = postprocess(clean_img.clone(), args)
    noisy_img = postprocess(noisy_img.clone(), args)
    rec_img = postprocess(rec_img.clone(), args)
    H_adj_noisy_img = postprocess(H_adj(noisy_img), args)

    # Permute images to NCHW format and move to the correct device
    clean_img = clean_img.to(DEVICE)
    rec_img = rec_img.to(DEVICE)

    if args.problem in ['superresolution', 'superresolution_bicubic']:
        noisy_img = H_adj_noisy_img.to(DEVICE)
    else:
        noisy_img = noisy_img.to(DEVICE)

    # Ensure images are in the expected format (N, C, H, W) and range [-1, 1] for LPIPS
    clean_img = 2 * clean_img - 1
    rec_img = 2 * rec_img - 1
    noisy_img = 2 * noisy_img - 1

    # Compute LPIPS values
    lpips_rec = loss_fn_alex(clean_img, rec_img, normalize=True).mean().item()
    lpips_noisy = loss_fn_alex(
        clean_img, noisy_img, normalize=True).mean().item()

    # Save LPIPS restored values
    rec_filename = os.path.join(
        args.save_path_ip, f'lpips_rec_batch{args.batch}.txt')

    with open(rec_filename, 'a') as file:
        file.write(f'{iter} {lpips_rec}\n')

    # Save LPIPS noisy values
    noisy_filename = os.path.join(
        args.save_path_ip, f'lpips_noisy_batch{args.batch}.txt')

    with open(noisy_filename, 'a') as file:
        file.write(f'{iter} {lpips_noisy}\n')


def compute_average_lpips(args):
    # Compute the average LPIPS values
    dict_lpips = {}
    for word in ['rec', 'noisy']:
        lpips_by_iteration = defaultdict(list)

        # Iterate over batches to collect LPIPS scores
        for batch in range(args.max_batch):
            filename = os.path.join(
                args.save_path_ip, f'lpips_{word}_batch{batch}.txt')

            with open(filename, 'r') as f:
                for line in f:
                    iteration, lpips = map(float, line.strip().split())
                    lpips_by_iteration[int(iteration)].append(lpips)

        # Calculate the average LPIPS score for each iteration
        lpips_averages = {iteration: np.mean(
            lpips_scores) for iteration, lpips_scores in lpips_by_iteration.items()}

        # Save the average LPIPS values to a text file
        avg_filename = os.path.join(
            args.save_path_ip, f'lpips_{word}_average.txt'
        )

        with open(avg_filename, 'a') as f:
            for iteration, avg_lpips in sorted(lpips_averages.items()):
                f.write(f'{iteration} {avg_lpips:.4f}\n')

        # Extract the last recorded LPIPS value for final comparison
        with open(avg_filename, 'r') as file:
            lines = file.readlines()
            lpips_values = [float(line.split()[1]) for line in lines]
            dict_lpips[word] = lpips_values[-1]

    # Save final LPIPS values for the given configuration
    with open(os.path.join(args.save_path, 'final_lpips.txt'), 'a') as file:
        # Write header if the file is empty
        if os.stat(os.path.join(args.save_path, 'final_lpips.txt')).st_size == 0:
            file.write('lpips_rec ')
            file.write('lpips_noisy ')
            for key in args.dict_cfg_method.keys():
                file.write(f'{key} ')
            file.write('\n')

        # Write the final average LPIPS scores and method configuration values
        file.write(f"{dict_lpips['rec']} ")
        file.write(f"{dict_lpips['noisy']} ")
        for value in args.dict_cfg_method.values():
            file.write(f'{value} ')
        file.write('\n')


def compute_ssim(clean_img, noisy_img, rec_img, args, H_adj, iter='final'):
    # Ensure images are in the appropriate range and format for SSIM calculation
    H_adj_noisy_img = postprocess(
        H_adj(noisy_img), args).cpu()
    clean_img = postprocess(clean_img.clone(), args).cpu()
    noisy_img = postprocess(noisy_img.clone(), args).cpu()
    rec_img = postprocess(rec_img.clone(), args).cpu()

    # Convert images to the appropriate format for SSIM calculation
    if args.problem == 'superresolution' or args.problem == 'superresolution_bicubic':
        noisy_img = H_adj_noisy_img
    else:
        noisy_img = noisy_img

    # Initialize SSIM metric for restored and noisy images
    ssim_metric = SSIM(data_range=1.0)
    ssim_metric_noisy = SSIM(data_range=1.0)

    # Compute SSIM values
    ssim_metric.update((rec_img, clean_img))
    ssim_rec = ssim_metric.compute()
    ssim_metric_noisy.update((noisy_img, clean_img))
    ssim_noisy = ssim_metric_noisy.compute()

    # Save SSIM restored values
    rec_filename = os.path.join(
        args.save_path_ip, f'ssim_rec_batch{args.batch}.txt')

    with open(rec_filename, 'a') as file:
        file.write(f'{iter} {ssim_rec}\n')

    # Save SSIM noisy values
    noisy_filename = os.path.join(
        args.save_path_ip, f'ssim_noisy_batch{args.batch}.txt')

    with open(noisy_filename, 'a') as file:
        file.write(f'{iter} {ssim_noisy}\n')


def compute_average_ssim(args):
    # Compute the average SSIM values
    dict_ssim = {}
    for word in ['rec', 'noisy']:
        ssim_by_iteration = defaultdict(list)

        for batch in range(args.max_batch):
            filename = os.path.join(
                args.save_path_ip, f'ssim_{word}_batch{batch}.txt')

            with open(filename, 'r') as f:
                for line in f:
                    iteration, ssim = map(float, line.strip().split())
                    ssim_by_iteration[int(iteration)].append(ssim)
        ssim_averages = {iteration: np.mean(
            ssims) for iteration, ssims in ssim_by_iteration.items()}

        avg_filename = os.path.join(
            args.save_path_ip, f'ssim_{word}_average.txt'
        )

        with open(avg_filename, 'a') as f:
            for iteration, avg_ssim in sorted(ssim_averages.items()):
                f.write(f'{iteration} {avg_ssim:.4f}\n')

        with open(avg_filename, 'r') as file:
            lines = file.readlines()
            ssim_values = [float(line.split()[1]) for line in lines]
            dict_ssim[word] = ssim_values[-1]

    # Save final SSIM values for a given config
    with open(os.path.join(args.save_path, 'final_ssim.txt'), 'a') as file:
        # header if file is empty
        if os.stat(os.path.join(args.save_path, 'final_ssim.txt')).st_size == 0:
            file.write('ssim_rec ')
            file.write('ssim_noisy ')
            for key in args.dict_cfg_method.keys():
                file.write(f'{key} ')
            file.write('\n')

        file.write(f'{dict_ssim["rec"]} ')
        file.write(f'{dict_ssim["noisy"]} ')
        for value in args.dict_cfg_method.values():
            file.write(f'{value} ')
        file.write('\n')


def compute_average_time(args):
    array_times = torch.zeros(args.max_batch)
    filename = os.path.join(
        args.save_path_ip, 'time_stats.txt')
    for batch in range(args.max_batch):
        with open(filename, 'r') as file:
            for line in file:
                # Convert the string representation of the dictionary to an actual dictionary
                data = ast.literal_eval(line.strip())
                # Check if the current batch number matches the one we're looking for
                if data['batch'] == batch:
                    array_times[batch] = data['time_per_batch']
                    break
    avg_filename = os.path.join(args.save_path_ip, f'time_average.txt')

    with open(avg_filename, 'a') as f:
        f.write(f'average time: {array_times.mean().item():.4f}\n')


def compute_average_memory(args):
    array_max_mem = torch.zeros(args.max_batch)
    filename = os.path.join(
        args.save_path_ip, 'memory_stats.txt')
    for batch in range(args.max_batch):
        with open(filename, 'r') as file:
            for line in file:
                # Convert the string representation of the dictionary to an actual dictionary
                data = ast.literal_eval(line.strip())
                # Check if the current batch number matches the one we're looking for
                if data['batch'] == batch:
                    array_max_mem[batch] = data['max_allocated']
                    break
    avg_filename = os.path.join(args.save_path_ip, f'max_memory_average.txt')

    with open(avg_filename, 'a') as f:
        f.write(f'average mem: {array_max_mem.mean().item():.4f}\n')


class MaskGenerator():
    # copied from https://www.kaggle.com/code/tom99763/inpainting-mask-generator
    def __init__(self, height, width, channels=3, rand_seed=None, filepath=None):
        self.height = height
        self.width = width
        self.channels = channels
        self.filepath = filepath
        self.mask_files = []

        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if any(
                filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(
                len(self.mask_files), self.filepath))

        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self):
        # print("height, width, channels", self.height, self.width, self.channels)
        img = np.zeros((self.height, self.width, self.channels), np.uint8)
        size = int((self.width + self.height) * 0.08)

        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        # Draw random lines
        for _ in range(10):
            x1, x2 = randint(self.width//2 - 30, self.width//2 +
                             30), randint(self.width//2 - 30, self.width//2 + 30)
            y1, y2 = randint(self.height//2 - 30, self.height//2 +
                             30), randint(self.height//2 - 30, self.height//2 + 30)
            thickness = randint(8, size)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)
        return 1 - img

    def _load_mask(self, rotation=True, dilation=True, cropping=True):
        mask = cv2.imread(os.path.join(self.filepath, np.random.choice(
            self.mask_files, 1, replace=False)[0]))

        if rotation:
            rand = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D(
                (mask.shape[1] / 2, mask.shape[0] / 2), rand, 1.5)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        if dilation:
            rand = np.random.randint(5, 47)
            kernel = np.ones((rand, rand), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        if cropping:
            x = np.random.randint(0, mask.shape[1] - self.width)
            y = np.random.randint(0, mask.shape[0] - self.height)
            mask = mask[y:y + self.height, x:x + self.width]

        return (mask > 1).astype(np.uint8)

    def sample(self, random_seed=None):
        if random_seed:
            seed(random_seed)
        if self.filepath and len(self.mask_files) > 0:
            return self._load_mask()
        else:
            return self._generate_mask()


def GMRES(A,                # Linear operator, matrix or function
          b,                # RHS of the linear system in which the first half has the same shape as grad_gx, the second half has the same shape as grad_fy
          x0=None,          # initial guess, tuple has the same shape as b
          max_iter=None,    # maximum number of GMRES iterations
          tol=1e-6,         # relative tolerance
          atol=1e-6,        # absolute tolerance
          track=False):     # If True, track the residual error of each iteration
    '''
    Return:
        sol: solution
        (j, err_history):
            j is the number of iterations used to achieve the target accuracy;
            err_history is a list of relative residual error at each iteration if track=True, empty list otherwise.
    '''
    if isinstance(A, torch.Tensor):
        Avp = partial(Mvp, A)
    elif hasattr(A, '__call__'):
        Avp = A
    else:
        raise ValueError('A must be a function or matrix')

    bnorm = torch.norm(b)

    if max_iter == 0 or bnorm < 1e-8:
        return b

    if max_iter is None:
        max_iter = b.shape[0]

    if x0 is None:
        x0 = torch.zeros_like(b)
        r0 = b
    else:
        r0 = b - Avp(x0)

    new_v, rnorm = _safe_normalize(r0)
    # initial guess residual
    beta = torch.zeros(max_iter + 1, device=b.device)
    beta[0] = rnorm
    err_history = []
    if track:
        err_history.append((rnorm / bnorm).item())

    V = []
    V.append(new_v)
    H = torch.zeros((max_iter + 1, max_iter + 1), device=b.device)
    cs = torch.zeros(max_iter, device=b.device)  # cosine values at each step
    ss = torch.zeros(max_iter, device=b.device)  # sine values at each step

    for j in range(max_iter):
        p = Avp(V[j])
        # Arnoldi iteration to get the j+1 th basis
        new_v = arnoldi(p, V, H, j + 1)
        V.append(new_v)

        H, cs, ss = apply_given_rotation(H, cs, ss, j)
        _check_nan(cs, f'{j}-th cosine contains NaN')
        _check_nan(ss, f'{j}-th sine contains NaN')
        beta[j + 1] = ss[j] * beta[j]
        beta[j] = cs[j] * beta[j]
        residual = torch.abs(beta[j + 1])
        if track:
            err_history.append((residual / bnorm).item())
        if residual < tol * bnorm or residual < atol:
            break
    y = torch.linalg.solve_triangular(
        H[0:j + 1, 0:j + 1], beta[0:j + 1].unsqueeze(-1), upper=True)  # j x j
    V = torch.stack(V[:-1], dim=0)
    sol = x0 + V.T @ y.squeeze(-1)
    return sol, (j, err_history)


def _check_nan(vec, msg):
    if torch.isnan(vec).any():
        raise ValueError(msg)


def _safe_normalize(x, threshold=None):
    norm = torch.norm(x)
    if threshold is None:
        threshold = torch.finfo(norm.dtype).eps
    normalized_x = x / norm if norm > threshold else torch.zeros_like(x)
    return normalized_x, norm


def Mvp(A, vec):
    return A @ vec


def arnoldi(vec,    # Matrix vector product
            V,      # List of existing basis
            H,      # H matrix
            j):     # number of basis
    '''
    Arnoldi iteration to find the j th l2-orthonormal vector
    compute the j-1 th column of Hessenberg matrix
    '''
    _check_nan(vec, 'Matrix vector product is Nan')

    for i in range(j):
        H[i, j - 1] = torch.dot(vec, V[i])
        vec = vec - H[i, j-1] * V[i]
    new_v, vnorm = _safe_normalize(vec)
    H[j, j - 1] = vnorm
    return new_v


def cal_rotation(a, b):
    '''
    Args:
        a: element h in position j
        b: element h in position j+1
    Returns:
        cosine = a / \sqrt{a^2 + b^2}
        sine = - b / \sqrt{a^2 + b^2}
    '''
    c = torch.sqrt(a * a + b * b)
    return a / c, - b / c


def apply_given_rotation(H, cs, ss, j):
    '''
    Apply givens rotation to H columns
    :param H:
    :param cs:
    :param ss:
    :param j:
    :return:
    '''
    # apply previous rotation to the 0->j-1 columns
    for i in range(j):
        tmp = cs[i] * H[i, j] - ss[i] * H[i + 1, j]
        H[i + 1, j] = cs[i] * H[i+1, j] + ss[i] * H[i, j]
        H[i, j] = tmp
    cs[j], ss[j] = cal_rotation(H[j, j], H[j + 1, j])
    H[j, j] = cs[j] * H[j, j] - ss[j] * H[j + 1, j]
    H[j + 1, j] = 0
    return H, cs, ss


def get_save_path_ip(dict_cfg_method):
    """
    dict_cfg_method contains keys and values of the method.
    Return path composed of key1=value1/key2=value2/.../keyN=valueN
    """
    path = ""
    for key, value in dict_cfg_method.items():
        path = os.path.join(path, f"{key}={value}")
    return path


# Function to create the downsampling matrix
def create_downsampling_matrix(H, W, sf, device):
    assert H % sf == 0 and W % sf == 0, "Image dimensions must be divisible by sf"

    H_ds, W_ds = H // sf, W // sf  # Downsampled dimensions
    N = H * W  # Total number of pixels in the original image
    M = H_ds * W_ds  # Total number of pixels in the downsampled image

    # Initialize downsampling matrix of size (M, N)
    downsample_matrix = torch.zeros((M, N), device=device)

    # Fill the matrix with 1s at positions corresponding to downsampling
    for i in range(H_ds):
        for j in range(W_ds):
            # The index in the downsampled matrix
            downsampled_idx = i * W_ds + j

            # The corresponding index in the original flattened matrix
            original_idx = (i * sf * W) + (j * sf)

            # Set the value to 1 to perform downsampling
            downsample_matrix[downsampled_idx, original_idx] = 1

    return downsample_matrix
