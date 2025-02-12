from math import ceil
import pytest
from pathlib import Path
import gdown
import torch
from torch.utils.data import DataLoader, Dataset
import pnpflow.degradations
from pnpflow.utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @pytest.mark.parametrize("device", device)
def test_degradation(device=torch.device("cpu")):
    test_sample = torch.ones((1, 3, 128, 128), device=device)
    p = 32
    degradation = pnpflow.degradations.BoxInpainting(p)
    y = degradation.H(test_sample)
    torch.testing.assert_close(
        y[:, :, 32:64, 32:64], torch.zeros((1, 3, 32, 32), device=device))


# @pytest.mark.parametrize("device", device)
def test_inference_model(device=torch.device("cpu")):
    test_sample = torch.ones((1, 3, 128, 128), device=device)
    model = pnpflow.models.UNet(input_channels=3,
                                input_height=128,
                                ch=32,
                                ch_mult=(1, 2, 4, 8),
                                num_res_blocks=6,
                                attn_resolutions=(16, 8),
                                resamp_with_conv=True,
                                )
    load_model(name_model="ot", model=model, state=None,
               download=True, dataset="celeba", device=device)
    model.eval()
    forward = model(test_sample, torch.ones(
        len(test_sample), device=device) * 1.0)
    assert forward.shape == test_sample.shape
