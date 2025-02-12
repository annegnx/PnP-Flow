from math import ceil
import pytest
from pathlib import Path
import gdown
import torch
from torch.utils.data import DataLoader, Dataset
import pnpflow.degradations


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device=torch.device("cpu")):
    output_path = "../model/celeba/gaussian/ot/"
    folder = Path(output_path)
    folder.mkdir(parents=True, exist_ok=True)
    drive_id = "1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6"
    celeba_url = f"https://drive.google.com/uc?id={drive_id}"
    gdown.download(celeba_url, output_path + "model_final.pt", quiet=False)

    model = pnpflow.models.UNet(input_channels=3,
                                input_height=128,
                                ch=32,
                                ch_mult=(1, 2, 4, 8),
                                num_res_blocks=6,
                                attn_resolutions=(16, 8),
                                resamp_with_conv=True,
                                )
    model.load_state_dict(torch.load(
        output_path + "model_final.pt"))
    return model.to(device)


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
    model = load_model(device)
    model.eval()
    forward = model(test_sample, torch.ones(
        len(test_sample), device=device) * 1.0)
    assert forward.shape == test_sample.shape
