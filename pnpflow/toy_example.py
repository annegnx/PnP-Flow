import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.distributions as D
from tqdm import trange
import matplotlib.animation as anim
import matplotlib.colors as mc


path_result = '../results/toy/'


def set_seed(seed: int):

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If you are using CUDA (GPU), set seed for PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multiple GPUs

    # Ensuring that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(5)


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


mix = D.Categorical(torch.ones(1,))
means = torch.tensor(
    np.array([(7 * np.cos(k * 2 * np.pi / 8), 7 * np.sin(k * 2 * np.pi / 8)) for k in range(1)]))

comp = D.Independent(D.Normal(means, 0.5 * torch.ones(1, 2)), 1)
gmm_8 = D.mixture_same_family.MixtureSameFamily(mix, comp)

latent = D.Normal(torch.zeros(2), torch.ones(2))


sigma = 0.1
dim = 2
batch_size = 128
model = MLP(dim=dim, time_varying=True)
optimizer = torch.optim.Adam(model.parameters())

steps = 1

# train
for k in trange(1000):
    optimizer.zero_grad()

    x0 = latent.sample_n(batch_size).float()
    x1 = gmm_8.sample_n(batch_size).float()

    t1 = torch.rand(x1.shape[0])

    a, b = np.ones(len(x0)) / len(x0), np.ones(len(x0)) / len(x0)

    M = ot.dist(x0.view(len(x0), -1).cpu().data.numpy(),
                x1.view(len(x1), -1).cpu().data.numpy())
    plan = ot.emd(a, b, M)
    # copied from alex tong code
    p = plan.flatten()
    p = p / p.sum()

    choices = np.random.choice(
        plan.shape[0] * plan.shape[1], p=p, size=len(x0), replace=True)
    i, j = np.divmod(choices, plan.shape[1])
    x0 = x0[i]
    x1 = x1[j]

    xt = t1[:, None] * x1 + (1 - t1[:, None]) * x0

    v_pred = model(torch.cat([xt, t1[:, None]], dim=-1))
    loss = torch.sum((v_pred - (x1 - x0))**2) / x0.shape[0]

    loss.backward()
    optimizer.step()
    if k % 1000 == 0:
        print(loss.item())


# eval

x0 = latent.sample_n(300).float()
x1 = gmm_8.sample_n(300).float()

sigma = 1.5
init = latent.sample().reshape((1, 2)).float()
y = torch.ones((1, 2)).float()
ground_truth = gmm_8.sample().reshape((1, 2)).float()
y = ground_truth + sigma * torch.randn_like(ground_truth)


def denoiser(x, t):
    t_ = t * torch.ones(1)[:, None]
    v = model(
        torch.cat([x, t_], dim=-1))
    return x + (1-t_) * v


def grad_datafit(x, y):
    return (x-y) / sigma**2


x = init.clone()


for i, t in enumerate(np.linspace(0, 1, 10)):

    fig = plt.figure()

    plt.scatter(x0[:, 0], x0[:, 1], color='#0084af',
                alpha=0.2,  label="latent")
    plt.scatter(x1[:, 0], x1[:, 1], color='#006a00', alpha=0.2, label="data")
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1],
                color='#0a1045', s=150, label='ground truth', marker='*')
    plt.scatter(y[:, 0], y[:, 1], color='#0a1045',
                s=150, label='y', marker='X')

    x_np = x.detach().numpy()

    plt.scatter(x_np[:, 0], x_np[:, 1], color='black', s=20,
                marker='o')

    grad_step = x - (1-t) * sigma**2 * grad_datafit(x,  y)

    grad_step_np = grad_step.detach().numpy()
    x_np = x.detach().numpy()
    plt.scatter(grad_step_np[0, 0], grad_step_np[0, 1],
                color='black', s=20, marker='o')

    # Step 4: Draw an arrow between the two points
    plt.arrow(x_np[0, 0], x_np[0, 1],
              grad_step_np[0, 0] - x_np[0, 0],
              grad_step_np[0, 1] - x_np[0, 1],
              color='#0003bd',
              head_width=0.4,
              linewidth=2.5,
              length_includes_head=True, label="Gradient Step")

    z = t * grad_step + (1-t) * torch.randn_like(grad_step)
    z_np = z.detach().numpy()

    plt.scatter(z_np[0, 0], z_np[0, 1],
                color='black', s=20,  marker='o')

    plt.arrow(grad_step_np[0, 0], grad_step_np[0, 1],
              z_np[0, 0] - grad_step_np[0, 0],
              z_np[0, 1] - grad_step_np[0, 1],
              color='#ffa927',
              head_width=0.4,
              linewidth=2.5,
              length_includes_head=True,  label="Interpolation Step")

    x = denoiser(z, t)
    x_np = x.detach().numpy()

    plt.scatter(x_np[0, 0], x_np[0, 1],
                color='black', s=10,  marker='o')

    # Step 4: Draw an arrow between the two points
    plt.arrow(z_np[0, 0], z_np[0, 1],
              x_np[0, 0] - z_np[0, 0],
              x_np[0, 1] - z_np[0, 1],
              color='#ff0419',
              head_width=0.4,
              linewidth=2.5,
              length_includes_head=True, label="PnP Step")

    plt.legend()
    plt.xticks()
    plt.yticks()
    plt.axis('off')
    fig.savefig(path_result + f'final_{t:2.2f}_toy.pdf')
plt.scatter(x_np[0, 0], x_np[0, 1],
            color='#fb8b24', s=150,  marker='o', label="solution")
plt.title(f"t{t:2.2}")
fig.savefig(path_result + f'final_toy.pdf')
