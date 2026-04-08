import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from torch.func import hessian, vmap, grad, jacfwd
from functools import partial

path_results = '../results/convergence/'

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

# seed 2
set_seed(7)

d, n, m = 2, 200, 100 

sigma_0 = 1.0
x0 = torch.randn((n, d)) * sigma_0
# mus = torch.tensor([[10.0, 0.0]])
mus = torch.tensor([
    [20.0, 0.0],
    [20.0, 20.0],
    # [20.0, -20.0],
    # [32.0, 0.0],
    # [32.0, 20.0],
    # [32.0, -20.0],
    ])
sigma_1 = 1.0
# 1 gaussian
# x1 = torch.randn((m, d)) * sigma_1 + mu_1
# gmm
x1 = torch.randn((mus.size(0) * m,d)) * sigma_1 + mus.repeat_interleave(m, dim=0)

def H(x):
    return x
    theta = np.pi/5
    M = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], dtype=torch.float)
    return M @ x

def H_adj(x):
    return x
    theta = - np.pi/5
    M = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], dtype=torch.float)
    return M @ x

def grad_datafit(x, y, sigma_noise=5e-2):
    return H_adj(H(x) - y)


def log_pt_gmm(x, t, mus, sigma_1):
    """
    x: (batch, d)
    t: scalar
    returns: (batch,)
    """
    sigma_t_sq = (t * sigma_1) ** 2 + (1 - t) ** 2
    d = mus.shape[-1]
    n = mus.shape[0]

    # log unnormalized weights: (n, batch)
    diff = x.unsqueeze(0) - t * mus.unsqueeze(1)  # (n, batch, d)
    log_w = -diff.pow(2).sum(-1) / (2 * sigma_t_sq)  # (n, batch)

    # log normalizing constant
    log_norm = -0.5 * d * torch.log(2 * torch.tensor(torch.pi) * sigma_t_sq)

    # log(1/n * sum_i N(x; t*mu_i, sigma_t^2))
    return torch.logsumexp(log_w + log_norm, dim=0) - torch.log(torch.tensor(float(n)))

def score_gmm(x, t, mus, sigma_1):
    """
    x: (batch, d)
    t: scalar
    returns: (batch, d)
    """
    sigma_t_sq = (t * sigma_1) ** 2 + (1 - t) ** 2
    
    diff = x.unsqueeze(0) - t * mus.unsqueeze(1)  # (n, batch, d)
    log_w = -diff.pow(2).sum(-1) / (2 * sigma_t_sq)  # (n, batch)
    
    # stable softmax weights
    log_w = log_w - log_w.max(dim=0).values
    w = torch.exp(log_w)
    w = w / w.sum(dim=0, keepdim=True)  # (n, batch)
    
    # weighted mean of means: (batch, d)
    weighted_mus = (w.unsqueeze(-1) * mus.unsqueeze(1)).sum(0)  # (batch, d)
    
    return (t * weighted_mus - x) / sigma_t_sq


def laplacian_vector_field(f, x):
    """
    f: callable R^n -> R^m
    x: tensor of shape (B, n)
    returns: tensor of shape (B, m)
    """
    def laplacian_single(x):
        # x: shape (n,)
        # jacobian of f: shape (m, n)
        # jacobian of jacobian of f: shape (m, n, n)
        # trace over last two dims: shape (m,)
        hess = jacfwd(jacfwd(f))(x)  # (m, n, n)
        return torch.diagonal(hess, dim1=-2, dim2=-1).sum(-1)  # (m,)

    return vmap(laplacian_single)(x)

grid_size = 200
xx, yy = torch.meshgrid(
    torch.linspace(-5, 35, grid_size),
    torch.linspace(-25, 25, grid_size),
    indexing='ij'
)
x_grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (grid^2, 2)
for t in [0.25, 0.5, 0.75, 1.0]:
    err = []

    score_fn = partial(score_gmm, t=t, mus=mus, sigma_1=sigma_1)
    corrected_score_ref = score_fn(t * x_grid) #+ (1 - t)**2 / 2 * laplacian_vector_field(score_fn, t * x_grid)

    sigma_sample = 0.3

    n_eps = 50
    with torch.no_grad():
        scores = torch.zeros_like(x_grid)  # (grid^2, 2)
        for k in range(n_eps):
            eps = torch.randn_like(x_grid)  # (grid^2, 2)
            x_noisy = t * x_grid + (1 - t) * sigma_sample * eps  # (grid^2, 2)
            scores += score_gmm(x_noisy, t, mus, sigma_1)

            err.append(((scores/(k + 1) - corrected_score_ref) ** 2).sum(-1).mean().item())
        scores /= n_eps  # (grid^2, 2)

    
    fig = plt.figure(figsize=(6,6))
    plt.plot(err)
    plt.grid()
    plt.title(f'Error of noisy score at t={t:.2f}')
    fig.savefig(path_results + f'err_score_{t:.2f}.pdf')
    plt.close()