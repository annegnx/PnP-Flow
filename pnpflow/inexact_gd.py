import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.distributions as D
from tqdm import trange
import matplotlib.animation as anim
import matplotlib.colors as mc

path_results = '../results/inexact/'

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

#TODO: (i) check with a point far from the mean (ii) effect of subiterations ?? (too simple example : 1 grad step = full minimization)

d, n, m = 2, 200, 100 

sigma_0 = 1.0
x0 = torch.randn((n, d)) * sigma_0
mus = torch.tensor([
    [20.0, 0.0],
    [20.0, 20.0],
    [20.0, -20.0],
    [32.0, 0.0],
    ])
sigma_1 = 1.0
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

def grad_datafit(x, y):
    return H_adj(H(x) - y)

def log_pt_gmm(x, t):
    sigma_t = ((t * sigma_1) ** 2 + (1 - t) ** 2) ** 0.5
    
    # log unnormalized weights: shape (n,)
    log_w = -((x.view(1, -1) - t * mus) ** 2).sum(-1) / (2 * sigma_t ** 2)
    
    # log normalizing constant of each Gaussian
    log_norm = -0.5 * mus.shape[-1] * torch.log(2 * torch.tensor(torch.pi) * sigma_t ** 2)
    
    # log(1/n * sum_i N(x; t*mu_i, sigma_t^2 I))
    # = log(1/n) + logsumexp(log_w + log_norm)
    return torch.log(torch.tensor(1.0 / mus.shape[0])) + torch.logsumexp(log_w + log_norm, dim=0)

def score_gmm(x, t):
    sigma_t = (t * sigma_1) ** 2 + (1 - t) ** 2
    
    # log unnormalized weights: shape (n,)
    log_w = -((x.view(1, -1) - t * mus) ** 2).sum(-1) / (2 * sigma_t ** 2)
    
    # numerically stable softmax weights
    log_w = log_w - log_w.max()  # subtract max for stability
    w = torch.exp(log_w)
    w = w / w.sum()  # normalize: shape (n,)
    
    # weighted mean of means: shape (d,)
    weighted_mus = (w.view(-1, 1) * mus).sum(0)  # shape (d,)
    
    return (t * weighted_mus - x) / sigma_t ** 2

def reg(x, t):
    at = - (1 - t) / (t)
    bt = - (1 - t) ** 2 / (t)
    return at * x + bt * score_gmm(x, t)

def objective_value_fun(x, t):
    F = 0.5 * ((H(x) - y) ** 2).sum()

    at = - (1 - t) / (2 * t)
    bt = - (1 - t) ** 2 / (t)
    R_t = at * (x ** 2).sum() + bt * log_pt_gmm(x, t)

    # return F.item()
    return (F + R_t).item()

def step_size(t):
    alpha = 2.0
    beta = 0.01
    # return (1 - t) ** beta
    return t ** alpha * (1 - t) ** beta

sigma_noise = 3.0
ground_truth = x1[np.random.randint(x1.size(0))]
y = H(ground_truth) + torch.randn(d) * sigma_noise

N = 10
delta = 1 / N
x = torch.randn_like(ground_truth)
for k in range(1, N + 1):
    t = k * delta

    gamma = (1 - t) ** (0.8)#1.0 / (k + 1)
    eta = 1.0 * (1 - t) ** 2 / t

    x_ref = x.clone()

    # z = x - gamma * grad_datafit(x, y)
    # x_new = z + eta * score_gmm(t * z, t)
    x_new = z = x - gamma * grad_datafit(x, y) + eta * score_gmm(t * x, t)


    if True:
        fig = plt.figure()
        plt.scatter(x0.numpy()[:, 0], x0.numpy()[:, 1], color='#0084af', alpha=0.2)
        plt.scatter(x1.numpy()[:, 0], x1.numpy()[:, 1], color='#006a00', alpha=0.2)
        plt.scatter((t * x1).numpy()[:, 0], (t * x1).numpy()[:, 1], color="#6a1500", alpha=0.1)
        plt.scatter(ground_truth.numpy()[0], ground_truth.numpy()[1], color='red', s=150, marker='*', label='ground truth')
        plt.scatter(y.numpy()[0], y.numpy()[1], color='orange', s=150, marker='X', label='measurement')
        plt.scatter(x_ref.numpy()[0], x_ref.numpy()[1], color='black', s=70, marker='o', label='latent')
        # plt.scatter(tmp_tilde.numpy()[0], tmp_tilde.numpy()[1], color='green', s=70, marker='o', label='z_t_tilde')
        # plt.scatter(eps.numpy()[0], eps.numpy()[1], color='green', s=70, marker='o', label='eps')
        # plt.scatter(tmp_mean.numpy()[0], tmp_mean.numpy()[1], color='yellow', s=70, marker='o', label='x_star')

        x_ref_np = x_ref.numpy()
        z_np = z.numpy()
        x_new_np = x_new.numpy()
        grad_step = - gamma * grad_datafit(x_ref, y).numpy()
        reg_step = (x_new - z).numpy()
        full_step = (- gamma * grad_datafit(x_ref, y) + eta * score_gmm(t * x_ref, t)).numpy()
        plt.arrow(x_ref_np[0], x_ref_np[1],
                x_new_np[0] - x_ref_np[0],
                x_new_np[1] - x_ref_np[1],
                color='#ff0419',
                head_width=0.4,
                linewidth=2.5,
                length_includes_head=True, label="PnP Step")
        plt.arrow(x_ref_np[0], x_ref_np[1],
                grad_step[0],
                grad_step[1],
                color='#0003bd',
                head_width=0.4,
                linewidth=2.5,
                length_includes_head=True, label="Grad Step")
        plt.arrow(z_np[0], z_np[1],
                reg_step[0],
                reg_step[1],
                color='#ffa927',
                head_width=0.4,
                linewidth=2.5,
                length_includes_head=True, label="Reg step")
        plt.arrow(x_ref_np[0], x_ref_np[1],
                full_step[0],
                full_step[1],
                color="#ff27ed",
                head_width=0.4,
                linewidth=2.5,
                length_includes_head=True, label="Single step")
        
        plt.legend()
        plt.xticks()
        plt.yticks()
        plt.axis('off')
        fig.savefig(path_results + f'final_{t:2.2f}_toy.pdf')
        plt.close()
    x = x_new
print("final distance   : ", ((x - ground_truth) ** 2).sum().sqrt().item())
print("original distance: ", ((y - ground_truth) ** 2).sum().sqrt().item())
