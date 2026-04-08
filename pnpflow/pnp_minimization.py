import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.distributions as D
from tqdm import trange
import matplotlib.animation as anim
import matplotlib.colors as mc

path_results = '../results/reg/'

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
# mus = torch.tensor([[10.0, 0.0]])
mus = torch.tensor([
    [20.0, 0.0],
    [20.0, 20.0],
    [20.0, -20.0],
    [32.0, 0.0],
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

def moreau_grad_enveloppe(x, t):
    at = - (1 - t) / (2 * t)
    bt = - (1 - t) ** 2 / (t)
    ct = - (1 - t) ** 3 / (2 * t ** 2)
    dt = - (1 - t) ** 4 / (2 * t ** 2)
    return at * x + bt * score_gmm(x, t)

def denoiser(x, t):
    at = 1 / t
    bt = (1 - t) ** 2 / t
    return at * x + bt * score_gmm(x, t)

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

d_list = []
# n_steps = [10, 25, 50, 100, 250,  500, 1000]
n_steps = [10]
for N in n_steps:
    delta = 1 / N
    num_samples = 100
    sub_iter = 1
    stoppage_time = 1.0 #0.5
    stoppage_iter = int(stoppage_time * N)

    z = torch.randn(d) * sigma_0
    z_list = [z]
    objective_values = []
    for iteration in range(1, N + 1):
        t = delta * iteration
        for k in range(sub_iter if iteration == stoppage_iter or stoppage_iter < 0 else 1):
            tmp = z - step_size(t) * grad_datafit(z, y, sigma_noise)

            exact_score = score_gmm(tmp, t)
            exact_scaled_score = score_gmm(t * tmp, t)
            score_err = []
            scaled_score_err = []

            M = 1
            for _ in range(M):
                tmp_mean = torch.zeros_like(tmp)
                for i in range(num_samples):
                    eps = torch.randn_like(tmp)
                    tmp_tilde = t * tmp + (1 - t) * eps
                    alpha = step_size(t) / t
                    tmp_mean += (1 - alpha) * tmp + alpha * denoiser(tmp_tilde, t) - alpha * (1 - t) / t * eps

                    # approx_score = tmp_mean / (i + 1) - tmp
                    # score_err.append((approx_score - exact_score).pow(2).sum().sqrt())
                    # scaled_score_err.append((approx_score - exact_scaled_score).pow(2).sum().sqrt())
                tmp = tmp_mean / num_samples
            next_z = tmp
            # tmp_tilde = next_z = tmp + step_size(t) * score_gmm(tmp, t)

            z_list.append(next_z.clone())
            if iteration == stoppage_iter:
                objective_values.append(objective_value_fun(next_z, t))
            if sub_iter > 1 and k < sub_iter - 1:
                z = next_z
        
        if False:
            fig = plt.figure()
            plt.plot(score_err, label='true score error')
            plt.plot(scaled_score_err, label='scaled score error')
            plt.title(f"Score error at t={t}")
            print(f"t={t:.2f} exact error", score_err[-1])
            print(f"t={t:.2f} scaled error", scaled_score_err[-1])
            plt.grid()
            plt.legend()
            plt.savefig(path_results + f'score_error_{t:.2f}.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()

        if iteration == stoppage_iter and sub_iter > 1 and True:
            print(objective_values)
            fig = plt.figure()
            plt.plot(objective_values)
            plt.xlabel(f'#Sub Iterations at t={stoppage_time}')
            plt.ylabel('Objective value F + R_t')
            plt.xticks()
            plt.yticks()
            plt.grid()
            fig.savefig(path_results + f'objective_minization_t={stoppage_time:.2f}.pdf')
            plt.close()
            break
        if False:
            fig = plt.figure()
            plt.scatter(x0.numpy()[:, 0], x0.numpy()[:, 1], color='#0084af', alpha=0.2)
            plt.scatter(x1.numpy()[:, 0], x1.numpy()[:, 1], color='#006a00', alpha=0.2)
            plt.scatter((t * x1).numpy()[:, 0], (t * x1).numpy()[:, 1], color="#6a1500", alpha=0.1)
            plt.scatter(ground_truth.numpy()[0], ground_truth.numpy()[1], color='red', s=150, marker='*', label='ground truth')
            plt.scatter(y.numpy()[0], y.numpy()[1], color='orange', s=150, marker='X', label='measurement')
            plt.scatter(z.numpy()[0], z.numpy()[1], color='black', s=70, marker='o', label='latent')
            plt.scatter(tmp_tilde.numpy()[0], tmp_tilde.numpy()[1], color='green', s=70, marker='o', label='z_t_tilde')
            plt.scatter(eps.numpy()[0], eps.numpy()[1], color='green', s=70, marker='o', label='eps')
            plt.scatter(tmp_mean.numpy()[0], tmp_mean.numpy()[1], color='yellow', s=70, marker='o', label='x_star')

            z_np = z.numpy()
            tmp_np = tmp_tilde.numpy()
            next_z_np = next_z.numpy()
            grad_step = - step_size(t) * grad_datafit(z, y, sigma_noise).numpy()
            reg_step = (denoiser(tmp_tilde, t) - tmp - (1 - t) / t * eps).numpy()
            plt.arrow(z_np[0], z_np[1],
                    next_z_np[0] - z_np[0],
                    next_z_np[1] - z_np[1],
                    color='#ff0419',
                    head_width=0.4,
                    linewidth=2.5,
                    length_includes_head=True, label="PnP Step")
            plt.arrow(z_np[0], z_np[1],
                    grad_step[0],
                    grad_step[1],
                    color='#0003bd',
                    head_width=0.4,
                    linewidth=2.5,
                    length_includes_head=True, label="Grad Step")
            plt.arrow(tmp_np[0], tmp_np[1],
                    reg_step[0],
                    reg_step[1],
                    color='#ffa927',
                    head_width=0.4,
                    linewidth=2.5,
                    length_includes_head=True, label="Reg step")
            

            plt.legend()
            plt.xticks()
            plt.yticks()
            plt.axis('off')
            fig.savefig(path_results + f'final_{t:2.2f}_toy.pdf')
            plt.close()
        z = next_z
    if ((ground_truth - z) ** 2).sum().isinf().any():
        d_list.append(0)
        break
    d_list.append(((ground_truth - z) ** 2).sum().sqrt().item())
print("final distance", ((ground_truth - z) ** 2).sum().sqrt().item())

fig = plt.figure()
plt.scatter(x0.numpy()[:, 0], x0.numpy()[:, 1], color='#0084af', alpha=0.2)
plt.scatter(x1.numpy()[:, 0], x1.numpy()[:, 1], color='#006a00', alpha=0.2)
plt.scatter(ground_truth.numpy()[0], ground_truth.numpy()[1], color='red', s=150, marker='*', label='ground truth')
plt.scatter(y.numpy()[0], y.numpy()[1], color='orange', s=150, marker='X', label='measurement')

x_list = [z[0] for z in z_list]
y_list = [z[1] for z in z_list]
p = len(z_list)
plt.plot(x_list[:p//2 + 1], y_list[:p//2 + 1], c='black', linestyle="dotted")
plt.plot(x_list[p//2:], y_list[p//2:], c='black')
plt.scatter(x_list[-1], y_list[-1], s=100, c='black')

plt.legend()
plt.xticks()
plt.yticks()
# plt.xlim(-5, 70)
# plt.ylim(-30, 50)
plt.axis('off')
fig.savefig(path_results + f'final_reg.pdf', bbox_inches='tight', pad_inches=0)
plt.close()

if True:
    fig = plt.figure()
    plt.plot(n_steps, d_list, c='blue', label='||z - x||^2')
    plt.plot(n_steps, [((ground_truth - y) ** 2).sum().item()] * len(d_list), c='red', label='||x - y||^2')
    plt.xticks()
    plt.yticks()
    # plt.ylim(min(d_list)*0.9, max(d_list)*1.1)
    plt.grid()
    plt.xlabel('#Total Iterations')
    plt.ylabel('Squared distance || ground_truth - z ||^2')
    fig.savefig(path_results + f'distance_plot.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(d_list)
    print(((ground_truth - y) ** 2).sum().sqrt().item())