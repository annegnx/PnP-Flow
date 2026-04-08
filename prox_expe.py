import torch

d = 100000
x = torch.randn(d)
y = torch.randn(d)
mask = torch.ones(d)
mask[d//4:d//2] = 0
H = torch.diag(mask)
H_adj = H


def prox_datafit(x, y, H, H_adj, step_size, lr=0.05, n_iter=100):
    z = x.clone()
    for i in range(n_iter):
        grad = (z - x) + step_size * H_adj @ (H @ z - y)
        z = (z - lr * grad)
        print(i, (grad ** 2).sum().item())
    return z

prox_datafit(x, y, H, H_adj, 1.0)