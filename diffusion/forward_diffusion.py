import torch

def forward_diffusion(x0, t, alpha_bar):
    alpha_bar = alpha_bar.to(x0.device)
    noise = torch.randn_like(x0)

    sqrt_alpha_bar = torch.sqrt(alpha_bar[t][:, None, None, None])
    sqrt_one_minus = torch.sqrt(1-alpha_bar[t][:, None, None, None])
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    return xt, noise