def forward_diffusion(x0, t, device):
    T = 1000
    noise_scheduler = torch.linspace(10**-4, 0.02, T)

    alpha = (1-noise_scheduler).to(device)
    alpha_bar = torch.cumprod(alpha, dim=0).to(device)

    noise = torch.randn_like(x0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t][:, None, None, None])
    sqrt_one_minus = torch.sqrt(1-alpha_bar[t][:, None, None, None])
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    return xt, noise