def reverse_diffusion(xt, t, label, alpha, alpha_bar):
    eps_theta = Unet(xt, t, label)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t])[:, None, None, None]
    x0_pred = (xt - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t

    beta_t = 1 - alpha[t]
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    alpha_bar_t_prev = alpha_bar[t-1] if t > 0 else torch.tensor(1.0, device=xt.device)

    coef_x0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

    mu_t = coef_x0 * x0_pred + coef_xt * xt
    sigma_t = torch.sqrt(beta_t * (1-alpha_bar_t_prev) / (1-alpha_bar_t))[:, None, None, None]

    z = torch.randn_like(xt)
    xt_prev = mu_t + sigma_t * z

    return xt_prev