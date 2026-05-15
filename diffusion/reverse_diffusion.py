import torch

def reverse_diffusion(xt, t, label, alpha, alpha_bar, model):
    if isinstance(t, torch.Tensor):
        t_index = t.item()
        t_tensor = t
    else:
        t_index = t
        t_tensor = torch.tensor([t], device=xt.device)

    eps_theta = model(xt, t_tensor, label)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t_tensor])[:, None, None, None]
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t_tensor])[:, None, None, None]
    x0_pred = (xt - sqrt_one_minus_alpha_bar_t * eps_theta) / sqrt_alpha_bar_t

    beta_t = 1 - alpha[t_tensor]
    alpha_t = alpha[t_tensor]
    alpha_bar_t = alpha_bar[t_tensor]
    if t_index > 0:
        alpha_bar_t_prev = alpha_bar[t_index - 1]
    else:
        alpha_bar_t_prev = torch.tensor(1.0, device=xt.device)

    coef_x0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)

    mu_t = coef_x0 * x0_pred + coef_xt * xt
    sigma_t = torch.sqrt(beta_t * (1-alpha_bar_t_prev) / (1-alpha_bar_t))[:, None, None, None]

    z = torch.randn_like(xt)
    xt_prev = mu_t + sigma_t * z

    return xt_prev