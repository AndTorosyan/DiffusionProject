import torch
import matplotlib.pyplot as plt
from reverse_diffusion import reverse_diffusion

def generate_digit(model, digit, device, alpha, alpha_bar, T):
    if digit not in range(10):
        print("Please enter a digit between 0 and 9.")
        return

    print(f"Generating digit: {digit}")
    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    label = torch.tensor([digit], device=device)

    with torch.no_grad():
        for t_val in range(T - 1, -1, -1):
            t_batch = torch.tensor([t_val], device=device)
            x = reverse_diffusion(x, t_batch, label, alpha, alpha_bar, model)

    plt.imshow(x.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f'Generated digit: {digit}')
    plt.axis('off')
    plt.show()

