import torch
from torch.utils.data import DataLoader
from dataset import load_dataset
from generation import generate_digit
from models import ConditionalUNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    Unet = ConditionalUNet()
    Unet.to(device)

    T = 1000
    noise_scheduler = torch.linspace(1e-4, 0.02, T, device=device)
    alpha = 1.0 - noise_scheduler
    alpha_bar = torch.cumprod(alpha, dim=0)

    # // Training section
    # If you want to train your own model, uncomment these lines and run the code:

    # Unet.fit(dataloader, alpha_bar, T, epochs=67, device=device)
    # torch.save(Unet, 'entire_model.pth')

    # // Testing section
    # If you want to test with your own .pth file, uncomment the following lines:

    # path = 'your_parameters.pth'
    # Unet = torch.load(path, map_location=device)
    # Unet.eval()
    # digit = int(input("Enter a digit to generate an image for (0-9): "))
    # generate_digit(Unet, digit, device, alpha, alpha_bar, T)

    print("Test complete.")
