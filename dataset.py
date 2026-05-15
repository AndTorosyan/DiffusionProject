dataset = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
)

# // For checking with matplotlib

# idx = random.randint(0, len(dataset)-1)
# image, label = dataset[idx]

# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(f"Label: {label}")
# plt.axis("off")
# plt.show()