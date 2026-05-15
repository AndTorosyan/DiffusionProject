class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 128)
        self.time_embed = nn.Sequential(
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.time_proj1 = nn.Linear(128, 32)
        self.time_proj2 = nn.Linear(128, 64)
        self.time_proj3 = nn.Linear(128, 128)

        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 128, 3, padding=1)

        # Decoder
        self.dec1 = nn.Conv2d(128+64, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64+32, 32, 3, padding=1)
        self.dec3 = nn.Conv2d(32, 1, 3, padding=1)

    def get_time_embedding(self, t, dim=128):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_embed(emb)

    def forward(self, x, t, label):
        temb = self.get_time_embedding(t) + self.label_embed(label)

        enc1 = F.relu(self.enc1(x))
        enc1 = enc1 + self.time_proj1(temb)[:, :, None, None]

        enc2 = F.relu(self.enc2(enc1))
        enc2 = enc2 + self.time_proj2(temb)[:, :, None, None]

        enc3 = F.relu(self.enc3(enc2))
        enc3 = enc3 + self.time_proj3(temb)[:, :, None, None]

        b = F.relu(self.bottleneck(enc3))

        dec1 = F.interpolate(b, scale_factor=2)
        dec1 = F.relu(self.dec1(torch.cat([dec1, enc2], dim=1)))

        dec2 = F.interpolate(dec1, scale_factor=2)
        dec2 = F.relu(self.dec2(torch.cat([dec2, enc1], dim=1)))

        return self.dec3(dec2)

    def fit(self, dataloader, alpha_bar, T, epochs=1):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.to(device)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (x, label) in enumerate(dataloader):
                x, label = x.to(device), label.to(device)  # ← label used now
                t = torch.randint(0, T, (x.size(0),), device=device)

                xt, noise = forward_diffusion(x, t, alpha_bar)
                pred_noise = self(xt, t, label)  # ← pass label

                loss = F.mse_loss(pred_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if (i+1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

# // Used for loading and testing in google colab

# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# Unet = torch.load('entire_model.pth', weights_only=False)
# Unet.eval()
# Unet.fit(dataloader, alpha_bar, T, epochs=67)
# torch.save(Unet, 'entire_model.pth')