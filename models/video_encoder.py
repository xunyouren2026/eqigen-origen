import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    def __init__(self, vae, config):
        super().__init__()
        self.vae = vae
        self.projection = nn.Linear(
            vae.latent_channels, config.dit_context_dim)
        self.aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.dit_context_dim, nhead=8, batch_first=True),
            num_layers=1
        )

    def forward(self, video):
        """
        video: (B, C, T, H, W)
        Returns: (B, D)
        """
        with torch.no_grad():
            z, _, _ = self.vae.encode(video)  # (B, latent, T, H//8, W//8)
        z_pool = z.mean(dim=(2, 3, 4))  # (B, latent)
        feat = self.projection(z_pool)  # (B, D)
        return feat
