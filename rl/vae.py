from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """PyTorch equivalent of Raffin/WorldModels ConvVAE for 80x160 RGB input."""

    def __init__(self, z_size: int = 512):
        super().__init__()
        self.z_size = z_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.enc_fc_mu = nn.Linear(3 * 8 * 256, z_size)
        self.enc_fc_logvar = nn.Linear(3 * 8 * 256, z_size)

        self.dec_fc = nn.Linear(z_size, 3 * 8 * 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.enc_fc_mu(h), self.enc_fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, 256, 3, 8)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    kl_tolerance: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = target.shape[0]
    r_loss = F.mse_loss(recon, target, reduction="sum") / batch_size
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    if kl_tolerance > 0:
        kl_per_sample = torch.maximum(
            kl_per_sample,
            torch.full_like(kl_per_sample, kl_tolerance * mu.shape[1]),
        )
    kl_loss = kl_per_sample.mean()
    loss = r_loss + beta * kl_loss
    return loss, r_loss, kl_loss
