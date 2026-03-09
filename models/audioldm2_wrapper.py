from __future__ import annotations
import torch
import torch.nn as nn


class DummyLatentAudioEncoder(nn.Module):
    """
    先用 dummy 版本把整條 pipeline 跑通。
    之後再替換成真正的 AudioLDM2 VAE / spectrogram encoder。
    """
    def __init__(self, latent_channels: int = 8, latent_h: int = 32, latent_w: int = 32):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
        )
        self.to_latent = nn.Linear(128, latent_channels * latent_h)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, 56000]
        x = audio.unsqueeze(1)                    # [B, 1, L]
        x = self.encoder(x)                       # [B, 128, T]
        x = x.transpose(1, 2)                     # [B, T, 128]
        x = self.to_latent(x)                     # [B, T, latent_channels*latent_h]
        x = x.transpose(1, 2)                     # [B, latent_channels*latent_h, T]
        x = torch.nn.functional.interpolate(x, size=32, mode="linear", align_corners=False)
        b = x.shape[0]
        x = x.view(b, self.latent_channels, self.latent_h, self.latent_w)
        return x