from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .subject_adapter import SubjectAdapter
from .eeg_projector import EEGProjector
from .audioldm2_wrapper import DummyLatentAudioEncoder


class SimpleUNet(nn.Module):
    def __init__(self, channels: int = 8):
        super().__init__()
        self.enc1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(128, 128, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, channels, 3, padding=1)

    def forward(self, x):
        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1))
        hm = F.silu(self.mid(h2))
        hd = F.silu(self.dec1(hm))
        out = self.out(hd + h1)
        return out


class EEGControlNetModel(nn.Module):
    def __init__(
        self,
        eeg_channels: int,
        num_subjects: int,
        use_subject_adapter: bool = True,
        subject_emb_dim: int = 64,
        latent_channels: int = 8,
        latent_h: int = 32,
        latent_w: int = 32,
    ):
        super().__init__()

        self.use_subject_adapter = use_subject_adapter

        if use_subject_adapter:
            self.subject_adapter = SubjectAdapter(
                num_subjects=num_subjects,
                eeg_channels=eeg_channels,
                emb_dim=subject_emb_dim,
            )

        self.audio_encoder = DummyLatentAudioEncoder(
            latent_channels=latent_channels,
            latent_h=latent_h,
            latent_w=latent_w,
        )

        self.projector = EEGProjector(
            in_channels=eeg_channels,
            out_channels=latent_channels,
            out_height=latent_h,
            out_width=latent_w,
        )

        self.control_unet = SimpleUNet(channels=latent_channels)

    def q_sample(self, z0: torch.Tensor, noise: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
        return (alpha ** 0.5) * z0 + ((1.0 - alpha) ** 0.5) * noise

    def forward(
        self,
        eeg: torch.Tensor,
        audio: torch.Tensor,
        subject_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.use_subject_adapter:
            eeg = self.subject_adapter(eeg, subject_idx)

        z0 = self.audio_encoder(audio)              # [B, C, H, W]
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, noise)

        eeg_cond = self.projector(eeg)              # [B, C, H, W]

        # paper 精神：encoder path condition
        eps_pred = self.control_unet(zt + eeg_cond)

        loss = F.mse_loss(eps_pred, noise)

        return {
            "loss": loss,
            "z0": z0,
            "zt": zt,
            "noise": noise,
            "eps_pred": eps_pred,
            "eeg_cond": eeg_cond,
        }