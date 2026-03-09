from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .subject_adapter import SubjectAdapter
from .eeg_projector import EEGProjector
from .audioldm2_wrapper import AudioLDM2MusicEncoderWrapper


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

        if hd.shape[-2:] != h1.shape[-2:]:
            hd = F.interpolate(
                hd,
                size=h1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        out = self.out(hd + h1)
        return out


class EEGControlNetModel(nn.Module):
    def __init__(
        self,
        eeg_channels: int,
        num_subjects: int,
        device: torch.device | str | None = None,
        use_subject_adapter: bool = True,
        subject_emb_dim: int = 64,
        audio_model_id: str = "cvssp/audioldm2-music",
        audio_sample_rate: int = 16000,
        audio_freeze_vae: bool = True,
        audio_use_mode: bool = False,
        latent_channels: int | None = None,
        latent_h: int = 32,
        latent_w: int = 32,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)

        self.use_subject_adapter = use_subject_adapter

        if use_subject_adapter:
            self.subject_adapter = SubjectAdapter(
                num_subjects=num_subjects,
                eeg_channels=eeg_channels,
                emb_dim=subject_emb_dim,
            )

        self.audio_encoder = AudioLDM2MusicEncoderWrapper(
            model_id=audio_model_id,
            sample_rate=audio_sample_rate,
            device=str(device),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            freeze_vae=audio_freeze_vae,
            use_mode=audio_use_mode,
        )

        inferred_latent_channels = getattr(self.audio_encoder.vae.config, "latent_channels", None)
        if latent_channels is None:
            if inferred_latent_channels is None:
                raise ValueError("latent_channels is None and could not be inferred from VAE config.")
            latent_channels = int(inferred_latent_channels)
        elif inferred_latent_channels is not None and int(inferred_latent_channels) != latent_channels:
            raise ValueError(
                f"latent_channels={latent_channels} does not match VAE latent_channels={int(inferred_latent_channels)}"
            )
        self.latent_channels = latent_channels

        self.projector = EEGProjector(
            in_channels=eeg_channels,
            out_channels=self.latent_channels,
            out_height=latent_h,
            out_width=latent_w,
        )

        self.control_unet = SimpleUNet(channels=self.latent_channels)

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
        if z0.dim() != 4:
            raise RuntimeError(f"Expected z0 to be 4D [B,C,H,W], got shape {tuple(z0.shape)}")
        if z0.shape[1] != self.latent_channels:
            raise RuntimeError(
                f"z0 channels ({z0.shape[1]}) != model latent_channels ({self.latent_channels})"
            )
        z0 = z0.float()

        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, noise)

        eeg_cond = self.projector(
            eeg,
            target_hw=(z0.shape[2], z0.shape[3]),
        )                                           # [B, C, H, W]
        if eeg_cond.shape != z0.shape:
            raise RuntimeError(
                f"shape mismatch: z0={tuple(z0.shape)}, eeg_cond={tuple(eeg_cond.shape)}"
            )
        eeg_cond = eeg_cond.to(dtype=z0.dtype)

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
