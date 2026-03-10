import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torchaudio
from diffusers import AudioLDM2Pipeline


@dataclass
class AudioLDM2LatentOutput:
    latents: torch.Tensor
    mel: torch.Tensor
    posterior_mean: Optional[torch.Tensor] = None
    posterior_logvar: Optional[torch.Tensor] = None


class AudioLDM2MusicEncoderWrapper(nn.Module):
    """
    waveform -> log-mel spectrogram -> AudioLDM2 VAE encoder -> latent z

    Notes
    -----
    - AudioLDM2's VAE encodes mel-spectrograms, not raw waveform directly.
    - Returned latents are scaled by vae.config.scaling_factor so they can be
      used directly in a latent diffusion training pipeline.
    """

    def __init__(
        self,
        model_id: str = "cvssp/audioldm2-music",
        sample_rate: int = 16000,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        freeze_vae: bool = True,
        use_mode: bool = False,
        cache_pipeline: bool = True,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device_name = device
        self.dtype = dtype
        self.use_mode = use_mode

        # Load the official pipeline, but we only keep the VAE.
        pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.vae = pipe.vae

        if cache_pipeline:
            # keep only what you need
            del pipe

        self.vae.to(device)
        self.vae.eval()

        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

        # AudioLDM2 uses log-mel spectrograms before the VAE.
        # These parameters are conservative defaults for AudioLDM2-style 16k audio.
        # You should verify them against the exact checkpoint config / preprocessing
        # you want to match.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            n_mels=64,
            f_min=0.0,
            f_max=8000.0,
            power=2.0,
            center=True,
            normalized=False,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

        # For numerical stability
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        ).to(device)

    @property
    def scaling_factor(self) -> float:
        return float(self.vae.config.scaling_factor)

    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [B, T] or [B, 1, T]
        returns: [B, 1, n_mels, n_frames]
        """
        if waveform.dim() == 3:
            if waveform.size(1) != 1:
                raise ValueError(f"Expected mono audio [B,1,T], got {waveform.shape}")
            waveform = waveform[:, 0, :]
        elif waveform.dim() != 2:
            raise ValueError(f"Expected [B,T] or [B,1,T], got {waveform.shape}")

        waveform = waveform.to(device=self.device_name, dtype=torch.float32)

        # Optional safety clamp if your dataset occasionally exceeds [-1, 1]
        waveform = waveform.clamp(-1.0, 1.0)

        mel = self.mel_transform(waveform)          # [B, n_mels, n_frames]
        mel_db = self.amplitude_to_db(mel)          # log-like mel scale

        # Normalize roughly to a VAE-friendly range.
        # You may later replace this with the exact AudioLDM2 preprocessing once verified.
        mel_db = mel_db / 80.0                      # roughly [-1, 0]
        mel_db = mel_db.unsqueeze(1)                # [B, 1, n_mels, n_frames]

        return mel_db.to(dtype=self.dtype)

    @torch.no_grad()
    def encode_mel(
        self,
        mel: torch.Tensor,
        sample_posterior: bool = True,
        return_stats: bool = False,
    ) -> AudioLDM2LatentOutput:
        """
        mel: [B, 1, n_mels, n_frames]
        """
        mel = mel.to(device=self.device_name, dtype=self.dtype)

        posterior = self.vae.encode(mel).latent_dist

        if self.use_mode or not sample_posterior:
            z = posterior.mode()
        else:
            z = posterior.sample()

        z = z * self.scaling_factor

        mean = getattr(posterior, "mean", None) if return_stats else None
        logvar = getattr(posterior, "logvar", None) if return_stats else None

        return AudioLDM2LatentOutput(
            latents=z,
            mel=mel,
            posterior_mean=mean,
            posterior_logvar=logvar,
        )

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor,
        sample_posterior: bool = True,
        return_mel: bool = False,
        return_stats: bool = False,
    ):
        mel = self.waveform_to_mel(waveform)
        out = self.encode_mel(
            mel,
            sample_posterior=sample_posterior,
            return_stats=return_stats,
        )

        if return_mel or return_stats:
            return out
        return out.latents