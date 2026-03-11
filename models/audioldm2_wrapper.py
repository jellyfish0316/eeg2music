from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    from diffusers import AudioLDM2Pipeline
except Exception:
    AudioLDM2Pipeline = None


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
        if torchaudio is None:
            raise ImportError("torchaudio is required to load the AudioLDM2 VAE wrapper.")
        if AudioLDM2Pipeline is None:
            raise ImportError("diffusers is required to load the AudioLDM2 VAE wrapper.")
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.device_name = device
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_mode = use_mode
        self._full_pipeline = None

        # Load the official pipeline, but we only keep the VAE.
        pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.vae = pipe.vae

        if cache_pipeline:
            # keep only what you need for encoder-side training by default
            del pipe

        self.vae.to(device)
        self.vae.eval()

        if freeze_vae:
            for p in self.vae.parameters():
                p.requires_grad = False

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

        waveform = waveform.to(device=self.device, dtype=torch.float32)

        waveform = waveform.clamp(-1.0, 1.0)

        mel = self.mel_transform(waveform)
        mel_db = self.amplitude_to_db(mel)
        mel_db = mel_db / 80.0
        mel_db = mel_db.unsqueeze(1)

        return mel_db.to(dtype=self.dtype)

    def _load_full_pipeline(self):
        if AudioLDM2Pipeline is None:
            raise ImportError("diffusers is required to load the AudioLDM2 full pipeline.")
        if self._full_pipeline is None:
            pipe = AudioLDM2Pipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)
            pipe = pipe.to(self.device)
            self._full_pipeline = pipe
        return self._full_pipeline

    @property
    def vocoder_sample_rate(self) -> int:
        pipe = self._load_full_pipeline()
        return int(pipe.vocoder.config.sampling_rate)

    @torch.no_grad()
    def infer_latent_shape(self, num_audio_samples: int) -> tuple[int, int, int]:
        dummy_waveform = torch.zeros(1, int(num_audio_samples), device=self.device, dtype=torch.float32)
        latents = self(dummy_waveform)
        if latents.dim() != 4:
            raise RuntimeError(f"Expected latent tensor [B,C,H,W], got {tuple(latents.shape)}")
        return tuple(int(v) for v in latents.shape[1:])

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
        mel = mel.to(device=self.device, dtype=self.dtype)

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
    def decode_latents_to_mel(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.dtype)
        # The training path uses checkpoint-derived latent grids in [C, H, W] = [C, 16, 87].
        # AudioLDM2's official decode path expects the spatial axes swapped back before vocoding.
        latents_for_decode = latents.transpose(-1, -2)
        decoded = self.vae.decode(latents_for_decode / self.scaling_factor)
        mel = decoded.sample if hasattr(decoded, "sample") else decoded
        if mel.dim() != 4:
            raise RuntimeError(f"Expected decoded mel [B,1,T,F], got {tuple(mel.shape)}")
        return mel

    @torch.no_grad()
    def decode_latents_to_waveform(self, latents: torch.Tensor) -> torch.Tensor:
        pipe = self._load_full_pipeline()
        mel = self.decode_latents_to_mel(latents)
        waveform = pipe.mel_spectrogram_to_waveform(mel)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return waveform

    @torch.no_grad()
    def get_audio_features(
        self,
        waveform: torch.Tensor,
        *,
        sample_rate: int | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        if torchaudio is None:
            raise ImportError("torchaudio is required for CLAP audio feature extraction.")
        pipe = self._load_full_pipeline()
        target_sr = int(pipe.feature_extractor.sampling_rate)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform [B,T], got {tuple(waveform.shape)}")

        audio = waveform.detach().cpu().float()
        input_sr = self.sample_rate if sample_rate is None else int(sample_rate)
        if input_sr != target_sr:
            audio = torchaudio.functional.resample(audio, orig_freq=input_sr, new_freq=target_sr)

        inputs = pipe.feature_extractor(
            [x.numpy() for x in audio],
            return_tensors="pt",
            sampling_rate=target_sr,
        )
        input_features = inputs.input_features.to(device=self.device, dtype=pipe.text_encoder.dtype)
        features = pipe.text_encoder.get_audio_features(input_features=input_features)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features.detach().cpu()

    @torch.no_grad()
    def compute_audio_similarity(
        self,
        waveform_a: torch.Tensor,
        waveform_b: torch.Tensor,
        *,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        feats_a = self.get_audio_features(waveform_a, sample_rate=sample_rate, normalize=True)
        feats_b = self.get_audio_features(waveform_b, sample_rate=sample_rate, normalize=True)
        if feats_a.shape != feats_b.shape:
            raise RuntimeError(
                f"Audio feature shape mismatch: {tuple(feats_a.shape)} vs {tuple(feats_b.shape)}"
            )
        return (feats_a * feats_b).sum(dim=-1)

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
