from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .subject_adapter import SubjectAdapter
from .eeg_global_encoder import EEGGlobalEncoder
from .eeg_hint_encoder import EEGHintEncoder
from .audioldm2_wrapper import AudioLDM2MusicEncoderWrapper
from .audioldm_unet_wrapper import AudioLDMUNetWrapper
from .audioldm_control_branch import AudioLDMControlBranch


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
        enable_audio_encoder: bool = True,
        latent_channels: int | None = None,
        eeg_global_dim: int = 512,
        diffusion_num_steps: int = 1000,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 2e-2,
        unet_base_channels: int = 128,
        prefer_audioldm_unet: bool = False,
        audioldm_unet_kwargs: dict | None = None,
        controlnet_enabled: bool = False,
        controlnet_hint_channels: int = 64,
        controlnet_zero_init: bool = True,
        controlnet_scale: float = 1.0,
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

        self.audio_encoder: AudioLDM2MusicEncoderWrapper | None = None
        inferred_latent_channels = None
        if enable_audio_encoder:
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
        self.latent_channels = int(latent_channels)

        self.eeg_global_encoder = EEGGlobalEncoder(
            in_channels=eeg_channels,
            out_dim=eeg_global_dim,
        )

        self.control_unet = AudioLDMUNetWrapper(
            latent_channels=self.latent_channels,
            extra_film_condition_dim=eeg_global_dim,
            base_channels=unet_base_channels,
            prefer_audioldm_unet=prefer_audioldm_unet,
            audioldm_unet_kwargs=audioldm_unet_kwargs,
        )

        self.controlnet_enabled = bool(controlnet_enabled)
        self.default_control_scale = float(controlnet_scale)
        self.eeg_hint_encoder: EEGHintEncoder | None = None
        self.control_branch: AudioLDMControlBranch | None = None
        if self.controlnet_enabled:
            specs = self.control_unet.control_specs
            self.eeg_hint_encoder = EEGHintEncoder(
                in_channels=eeg_channels,
                hint_channels=controlnet_hint_channels,
            )
            self.control_branch = AudioLDMControlBranch(
                latent_channels=self.latent_channels,
                hint_channels=controlnet_hint_channels,
                eeg_global_dim=eeg_global_dim,
                input_block_channels=specs["input_block_channels"],
                input_block_ds=specs["input_block_ds"],
                middle_block_channel=specs["middle_block_channel"],
                middle_block_ds=specs["middle_block_ds"],
                hidden_channels=max(64, controlnet_hint_channels),
                zero_init=controlnet_zero_init,
            )

        self.num_train_timesteps = int(diffusion_num_steps)
        betas = torch.linspace(
            diffusion_beta_start,
            diffusion_beta_end,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.num_train_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def q_sample(
        self,
        z0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[timesteps][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None, None]
        zt = sqrt_alpha_t * z0 + sqrt_one_minus_alpha_t * noise
        return zt, noise

    def forward(
        self,
        eeg: torch.Tensor,
        subject_idx: torch.Tensor,
        audio: torch.Tensor | None = None,
        z0: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        control_scale: float | None = None,
        use_control: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.use_subject_adapter:
            eeg = self.subject_adapter(eeg, subject_idx)

        if z0 is None:
            if audio is None:
                raise ValueError("Either z0 or audio must be provided.")
            if self.audio_encoder is None:
                raise RuntimeError("audio_encoder is disabled but audio was provided.")
            z0 = self.audio_encoder(audio)

        if z0.dim() != 4:
            raise RuntimeError(f"Expected z0 to be 4D [B,C,H,W], got {tuple(z0.shape)}")
        if z0.shape[1] != self.latent_channels:
            raise RuntimeError(f"z0 channels ({z0.shape[1]}) != model latent_channels ({self.latent_channels})")
        z0 = z0.float()

        if timesteps is None:
            timesteps = self.sample_timesteps(batch_size=z0.shape[0], device=z0.device)
        if timesteps.shape != (z0.shape[0],):
            raise RuntimeError(f"Expected timesteps {(z0.shape[0],)}, got {tuple(timesteps.shape)}")

        zt, noise = self.q_sample(z0, timesteps=timesteps, noise=noise)
        eeg_global = self.eeg_global_encoder(eeg).to(dtype=z0.dtype)

        use_control = bool(use_control and self.controlnet_enabled)
        control_residuals = None
        eeg_hint = None
        if use_control:
            if self.eeg_hint_encoder is None or self.control_branch is None:
                raise RuntimeError("controlnet_enabled=True but control modules are missing.")
            eeg_hint = self.eeg_hint_encoder(
                eeg,
                target_hw=(z0.shape[2], z0.shape[3]),
            ).to(dtype=z0.dtype)
            control_residuals = self.control_branch(
                zt=zt,
                hint=eeg_hint,
                timesteps=timesteps,
                y=eeg_global,
            )

        eps_pred = self.control_unet(
            x=zt,
            timesteps=timesteps,
            y=eeg_global,
            control_residuals=control_residuals,
            control_scale=self.default_control_scale if control_scale is None else float(control_scale),
        )
        loss = F.mse_loss(eps_pred, noise)

        return {
            "loss": loss,
            "z0": z0,
            "zt": zt,
            "timesteps": timesteps,
            "noise": noise,
            "eps_pred": eps_pred,
            "eeg_global": eeg_global,
            "eeg_hint": eeg_hint,
            "use_control": torch.tensor(use_control, device=z0.device),
        }
