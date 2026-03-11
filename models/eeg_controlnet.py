from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audioldm2_wrapper import AudioLDM2MusicEncoderWrapper
from .audioldm_control_branch import AudioLDMControlBranch
from .audioldm_unet_wrapper import AudioLDMUNetWrapper
from .eeg_projector import EEGProjector
from .subject_adapter import SubjectAdapter


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
        text_prompt: str = "Pop music",
        text_cache_path: str | None = None,
        enable_audio_encoder: bool = True,
        latent_channels: int | None = None,
        latent_grid: tuple[int, int, int] | None = None,
        projector_channels: tuple[int, ...] = (256, 512, 1024, 2048),
        projector_strides: tuple[int, ...] = (5, 2, 2, 2),
        projector_use_linear_fallback: bool = True,
        diffusion_num_steps: int = 1000,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 2e-2,
        unet_cache_pipeline: bool = True,
        controlnet_enabled: bool = False,
        controlnet_zero_init: bool = True,
        controlnet_scale: float = 1.0,
        controlnet_copy_encoder_weights: bool = True,
        controlnet_inject_middle_block: bool = True,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)

        self.use_subject_adapter = bool(use_subject_adapter)
        if self.use_subject_adapter:
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
        elif inferred_latent_channels is not None and int(inferred_latent_channels) != int(latent_channels):
            raise ValueError(
                f"latent_channels={latent_channels} does not match VAE latent_channels={int(inferred_latent_channels)}"
            )
        self.latent_channels = int(latent_channels)

        if latent_grid is None:
            raise ValueError("latent_grid must be checkpoint-derived before model construction.")
        self.latent_grid = tuple(int(v) for v in latent_grid)
        if len(self.latent_grid) != 3:
            raise ValueError(f"Expected latent_grid=(C,H,W), got {self.latent_grid}")
        if self.latent_grid[0] != self.latent_channels:
            raise ValueError(
                f"latent_grid channel dimension ({self.latent_grid[0]}) does not match latent_channels ({self.latent_channels})"
            )

        self.projector = EEGProjector(
            in_channels=eeg_channels,
            conv_channels=projector_channels,
            strides=projector_strides,
            latent_grid=self.latent_grid,
            use_linear_fallback=projector_use_linear_fallback,
        )

        self.control_unet = AudioLDMUNetWrapper(
            model_id=audio_model_id,
            device=device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            cache_pipeline=bool(unet_cache_pipeline),
            text_prompt=text_prompt,
            text_cache_path=text_cache_path,
        )
        unet_in_channels = getattr(self.control_unet.config, "in_channels", None)
        unet_out_channels = getattr(self.control_unet.config, "out_channels", None)
        if unet_in_channels is not None and int(unet_in_channels) != self.latent_channels:
            raise ValueError(
                f"Pretrained U-Net in_channels ({int(unet_in_channels)}) do not match latent_channels ({self.latent_channels})."
            )
        if unet_out_channels is not None and int(unet_out_channels) != self.latent_channels:
            raise ValueError(
                f"Pretrained U-Net out_channels ({int(unet_out_channels)}) do not match latent_channels ({self.latent_channels})."
            )

        self.controlnet_enabled = bool(controlnet_enabled)
        self.default_control_scale = float(controlnet_scale)
        self.control_branch: AudioLDMControlBranch | None = None
        if self.controlnet_enabled:
            if not bool(controlnet_copy_encoder_weights):
                raise ValueError("Paper-aligned ControlNet requires copy_encoder_weights=True.")
            specs = self.control_unet.control_specs
            control_modules = self.control_unet.get_control_modules()
            self.control_branch = AudioLDMControlBranch(
                conv_in=control_modules["conv_in"],
                down_blocks=control_modules["down_blocks"],
                mid_block=control_modules["mid_block"],
                time_proj=self.control_unet.get_time_proj(),
                time_embedding=self.control_unet.get_time_embedding(),
                latent_channels=self.latent_channels,
                latent_hw=(self.latent_grid[1], self.latent_grid[2]),
                cross_attention_dims=self.control_unet.get_cross_attention_dims(),
                input_block_channels=specs["input_block_channels"],
                middle_block_channel=specs["middle_block_channel"],
                zero_init=controlnet_zero_init,
                inject_middle_block=controlnet_inject_middle_block,
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
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(
        self,
        z0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(z0)
        noise = noise.to(dtype=z0.dtype)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[timesteps][:, None, None, None].to(dtype=z0.dtype)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None, None].to(dtype=z0.dtype)
        zt = sqrt_alpha_t * z0 + sqrt_one_minus_alpha_t * noise
        return zt, noise

    def predict_noise(
        self,
        eeg: torch.Tensor,
        subject_idx: torch.Tensor,
        zt: torch.Tensor,
        timesteps: torch.Tensor,
        control_scale: float | None = None,
        use_control: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, object] | None]:
        if zt.dim() != 4:
            raise RuntimeError(f"Expected zt to be 4D [B,C,H,W], got {tuple(zt.shape)}")
        if zt.shape[1] != self.latent_channels:
            raise RuntimeError(f"zt channels ({zt.shape[1]}) != model latent_channels ({self.latent_channels})")

        if self.use_subject_adapter:
            eeg = self.subject_adapter(eeg, subject_idx)

        unet_dtype = self.control_unet.dtype
        zt = zt.to(dtype=unet_dtype)
        projected_latent = self.projector(eeg).to(dtype=unet_dtype)
        encoder_state_dict = self.control_unet.get_text_conditioning(
            batch_size=zt.shape[0],
            device=zt.device,
            dtype=unet_dtype,
        )

        use_control = bool(use_control and self.controlnet_enabled)
        control_residuals = None
        if use_control:
            if self.control_branch is None:
                raise RuntimeError("controlnet_enabled=True but control_branch is missing.")
            control_residuals = self.control_branch(
                zt=zt,
                projected_latent=projected_latent,
                timesteps=timesteps,
                encoder_hidden_states=encoder_state_dict["encoder_hidden_states"],
                encoder_hidden_states_1=encoder_state_dict["encoder_hidden_states_1"],
            )

        eps_pred = self.control_unet(
            x=zt,
            timesteps=timesteps,
            encoder_hidden_states=encoder_state_dict["encoder_hidden_states"],
            encoder_hidden_states_1=encoder_state_dict["encoder_hidden_states_1"],
            control_residuals=control_residuals,
            control_scale=self.default_control_scale if control_scale is None else float(control_scale),
        )
        return {
            "zt": zt,
            "projected_latent": projected_latent,
            "eps_pred": eps_pred,
            "control_residuals": control_residuals,
            "use_control": torch.tensor(use_control, device=zt.device),
        }

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
    ) -> dict[str, torch.Tensor | dict[str, object] | None]:
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
        unet_dtype = self.control_unet.dtype
        z0 = z0.to(dtype=unet_dtype)

        if timesteps is None:
            timesteps = self.sample_timesteps(batch_size=z0.shape[0], device=z0.device)
        if timesteps.shape != (z0.shape[0],):
            raise RuntimeError(f"Expected timesteps {(z0.shape[0],)}, got {tuple(timesteps.shape)}")

        zt, noise = self.q_sample(z0, timesteps=timesteps, noise=noise)
        pred = self.predict_noise(
            eeg=eeg,
            subject_idx=subject_idx,
            zt=zt,
            timesteps=timesteps,
            control_scale=control_scale,
            use_control=use_control,
        )
        eps_pred = pred["eps_pred"]
        loss = F.mse_loss(eps_pred.float(), noise.float())
        return {
            "loss": loss,
            "z0": z0,
            "zt": pred["zt"],
            "timesteps": timesteps,
            "noise": noise,
            "projected_latent": pred["projected_latent"],
            "eps_pred": eps_pred,
            "control_residuals": pred["control_residuals"],
            "use_control": pred["use_control"],
        }
