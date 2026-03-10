from __future__ import annotations

import math
import warnings
import torch
import torch.nn as nn

try:
    from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel as AudioLDMUNetModel
except Exception:
    AudioLDMUNetModel = None


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Sinusoidal timestep embedding compatible with DDPM/AudioLDM-style UNet.
    """
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)

    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AudioLDMUNetWrapper(nn.Module):
    """
    Minimal AudioLDM-style UNet wrapper with global FiLM conditioning.

    Interface is intentionally close to AudioLDM training UNetModel:
      forward(x, timesteps, y)
    where y is a global condition vector [B, D].
    """

    def __init__(
        self,
        latent_channels: int = 8,
        extra_film_condition_dim: int = 512,
        base_channels: int = 128,
        prefer_audioldm_unet: bool = False,
        audioldm_unet_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        self.latent_channels = latent_channels
        self.extra_film_condition_dim = extra_film_condition_dim
        self.using_audioldm_unet = False
        self.audioldm_unet_available = AudioLDMUNetModel is not None

        unet_kwargs = dict(audioldm_unet_kwargs or {})
        if prefer_audioldm_unet and self.audioldm_unet_available:
            # Required by AudioLDM UNetModel signature.
            unet_kwargs.setdefault("image_size", 16)
            unet_kwargs.setdefault("in_channels", latent_channels)
            unet_kwargs.setdefault("model_channels", base_channels)
            unet_kwargs.setdefault("out_channels", latent_channels)
            unet_kwargs.setdefault("num_res_blocks", 2)
            unet_kwargs.setdefault("attention_resolutions", (2, 4, 8))

            # Stable defaults for first integration pass.
            unet_kwargs.setdefault("channel_mult", (1, 2, 4, 8))
            unet_kwargs.setdefault("dropout", 0.0)
            unet_kwargs.setdefault("num_head_channels", 32)
            unet_kwargs.setdefault("use_scale_shift_norm", True)
            unet_kwargs.setdefault("extra_film_condition_dim", extra_film_condition_dim)

            try:
                self.backbone = AudioLDMUNetModel(**unet_kwargs)
                self.using_audioldm_unet = True
                return
            except Exception as exc:
                warnings.warn(
                    f"AudioLDM UNet init failed ({type(exc).__name__}: {exc}). "
                    "Falling back to local UNet wrapper.",
                    RuntimeWarning,
                )

        # Fallback UNet keeps the same conditioning contract as AudioLDM UNet:
        # emb = time_embed(timestep_embedding(t)) and optional FiLM concat by y.
        fallback_model_channels = int(unet_kwargs.get("model_channels", base_channels))
        fallback_dropout = float(unet_kwargs.get("dropout", 0.0))
        fallback_use_fp16 = bool(unet_kwargs.get("use_fp16", False))
        self.dtype = torch.float16 if fallback_use_fp16 else torch.float32
        self.model_channels = fallback_model_channels
        self.time_embed_dim = fallback_model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(fallback_model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.film = nn.Sequential(
            nn.Linear(extra_film_condition_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Minimal residual-style stack driven by emb -> FiLM on feature maps.
        self.in_conv = nn.Conv2d(latent_channels, fallback_model_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, fallback_model_channels)
        self.norm2 = nn.GroupNorm(8, fallback_model_channels)
        self.emb_to_scale_shift_1 = nn.Linear(self.time_embed_dim * 2, 2 * fallback_model_channels)
        self.emb_to_scale_shift_2 = nn.Linear(self.time_embed_dim * 2, 2 * fallback_model_channels)
        self.mid1 = nn.Conv2d(
            fallback_model_channels, fallback_model_channels, kernel_size=3, padding=1
        )
        self.mid2 = nn.Conv2d(
            fallback_model_channels, fallback_model_channels, kernel_size=3, padding=1
        )
        self.dropout = nn.Dropout(fallback_dropout)
        self.out_conv = nn.Conv2d(fallback_model_channels, latent_channels, kernel_size=3, padding=1)

    @property
    def backend_name(self) -> str:
        return "audioldm_unet" if self.using_audioldm_unet else "fallback_unet"

    @staticmethod
    def _apply_scale_shift(
        h: torch.Tensor,
        scale_shift: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)
        return h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    def _get_unet_spatial_factor(self) -> int:
        """
        Infer total downsample factor from channel_mult depth.
        AudioLDM/OpenAI UNet downsamples len(channel_mult)-1 times.
        """
        if not self.using_audioldm_unet:
            return 1
        channel_mult = getattr(self.backbone, "channel_mult", None)
        if isinstance(channel_mult, (list, tuple)) and len(channel_mult) > 0:
            return 2 ** (len(channel_mult) - 1)
        return 8

    @staticmethod
    def _pad_for_unet(
        x: torch.Tensor,
        factor: int,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Pad H/W to multiples of factor using replication padding.
        Returns padded tensor and original (H, W).
        """
        if factor <= 1:
            return x, (x.shape[-2], x.shape[-1])
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (h, w)

    @staticmethod
    def _crop_to_hw(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        h, w = hw
        return x[..., :h, :w]

    def _fallback_forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = timestep_embedding(timesteps, self.model_channels)
        temb = self.time_embed(t_emb)  # [B, 4C]
        yemb = self.film(y)            # [B, 4C]
        emb = torch.cat([temb, yemb], dim=-1)  # [B, 8C]

        h = self.in_conv(x).to(dtype=x.dtype)

        ss1 = self.emb_to_scale_shift_1(emb)
        h = self.norm1(h)
        h = self._apply_scale_shift(h, ss1)
        h = torch.nn.functional.silu(h)
        h = self.mid1(h)

        ss2 = self.emb_to_scale_shift_2(emb)
        h = self.norm2(h)
        h = self._apply_scale_shift(h, ss2)
        h = torch.nn.functional.silu(h)
        h = self.dropout(h)
        h = self.mid2(h)

        eps = self.out_conv(h)
        return eps

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        context_list=None,
        context_attn_mask_list=None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if timesteps is None:
            raise ValueError("timesteps is required")
        if y is None:
            raise ValueError("y is required when using EEG global FiLM conditioning")
        if x.dim() != 4:
            raise ValueError(f"Expected x=[B,C,H,W], got {tuple(x.shape)}")
        if timesteps.dim() != 1:
            raise ValueError(f"Expected timesteps=[B], got {tuple(timesteps.shape)}")
        if y.dim() != 2:
            raise ValueError(f"Expected y=[B,D], got {tuple(y.shape)}")
        if x.shape[0] != timesteps.shape[0] or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch mismatch x={tuple(x.shape)}, t={tuple(timesteps.shape)}, y={tuple(y.shape)}"
            )
        if y.shape[-1] != self.extra_film_condition_dim:
            raise ValueError(
                f"Expected y dim {self.extra_film_condition_dim}, got {tuple(y.shape)}"
            )

        if self.using_audioldm_unet:
            if context_list is None:
                context_list = []
            if context_attn_mask_list is None:
                context_attn_mask_list = [None] * len(context_list)
            if len(context_attn_mask_list) != len(context_list):
                raise ValueError(
                    "context_attn_mask_list length must match context_list length: "
                    f"{len(context_attn_mask_list)} != {len(context_list)}"
                )

            # AudioLDM UNet can mismatch skip-connection shapes on odd spatial sizes
            # (e.g. latent width 87). Pad to valid multiples, then crop back.
            spatial_factor = self._get_unet_spatial_factor()
            x_in, orig_hw = self._pad_for_unet(x, factor=spatial_factor)
            eps = self.backbone(
                x=x_in,
                timesteps=timesteps,
                y=y,
                context_list=context_list,
                context_attn_mask_list=context_attn_mask_list,
            )
            if isinstance(eps, tuple):
                eps = eps[0]
            if hasattr(eps, "sample"):
                eps = eps.sample
            eps = self._crop_to_hw(eps, orig_hw)
            return eps

        eps = self._fallback_forward(x=x, timesteps=timesteps, y=y)
        return eps
