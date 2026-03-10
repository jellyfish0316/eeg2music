from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class AudioLDMControlBranch(nn.Module):
    """
    Minimal ControlNet-like branch producing residuals for input/middle blocks.
    """

    def __init__(
        self,
        latent_channels: int,
        hint_channels: int,
        eeg_global_dim: int,
        input_block_channels: list[int],
        input_block_ds: list[int],
        middle_block_channel: int,
        middle_block_ds: int,
        hidden_channels: int = 128,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_block_ds = list(input_block_ds)
        self.middle_block_ds = int(middle_block_ds)

        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(eeg_global_dim + hidden_channels, 2 * hidden_channels),
            nn.SiLU(),
            nn.Linear(2 * hidden_channels, 2 * hidden_channels),
        )

        self.stem = nn.Sequential(
            nn.Conv2d(latent_channels + hint_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )

        conv_builder = zero_module if zero_init else (lambda m: m)
        self.down_zero_convs = nn.ModuleList(
            [
                conv_builder(nn.Conv2d(hidden_channels, out_ch, kernel_size=1))
                for out_ch in input_block_channels
            ]
        )
        self.mid_zero_conv = conv_builder(
            nn.Conv2d(hidden_channels, middle_block_channel, kernel_size=1)
        )

    def _target_hw(self, hw: tuple[int, int], ds: int) -> tuple[int, int]:
        h, w = hw
        if ds <= 1:
            return h, w
        return max(1, h // ds), max(1, w // ds)

    def forward(
        self,
        zt: torch.Tensor,
        hint: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, object]:
        if hint.shape[-2:] != zt.shape[-2:]:
            hint = F.interpolate(
                hint,
                size=zt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        base = self.stem(torch.cat([zt, hint], dim=1))

        t = timesteps.float().unsqueeze(-1)
        t_emb = self.time_embed(t)
        cond = torch.cat([y, t_emb], dim=-1)
        gamma_beta = self.cond_proj(cond)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        base = base * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        input_hw = (base.shape[-2], base.shape[-1])
        down_block_residuals = []
        for ds, proj in zip(self.input_block_ds, self.down_zero_convs):
            target_hw = self._target_hw(input_hw, ds)
            feat = F.interpolate(
                base,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            down_block_residuals.append(proj(feat))

        mid_feat = F.interpolate(
            base,
            size=self._target_hw(input_hw, self.middle_block_ds),
            mode="bilinear",
            align_corners=False,
        )
        mid_block_residual = self.mid_zero_conv(mid_feat)

        return {
            "down_block_residuals": down_block_residuals,
            "mid_block_residual": mid_block_residual,
        }
