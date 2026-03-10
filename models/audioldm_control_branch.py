from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .audioldm_unet_wrapper import timestep_embedding


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class AudioLDMControlBranch(nn.Module):
    def __init__(
        self,
        input_blocks: nn.ModuleList,
        middle_block: nn.Module,
        time_embed: nn.Module,
        model_channels: int,
        input_block_channels: list[int],
        middle_block_channel: int,
        zero_init: bool = True,
        inject_middle_block: bool = True,
    ) -> None:
        super().__init__()
        self.model_channels = int(model_channels)
        self.time_embed = copy.deepcopy(time_embed)
        self.input_blocks = copy.deepcopy(input_blocks)
        self.middle_block = copy.deepcopy(middle_block)

        conv_builder = zero_module if zero_init else (lambda m: m)
        self.down_zero_convs = nn.ModuleList(
            [conv_builder(nn.Conv2d(int(ch), int(ch), kernel_size=1)) for ch in input_block_channels]
        )
        self.mid_zero_conv = conv_builder(nn.Conv2d(int(middle_block_channel), int(middle_block_channel), kernel_size=1))
        self.inject_middle_block = bool(inject_middle_block)

    def forward(
        self,
        zt: torch.Tensor,
        projected_latent: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict[str, object]:
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = (zt + projected_latent).to(dtype=zt.dtype)
        down_block_residuals = []

        for block, zero_conv in zip(self.input_blocks, self.down_zero_convs):
            h = block(h, emb, [], [])
            down_block_residuals.append(zero_conv(h))

        h = self.middle_block(h, emb, [], [])
        mid_block_residual = self.mid_zero_conv(h) if self.inject_middle_block else None
        return {
            "down_block_residuals": down_block_residuals,
            "mid_block_residual": mid_block_residual,
        }
