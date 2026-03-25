from __future__ import annotations

import torch
import torch.nn as nn


class EEGProjector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: list[int] | tuple[int, ...] = (256, 512, 1024, 2048),
        strides: list[int] | tuple[int, ...] = (5, 2, 2, 2),
        latent_grid: tuple[int, int, int] = (8, 16, 87),
        kernel_sizes: list[int] | tuple[int, ...] = (7, 5, 5, 5),
    ) -> None:
        super().__init__()
        if len(conv_channels) != len(strides):
            raise ValueError("conv_channels and strides must have the same length.")
        if len(kernel_sizes) != len(conv_channels):
            raise ValueError("kernel_sizes and conv_channels must have the same length.")

        self.latent_channels, self.latent_height, self.latent_width = [int(v) for v in latent_grid]
        self.target_length = self.latent_height * self.latent_width

        prev_channels = int(in_channels)
        layers: list[nn.Module] = []
        for out_channels, stride, kernel_size in zip(conv_channels, strides, kernel_sizes):
            out_channels = int(out_channels)
            layers.extend(
                [
                    nn.Conv1d(
                        prev_channels,
                        out_channels,
                        kernel_size=int(kernel_size),
                        stride=int(stride),
                        padding=int(kernel_size) // 2,
                    ),
                    nn.GroupNorm(self._group_count(out_channels), out_channels),
                    nn.SiLU(),
                ]
            )
            prev_channels = out_channels
        self.temporal_conv = nn.Sequential(*layers)
        self.channel_proj = nn.Conv1d(prev_channels, self.latent_channels, kernel_size=1)

    @staticmethod
    def _group_count(channels: int) -> int:
        for groups in (32, 16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    @property
    def latent_grid(self) -> tuple[int, int, int]:
        return (self.latent_channels, self.latent_height, self.latent_width)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(eeg)
        x = self.channel_proj(x)
        batch_size, channels, length = x.shape
        if length == self.target_length:
            return x.view(batch_size, channels, self.latent_height, self.latent_width)

        raise RuntimeError(
            "Projector temporal length does not match checkpoint-derived latent grid "
            f"(got {length}, expected {self.target_length}). "
            "Adjust projector strides/kernel sizes or latent_grid so the strided Conv1d path reshapes directly."
        )
