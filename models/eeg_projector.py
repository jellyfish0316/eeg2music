from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGProjector(nn.Module):
    def __init__(
        self,
        in_channels: int = 125,
        conv_channels: list[int] | tuple[int, ...] = (256, 512, 1024, 2048),
        strides: list[int] | tuple[int, ...] = (5, 2, 2, 2),
        out_channels: int = 8,
        out_height: int = 32,
        out_width: int = 32,
    ):
        super().__init__()
        assert len(conv_channels) == len(strides)

        layers = []
        prev_c = in_channels
        kernels = [7, 5, 5, 5]

        for c, s, k in zip(conv_channels, strides, kernels):
            layers.extend([
                nn.Conv1d(prev_c, c, kernel_size=k, stride=s, padding=k // 2),
                nn.GroupNorm(8, c),
                nn.SiLU(),
            ])
            prev_c = c

        self.net = nn.Sequential(*layers)

        self.out_channels = out_channels
        self.out_height = out_height
        self.out_width = out_width

        self.proj = nn.Conv1d(prev_c, out_channels * out_height, kernel_size=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: [B, C, T]
        x = self.net(eeg)                    # [B, C', T']
        x = self.proj(x)                     # [B, out_channels*out_height, T']
        x = F.interpolate(
            x, size=self.out_width, mode="linear", align_corners=False
        )                                    # [B, out_channels*out_height, out_width]
        b = x.shape[0]
        x = x.view(b, self.out_channels, self.out_height, self.out_width)
        return x