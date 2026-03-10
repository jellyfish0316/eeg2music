from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGHintEncoder(nn.Module):
    """
    Encode EEG [B, C, T] into spatial hint map [B, hint_channels, H, W].
    """

    def __init__(
        self,
        in_channels: int = 125,
        hidden_channels: int = 256,
        hint_channels: int = 64,
    ) -> None:
        super().__init__()
        self.hint_channels = hint_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hint_channels, kernel_size=1),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        h, w = target_hw
        x = self.net(eeg)
        x = F.interpolate(x, size=h * w, mode="linear", align_corners=False)
        x = x.view(x.shape[0], self.hint_channels, h, w)
        return x
