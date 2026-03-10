from __future__ import annotations

import torch
import torch.nn as nn


class EEGGlobalEncoder(nn.Module):
    """
    Encode EEG [B, C, T] into a global embedding [B, D].
    """

    def __init__(
        self,
        in_channels: int = 125,
        hidden_channels: int = 256,
        out_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: [B, C, T]
        x = self.conv(eeg)          # [B, H, T']
        x = self.pool(x).squeeze(-1)  # [B, H]
        x = self.proj(x)            # [B, D]
        return x
