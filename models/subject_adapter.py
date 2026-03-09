from __future__ import annotations
import torch
import torch.nn as nn


class SubjectAdapter(nn.Module):
    def __init__(self, num_subjects: int, eeg_channels: int, emb_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_subjects, emb_dim)
        self.to_scale = nn.Linear(emb_dim, eeg_channels)
        self.to_shift = nn.Linear(emb_dim, eeg_channels)

    def forward(self, eeg: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        # eeg: [B, C, T]
        emb = self.embedding(subject_idx)          # [B, D]
        scale = self.to_scale(emb).unsqueeze(-1)   # [B, C, 1]
        shift = self.to_shift(emb).unsqueeze(-1)   # [B, C, 1]
        return eeg * (1.0 + scale) + shift