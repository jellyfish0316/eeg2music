from __future__ import annotations
import torch


def collate_fn(batch):
    eeg = torch.stack([x["eeg"] for x in batch], dim=0)
    audio = torch.stack([x["audio"] for x in batch], dim=0)
    subject_idx = torch.stack([x["subject_idx"] for x in batch], dim=0)
    chunk_idx = torch.stack([x["chunk_idx"] for x in batch], dim=0)
    text = [x["text"] for x in batch]

    return {
        "eeg": eeg,                  # [B, C, T]
        "audio": audio,              # [B, L]
        "subject_idx": subject_idx,  # [B]
        "chunk_idx": chunk_idx,      # [B]
        "text": text,
    }