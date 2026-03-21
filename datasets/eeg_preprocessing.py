from __future__ import annotations

from typing import Any

import numpy as np


def apply_source_level_eeg_preprocessing(
    data: np.ndarray,
    preprocessing: dict[str, Any] | None,
) -> np.ndarray:
    """
    Apply dataset-level EEG preprocessing to a full source tensor.

    Expected input shape: [channels, time, subjects]
    """
    if preprocessing is None:
        return data.astype(np.float32, copy=False)

    out = data.astype(np.float32, copy=True)

    # Paper-aligned default: keep the first 124 non-face channels.
    if bool(preprocessing.get("drop_face_channels", False)):
        keep_n = int(preprocessing.get("keep_first_n_channels", 124))
        if keep_n <= 0 or keep_n > out.shape[0]:
            raise ValueError(
                f"Invalid keep_first_n_channels={keep_n}; data has {out.shape[0]} channels."
            )
        out = out[:keep_n]

    center_n = preprocessing.get("center_using_first_samples", None)
    if center_n is not None:
        center_n = int(center_n)
        if center_n <= 0:
            raise ValueError(f"center_using_first_samples must be > 0, got {center_n}")
        use_n = min(center_n, out.shape[1])
        baseline = out[:, :use_n, :].mean(axis=1, keepdims=True)
        out = out - baseline

    clamp_std = preprocessing.get("std_clamp", None)
    if clamp_std is not None:
        clamp_std = float(clamp_std)
        if clamp_std <= 0:
            raise ValueError(f"std_clamp must be > 0, got {clamp_std}")
        std = out.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        limit = clamp_std * std
        out = np.clip(out, -limit, limit)

    return out.astype(np.float32, copy=False)


def apply_chunk_level_eeg_preprocessing(
    eeg: np.ndarray,
    preprocessing: dict[str, Any] | None,
    *,
    fallback_normalize: bool = True,
) -> np.ndarray:
    """
    Apply chunk-level preprocessing to a [channels, time] EEG chunk.
    """
    cfg = dict(preprocessing or {})
    do_norm = bool(cfg.get("per_channel_normalization", fallback_normalize))
    if do_norm:
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True) + 1e-8
        eeg = (eeg - mean) / std
    return eeg.astype(np.float32, copy=False)
