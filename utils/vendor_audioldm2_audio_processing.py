from __future__ import annotations

import numpy as np
import torch
import librosa.util as librosa_util
from scipy.signal import get_window


def window_sumsquare(
    window,
    n_frames,
    hop_length,
    win_length,
    n_fft,
    dtype=np.float32,
    norm=None,
):
    """
    Vendored from AudioLDM2:
    https://raw.githubusercontent.com/haoheliu/AudioLDM2/main/audioldm2/utilities/audio/audio_processing.py
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, size=n_fft)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-5):
    """
    Vendored from AudioLDM2:
    https://raw.githubusercontent.com/haoheliu/AudioLDM2/main/audioldm2/utilities/audio/audio_processing.py
    """
    return normalize_fun(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    Vendored from AudioLDM2:
    https://raw.githubusercontent.com/haoheliu/AudioLDM2/main/audioldm2/utilities/audio/audio_processing.py
    """
    return torch.exp(x) / C
