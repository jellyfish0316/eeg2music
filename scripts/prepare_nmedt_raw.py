#!/usr/bin/env python3
"""
Inspect and convert raw NMED-T MATLAB v7.3 EEG recordings into the
song-level MATLAB v5 files expected by this repo.

This fills the gap between the current training pipeline, which expects
`song21_Imputed.mat`-style files with shape `[channels, time, subjects]`,
and raw participant recordings like `02_1_raw.mat`.

The script intentionally supports two modes:

1. inspect
   - list dataset keys / shapes / dtypes in a raw `.mat`
   - use this first when you are unsure about raw key names

2. convert
   - extract per-song EEG slices from raw recordings
   - optionally apply paper-style robust scaling + std clamp
   - save `songXX_Imputed.mat` files compatible with the existing dataset

Because raw files vary by export, the conversion step is explicit about the
important assumptions instead of trying to guess too much.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


def _import_or_exit():
    missing: list[str] = []
    try:
        import h5py  # type: ignore
    except ImportError:
        h5py = None
        missing.append("h5py")

    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
        missing.append("numpy")

    try:
        from scipy.io import savemat  # type: ignore
    except ImportError:
        savemat = None
        missing.append("scipy")

    if missing:
        deps = ", ".join(missing)
        raise SystemExit(
            f"Missing Python dependencies: {deps}. "
            "Install them in the environment that will preprocess raw NMED-T files."
        )
    return h5py, np, savemat


def _walk_hdf5(obj, prefix: str = "") -> Iterable[tuple[str, str, tuple[int, ...] | None, str | None]]:
    for key in obj.keys():
        item = obj[key]
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            yield path, "dataset", tuple(item.shape), str(item.dtype)
        else:
            yield path, "group", None, None
            yield from _walk_hdf5(item, path)


def inspect_raw_file(path: Path) -> None:
    h5py, _, _ = _import_or_exit()
    with h5py.File(path, "r") as f:
        for key_path, kind, shape, dtype in _walk_hdf5(f):
            if kind == "dataset":
                print(f"{key_path}\tdataset\tshape={shape}\tdtype={dtype}")
            else:
                print(f"{key_path}\tgroup")


def _read_dataset(file_path: Path, key: str):
    h5py, np, _ = _import_or_exit()
    with h5py.File(file_path, "r") as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not found in {file_path}. Run inspect mode first.")
        arr = np.array(f[key])
    return arr


def _percentile_scale(signal, np):
    """
    Robust per-channel scaling using median / IQR.
    Input shape: [channels, time]
    """
    med = np.median(signal, axis=1, keepdims=True)
    q75 = np.percentile(signal, 75.0, axis=1, keepdims=True)
    q25 = np.percentile(signal, 25.0, axis=1, keepdims=True)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    return (signal - med) / iqr


def _std_clamp(signal, clamp_std: float, np):
    std = np.std(signal, axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    limit = clamp_std * std
    return np.clip(signal, -limit, limit)


def _downsample_linear(signal, src_fs: int, dst_fs: int, np):
    if src_fs == dst_fs:
        return signal.astype(np.float32, copy=False)

    if src_fs <= 0 or dst_fs <= 0:
        raise ValueError(f"Sampling rates must be positive, got src={src_fs}, dst={dst_fs}")

    n_channels, n_src = signal.shape
    duration = n_src / float(src_fs)
    n_dst = int(round(duration * dst_fs))
    if n_dst <= 1:
        raise ValueError("Downsampled signal would be empty; check source shape / sampling rates.")

    src_idx = np.linspace(0.0, n_src - 1, n_src, dtype=np.float64)
    dst_idx = np.linspace(0.0, n_src - 1, n_dst, dtype=np.float64)
    out = np.empty((n_channels, n_dst), dtype=np.float32)
    for ch in range(n_channels):
        out[ch] = np.interp(dst_idx, src_idx, signal[ch]).astype(np.float32)
    return out


def _parse_song_map(text: str) -> dict[str, list[str]]:
    """
    Example:
      '{"02_1_raw.mat":["song21","song22"],"02_2_raw.mat":["song23","song24"]}'
    """
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("recording-song-map must be a JSON object.")

    normalized: dict[str, list[str]] = {}
    for k, v in parsed.items():
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise ValueError("Each recording-song-map value must be a list of song names.")
        normalized[str(k)] = [str(item) for item in v]
    return normalized


def _ensure_2d_eeg(arr, np):
    """
    Normalize raw EEG array into [channels, time].
    We keep this strict to avoid silently scrambling axes.
    """
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D EEG array from the raw file, got shape={arr.shape}. "
            "If your raw export stores trials/metadata differently, inspect the file first."
        )

    channels_first = arr.shape[0] < arr.shape[1]
    eeg = arr if channels_first else arr.T
    return eeg.astype(np.float32, copy=False)


def convert_recordings(
    raw_dir: Path,
    output_dir: Path,
    eeg_key: str,
    recording_song_map: dict[str, list[str]],
    src_fs: int,
    dst_fs: int,
    seconds_per_song: float,
    robust_scale: bool,
    clamp_std: float | None,
) -> None:
    _, np, savemat = _import_or_exit()

    song_accumulator: dict[str, list] = {}
    samples_per_song_src = int(round(seconds_per_song * src_fs))

    if samples_per_song_src <= 0:
        raise ValueError("seconds-per-song must produce a positive sample count.")

    for recording_name, songs in recording_song_map.items():
        recording_path = raw_dir / recording_name
        if not recording_path.exists():
            raise FileNotFoundError(f"Recording not found: {recording_path}")

        raw = _read_dataset(recording_path, eeg_key)
        eeg = _ensure_2d_eeg(raw, np)

        expected_total = samples_per_song_src * len(songs)
        if eeg.shape[1] < expected_total:
            raise ValueError(
                f"{recording_name} has only {eeg.shape[1]} time samples, but "
                f"{len(songs)} songs x {samples_per_song_src} samples/song "
                f"requires at least {expected_total}. "
                "Update recording-song-map or seconds-per-song."
            )

        for song_index, song_name in enumerate(songs):
            start = song_index * samples_per_song_src
            stop = start + samples_per_song_src
            song_signal = eeg[:, start:stop]

            if robust_scale:
                song_signal = _percentile_scale(song_signal, np)
            if clamp_std is not None:
                song_signal = _std_clamp(song_signal, clamp_std, np)

            song_signal = _downsample_linear(song_signal, src_fs=src_fs, dst_fs=dst_fs, np=np)
            song_signal = song_signal.astype(np.float32, copy=False)
            song_accumulator.setdefault(song_name, []).append(song_signal)

    output_dir.mkdir(parents=True, exist_ok=True)

    for song_name, subject_signals in sorted(song_accumulator.items()):
        if not subject_signals:
            continue
        stacked = np.stack(subject_signals, axis=2).astype(np.float32, copy=False)
        song_num = "".join(ch for ch in song_name if ch.isdigit())
        if not song_num:
            raise ValueError(f"Could not infer song number from '{song_name}'.")

        out_path = output_dir / f"{song_name}_Imputed.mat"
        data_key = f"data{song_num}"
        savemat(out_path, {data_key: stacked})
        print(f"saved {out_path} key={data_key} shape={stacked.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect raw NMED-T MATLAB v7.3 files and convert them to song-level mats."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="List keys / shapes in a raw .mat file.")
    inspect_parser.add_argument("--file", type=Path, required=True, help="Path to a raw MATLAB v7.3 file.")

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert raw recordings into song-level .mat files compatible with the current dataset code.",
    )
    convert_parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing *_raw.mat files.")
    convert_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for song-level outputs.")
    convert_parser.add_argument(
        "--eeg-key",
        type=str,
        required=True,
        help="Root-level HDF5 key containing the raw EEG array. Use inspect mode first.",
    )
    convert_parser.add_argument(
        "--recording-song-map",
        type=str,
        required=True,
        help=(
            "JSON object mapping each raw recording filename to the ordered songs it contains. "
            'Example: {"02_1_raw.mat":["song21","song22"],"02_2_raw.mat":["song23","song24"]}'
        ),
    )
    convert_parser.add_argument(
        "--src-fs",
        type=int,
        default=1000,
        help="Sampling rate in the raw recording. Default assumes 1000 Hz raw EEG.",
    )
    convert_parser.add_argument(
        "--dst-fs",
        type=int,
        default=125,
        help="Sampling rate for model-ready output files. Default matches current repo assumptions.",
    )
    convert_parser.add_argument(
        "--seconds-per-song",
        type=float,
        required=True,
        help="Duration of each contiguous song block in the raw recordings, in seconds.",
    )
    convert_parser.add_argument(
        "--robust-scale",
        action="store_true",
        help="Apply per-channel median / IQR scaling before saving.",
    )
    convert_parser.add_argument(
        "--clamp-std",
        type=float,
        default=None,
        help="Clamp each channel to +/- N std after optional scaling. Use 20 to mirror the paper.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        inspect_raw_file(args.file)
        return

    if args.command == "convert":
        recording_song_map = _parse_song_map(args.recording_song_map)
        convert_recordings(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            eeg_key=args.eeg_key,
            recording_song_map=recording_song_map,
            src_fs=args.src_fs,
            dst_fs=args.dst_fs,
            seconds_per_song=args.seconds_per_song,
            robust_scale=args.robust_scale,
            clamp_std=args.clamp_std,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
