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
   - apply the paper-aligned minimal preprocessing requested here
   - save `songXX_Imputed.mat` files compatible with the existing dataset

Because raw files vary by export, the conversion step is explicit about the
important assumptions instead of trying to guess too much.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import h5py  # type: ignore
import numpy as np  # type: ignore
from scipy.io import savemat  # type: ignore

RAW_TO_CLEAN_SUBJECT = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 9,
    12: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    19: 16,
    20: 17,
    21: 18,
    23: 19,
}

SONG_DURATION_S = {
    21: 278,  # 4:38
    22: 271,  # 4:31
    23: 276,  # 4:36
    24: 294,  # 4:54
    25: 289,  # 4:49
    26: 276,  # 4:36
    27: 292,  # 4:52
    28: 292,  # 4:52
    29: 294,  # 4:54
    30: 298,  # 4:58
}


# Inspect helpers
def _walk_hdf5(obj, prefix: str = "") -> Iterable[tuple[str, str, tuple[int, ...] | None, str | None]]:
    for key in obj.keys():
        item = obj[key]
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if hasattr(item, "shape") and hasattr(item, "dtype"):
            yield path, "dataset", tuple(item.shape), str(item.dtype)
        else:
            yield path, "group", None, None
            yield from _walk_hdf5(item, path)


def _decode_matlab_char(arr, np) -> str | None:
    arr = np.asarray(arr)
    if arr.size == 0:
        return ""
    flat = arr.reshape(-1)
    if flat.dtype.kind in {"u", "i"}:
        try:
            codes = [int(v) for v in flat if int(v) != 0]
            if not codes:
                return ""
            return "".join(chr(c) for c in codes)
        except Exception:
            return None
    return None


def _read_scalar_like(value, h5_file, np):
    if hasattr(value, "dtype") and str(value.dtype) == "object":
        raw = np.array(value)
        if raw.size == 1:
            ref = raw.reshape(-1)[0]
            if ref:
                return _read_scalar_like(h5_file[ref], h5_file, np)
        return f"<object shape={tuple(raw.shape)}>"

    arr = np.array(value)
    if arr.size == 1:
        return arr.reshape(-1)[0].item()

    maybe_text = _decode_matlab_char(arr, np)
    if maybe_text is not None:
        return maybe_text

    return f"<array shape={tuple(arr.shape)} dtype={arr.dtype}>"


def _preview_object_dataset(dataset, h5_file, np, rows: int = 8) -> list[list[object]]:
    raw = np.array(dataset)
    if raw.dtype != object:
        return []

    if raw.ndim == 1:
        raw = raw[:, None]

    out: list[list[object]] = []
    for row_idx in range(min(rows, raw.shape[0])):
        row: list[object] = []
        for col_idx in range(raw.shape[1]):
            ref = raw[row_idx, col_idx]
            if not ref:
                row.append(None)
                continue
            row.append(_read_scalar_like(h5_file[ref], h5_file, np))
        out.append(row)
    return out


def _collect_first_col_strings(dataset, h5_file, np, limit: int | None = None) -> list[str]:
    raw = np.array(dataset)
    if raw.dtype != object:
        return []
    if raw.ndim == 1:
        raw = raw[:, None]
    values: list[str] = []
    max_rows = raw.shape[0] if limit is None else min(limit, raw.shape[0])
    for row_idx in range(max_rows):
        ref = raw[row_idx, 0]
        if not ref:
            continue
        value = _read_scalar_like(h5_file[ref], h5_file, np)
        if isinstance(value, str):
            values.append(value)
    return values


def inspect_raw_file(path: Path, *, preview_rows: int = 8) -> None:
    with h5py.File(path, "r") as f:
        for key_path, kind, shape, dtype in _walk_hdf5(f):
            if kind == "dataset":
                print(f"{key_path}\tdataset\tshape={shape}\tdtype={dtype}")
                if key_path.count("/") == 1:
                    ds = f[key_path.lstrip("/")]
                    if shape == (1, 1):
                        try:
                            value = _read_scalar_like(ds, f, np)
                            print(f"  value={value}")
                        except Exception:
                            pass
                    if dtype == "object":
                        try:
                            labels = _collect_first_col_strings(ds, f, np)
                            if labels:
                                unique_labels = sorted(set(labels))
                                print(f"  first_col_unique={unique_labels}")
                            preview = _preview_object_dataset(ds, f, np, rows=preview_rows)
                            if preview:
                                print("  preview:")
                                for row in preview:
                                    print(f"    {row}")
                        except Exception as exc:
                            print(f"  preview_error={type(exc).__name__}: {exc}")
            else:
                print(f"{key_path}\tgroup")


# Raw HDF5 read helpers


def _require_hdf5_key(h5_file, key: str, file_path: Path):
    if key not in h5_file:
        raise KeyError(f"Key '{key}' not found in {file_path}.")
    return h5_file[key]


def _read_dataset(file_path: Path, key: str):
    with h5py.File(file_path, "r") as f:
        try:
            return np.array(_require_hdf5_key(f, key, file_path))
        except KeyError as exc:
            raise KeyError(f"{exc.args[0]} Run inspect mode first.") from exc


def _read_hdf5_scalar(file_path: Path, key: str):
    with h5py.File(file_path, "r") as f:
        return _read_scalar_like(_require_hdf5_key(f, key, file_path), f, np)


def _read_hdf5_object_rows(file_path: Path, key: str) -> list[list[object]]:
    with h5py.File(file_path, "r") as f:
        ds = _require_hdf5_key(f, key, file_path)
        raw = np.array(ds)
        if raw.dtype != object:
            raise ValueError(f"Expected object dataset for {key}, got {raw.dtype}.")
        if raw.ndim == 1:
            raw = raw[:, None]
        rows: list[list[object]] = []
        for row_idx in range(raw.shape[0]):
            row: list[object] = []
            for col_idx in range(raw.shape[1]):
                ref = raw[row_idx, col_idx]
                if not ref:
                    row.append(None)
                    continue
                row.append(_read_scalar_like(f[ref], f, np))
            rows.append(row)
        return rows


# EEG preprocessing helpers


def _std_clamp(signal, clamp_std: float, np):
    std = np.std(signal, axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    limit = clamp_std * std
    return np.clip(signal, -limit, limit)


def _robust_scale(signal, np):
    median = np.median(signal, axis=1, keepdims=True)
    q75 = np.percentile(signal, 75.0, axis=1, keepdims=True)
    q25 = np.percentile(signal, 25.0, axis=1, keepdims=True)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    return ((signal - median) / iqr).astype(np.float32, copy=False)


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


def _center_using_first_samples(signal, center_n: int, np):
    if center_n <= 0:
        return signal.astype(np.float32, copy=False)
    use_n = min(center_n, signal.shape[1])
    baseline = signal[:, :use_n].mean(axis=1, keepdims=True)
    return (signal - baseline).astype(np.float32, copy=False)


def _preprocess_song_signal(
    song_signal,
    *,
    robust_scale: bool,
    center_using_first_samples: int,
    clamp_std: float | None,
    src_fs: int,
    dst_fs: int,
    np,
):
    if robust_scale:
        song_signal = _robust_scale(song_signal, np)
    song_signal = _center_using_first_samples(song_signal, center_using_first_samples, np)
    if clamp_std is not None:
        song_signal = _std_clamp(song_signal, clamp_std, np)
    song_signal = _downsample_linear(song_signal, src_fs=src_fs, dst_fs=dst_fs, np=np)
    return song_signal.astype(np.float32, copy=False)


# Conversion helpers


def _parse_trigger_label(label: object) -> int | None:
    if not isinstance(label, str):
        return None
    text = label.strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def _parse_din_events(file_path: Path, key: str) -> list[dict[str, object]]:
    rows = _read_hdf5_object_rows(file_path, key)
    events: list[dict[str, object]] = []
    for row in rows:
        if len(row) < 2:
            continue
        label = row[0]
        trigger_num = _parse_trigger_label(label)
        try:
            sample = int(round(float(row[1])))
        except Exception:
            continue
        events.append(
            {
                "label": label,
                "trigger": trigger_num,
                "sample": sample,
                "raw_row": row,
            }
        )
    return events


def _extract_subject_from_filename(path: Path) -> int:
    prefix = path.name.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError as exc:
        raise ValueError(f"Could not parse raw subject id from filename '{path.name}'.") from exc


def _find_click_corrected_onset(song_sample: int, click_samples: list[int], src_fs: int) -> int:
    click_offset = int(round(1.0 * src_fs))
    tolerance = int(round(2.5 * src_fs))
    for click_sample in click_samples:
        if song_sample <= click_sample <= song_sample + tolerance:
            return click_sample - click_offset
    return song_sample


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
    src_fs: int,
    dst_fs: int,
    clamp_std: float | None,
    din_key: str = "DIN_1",
    keep_eeg_channels: int = 124,
    output_suffix: str = "Processed",
    center_using_first_samples: int = 1000,
    robust_scale: bool = True,
) -> None:
    song_accumulator: dict[str, dict[int, np.ndarray]] = {}
    recording_paths = sorted(raw_dir.glob("*_raw.mat"))

    for recording_path in recording_paths:
        if not recording_path.exists():
            raise FileNotFoundError(f"Recording not found: {recording_path}")

        raw = _read_dataset(recording_path, eeg_key)
        eeg = _ensure_2d_eeg(raw, np)
        if keep_eeg_channels > 0:
            if eeg.shape[0] < keep_eeg_channels:
                raise ValueError(
                    f"{recording_path.name} has only {eeg.shape[0]} channels; "
                    f"cannot keep first {keep_eeg_channels}."
                )
            eeg = eeg[:keep_eeg_channels]

        raw_subject = _extract_subject_from_filename(recording_path)
        if raw_subject not in RAW_TO_CLEAN_SUBJECT:
            raise ValueError(
                f"Raw subject {raw_subject} not found in NMED-T raw->clean mapping. "
                "Expected raw ids 2..23 excluding 1, 18, 22."
            )
        clean_subject = RAW_TO_CLEAN_SUBJECT[raw_subject]

        file_fs = _read_hdf5_scalar(recording_path, "fs")
        if int(round(float(file_fs))) != int(src_fs):
            raise ValueError(
                f"{recording_path.name} fs={file_fs} disagrees with --src-fs={src_fs}."
            )

        events = _parse_din_events(recording_path, din_key)
        song_events = [e for e in events if e["trigger"] in SONG_DURATION_S]
        click_samples = [int(e["sample"]) for e in events if e["trigger"] == 128]
        if not song_events:
            continue

        for event in song_events:
            trigger = int(event["trigger"])
            song_name = f"song{trigger}"
            start = _find_click_corrected_onset(int(event["sample"]), click_samples, src_fs)
            duration_samples = int(round(SONG_DURATION_S[trigger] * src_fs))
            stop = start + duration_samples
            if start < 0 or stop > eeg.shape[1]:
                raise ValueError(
                    f"{recording_path.name} {song_name} slice [{start}:{stop}] is outside EEG length {eeg.shape[1]}."
                )
            song_signal = eeg[:, start:stop]
            song_signal = _preprocess_song_signal(
                song_signal,
                robust_scale=robust_scale,
                center_using_first_samples=center_using_first_samples,
                clamp_std=clamp_std,
                src_fs=src_fs,
                dst_fs=dst_fs,
                np=np,
            )
            song_accumulator.setdefault(song_name, {})[clean_subject] = song_signal

    output_dir.mkdir(parents=True, exist_ok=True)

    for song_name, by_subject in sorted(song_accumulator.items()):
        if not by_subject:
            continue
        expected_subjects = sorted(by_subject)
        if expected_subjects != list(range(20)):
            missing = sorted(set(range(20)) - set(by_subject))
            raise ValueError(
                f"{song_name} is missing some subjects after raw conversion. "
                f"present={sorted(by_subject)} missing={missing}"
            )
        ordered = [by_subject[idx] for idx in sorted(by_subject)]
        first_shape = ordered[0].shape
        for idx, arr in enumerate(ordered[1:], start=1):
            if arr.shape != first_shape:
                raise ValueError(
                    f"{song_name} subject {idx} shape mismatch: {arr.shape} vs {first_shape}."
                )
        stacked = np.stack(ordered, axis=2).astype(np.float32, copy=False)
        song_num = "".join(ch for ch in song_name if ch.isdigit())
        if not song_num:
            raise ValueError(f"Could not infer song number from '{song_name}'.")

        out_path = output_dir / f"{song_name}_{output_suffix}.mat"
        data_key = f"data{song_num}"
        savemat(out_path, {data_key: stacked})
        print(f"saved {out_path} key={data_key} shape={stacked.shape}")


# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect raw NMED-T MATLAB v7.3 files and convert them to song-level mats."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="List keys / shapes in a raw .mat file.")
    inspect_parser.add_argument("--file", type=Path, required=True, help="Path to a raw MATLAB v7.3 file.")
    inspect_parser.add_argument(
        "--preview-rows",
        type=int,
        default=8,
        help="Number of preview rows to print for object datasets like DIN_1.",
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert raw recordings into song-level .mat files compatible with the current dataset code.",
    )
    convert_parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing *_raw.mat files.")
    convert_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for song-level outputs.")
    convert_parser.add_argument(
        "--eeg-key",
        type=str,
        default="X",
        help="Root-level HDF5 key containing the raw EEG array. Use inspect mode first.",
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
        "--clamp-std",
        type=float,
        default=20.0,
        help="Clamp each channel to +/- N std after centering. Default 20 matches the paper.",
    )
    convert_parser.add_argument(
        "--din-key",
        type=str,
        default="DIN_1",
        help="Root-level HDF5 key containing raw trigger events.",
    )
    convert_parser.add_argument(
        "--keep-eeg-channels",
        type=int,
        default=124,
        help=(
            "Keep the first N EEG channels from X. "
            "For NMED-T raw, channels 125-128 are face electrodes and row 129 is the vertex reference, "
            "so the paper-aligned default is 124."
        ),
    )
    convert_parser.add_argument(
        "--output-suffix",
        type=str,
        default="Processed",
        help="Suffix used in saved song filenames, e.g. song21_Processed.mat.",
    )
    convert_parser.add_argument(
        "--center-using-first-samples",
        type=int,
        default=1000,
        help="Center each extracted song using the mean of its first N samples. Default matches the paper's 1000 samples.",
    )
    convert_parser.add_argument(
        "--robust-scale",
        action="store_true",
        help="Apply per-channel robust scaling using median/IQR before centering and clamping.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        inspect_raw_file(args.file, preview_rows=int(args.preview_rows))
        return

    if args.command == "convert":
        convert_recordings(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            eeg_key=args.eeg_key,
            src_fs=args.src_fs,
            dst_fs=args.dst_fs,
            clamp_std=args.clamp_std,
            din_key=args.din_key,
            keep_eeg_channels=int(args.keep_eeg_channels),
            output_suffix=str(args.output_suffix),
            center_using_first_samples=int(args.center_using_first_samples),
            robust_scale=bool(args.robust_scale),
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
