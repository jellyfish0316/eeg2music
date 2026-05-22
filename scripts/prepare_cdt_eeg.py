from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.io import savemat


DEFAULT_CONDITION_TRIGGERS: dict[str, tuple[int, int]] = {
    "drum": (21, 31),
    "vocal": (22, 32),
    "guitar": (23, 33),
    "passive": (24, 34),
}


def _import_mne():
    try:
        import mne  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "This script needs MNE to read Curry/CDT files. Install it in your env with:\n"
            "  pip install mne\n"
            "Then rerun this command."
        ) from exc
    return mne


def _robust_scale(signal: np.ndarray) -> np.ndarray:
    median = np.median(signal, axis=1, keepdims=True)
    q75 = np.percentile(signal, 75.0, axis=1, keepdims=True)
    q25 = np.percentile(signal, 25.0, axis=1, keepdims=True)
    iqr = np.where((q75 - q25) < 1e-6, 1.0, q75 - q25)
    return ((signal - median) / iqr).astype(np.float32, copy=False)


def _center_using_first_samples(signal: np.ndarray, center_n: int) -> np.ndarray:
    if center_n <= 0:
        return signal.astype(np.float32, copy=False)
    use_n = min(center_n, signal.shape[1])
    baseline = signal[:, :use_n].mean(axis=1, keepdims=True)
    return (signal - baseline).astype(np.float32, copy=False)


def _std_clamp(signal: np.ndarray, clamp_std: float | None) -> np.ndarray:
    if clamp_std is None:
        return signal.astype(np.float32, copy=False)
    std = signal.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    limit = clamp_std * std
    return np.clip(signal, -limit, limit).astype(np.float32, copy=False)


def _resample_linear(signal: np.ndarray, *, src_fs: float, dst_fs: float) -> np.ndarray:
    if abs(float(src_fs) - float(dst_fs)) < 1e-6:
        return signal.astype(np.float32, copy=False)
    n_channels, n_src = signal.shape
    duration = n_src / float(src_fs)
    n_dst = int(round(duration * float(dst_fs)))
    if n_dst <= 1:
        raise ValueError("Resampled signal would be empty; check source length / sampling rates.")
    src_idx = np.linspace(0.0, n_src - 1, n_src, dtype=np.float64)
    dst_idx = np.linspace(0.0, n_src - 1, n_dst, dtype=np.float64)
    out = np.empty((n_channels, n_dst), dtype=np.float32)
    for ch_idx in range(n_channels):
        out[ch_idx] = np.interp(dst_idx, src_idx, signal[ch_idx]).astype(np.float32)
    return out


def _infer_data_key(output_path: Path, song_name: str | None) -> str:
    text = song_name or output_path.stem
    match = re.search(r"song\s*([0-9]+)", text, flags=re.IGNORECASE)
    if match:
        return f"data{match.group(1)}"
    digits = re.findall(r"[0-9]+", text)
    if digits:
        return f"data{digits[-1]}"
    return "data"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_int(text: str, key: str) -> int:
    match = re.search(rf"{re.escape(key)}\s*=\s*(-?\d+)", text)
    if match is None:
        raise ValueError(f"Could not find {key!r}.")
    return int(match.group(1))


def _extract_float(text: str, key: str) -> float:
    match = re.search(rf"{re.escape(key)}\s*=\s*(-?\d+(?:\.\d+)?)", text)
    if match is None:
        raise ValueError(f"Could not find {key!r}.")
    return float(match.group(1))


def _extract_list(text: str, name: str) -> list[str]:
    start_marker = f"{name} START_LIST"
    end_marker = f"{name} END_LIST"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        return []
    lines = text[start:end].splitlines()[1:]
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _parse_trigger_code(description: object) -> int | None:
    if not isinstance(description, str):
        return None
    digits = re.findall(r"[0-9]+", description)
    if not digits:
        return None
    return int(digits[-1])


def _parse_curry_events(ceo_path: Path) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    for line in _extract_list(_read_text(ceo_path), "NUMBER_LIST"):
        nums = [int(x) for x in re.findall(r"-?\d+", line)]
        if len(nums) >= 3:
            events.append((nums[0], nums[2]))
    return events


def _fallback_curry_paths(path: Path) -> tuple[Path, Path, Path]:
    data_path = path
    dpa_path = path.with_suffix(path.suffix + ".dpa")
    ceo_path = path.with_suffix(path.suffix + ".ceo")
    if not dpa_path.exists():
        raise FileNotFoundError(f"Missing Curry header: {dpa_path}")
    if not ceo_path.exists():
        raise FileNotFoundError(f"Missing Curry events: {ceo_path}")
    return data_path, dpa_path, ceo_path


def _read_curry_fallback_segment(
    path: Path,
    *,
    condition: str,
    start_code: int,
    end_code: int,
    occurrence: int,
    fallback_duration: float | None,
    keep_eeg_channels: int | None,
    scale: float,
) -> tuple[np.ndarray, float, list[str], tuple[int, int, int]]:
    data_path, dpa_path, ceo_path = _fallback_curry_paths(path)
    header = _read_text(dpa_path)
    sfreq = _extract_float(header, "SampleFreqHz")
    header_end_sample = _extract_int(header, "NumSamples")
    n_channels = _extract_int(header, "NumChannels")
    n_eeg_channels = _extract_int(header, "NumChanThisGroup")
    labels = _extract_list(header, "LABELS")[:n_eeg_channels]
    if len(labels) < n_eeg_channels:
        labels = [f"EEG{idx + 1:03d}" for idx in range(n_eeg_channels)]

    item_count = data_path.stat().st_size // np.dtype("<f4").itemsize
    if item_count % n_channels != 0:
        raise ValueError(
            f"{path}: binary size is not divisible by {n_channels} float32 channels. "
            f"items={item_count}"
        )
    n_local_samples = item_count // n_channels
    sample_offset = header_end_sample - n_local_samples

    matches: list[tuple[int, int]] = []
    starts = [sample for sample, code in _parse_curry_events(ceo_path) if code == start_code]
    ends = [sample for sample, code in _parse_curry_events(ceo_path) if code == end_code]
    for start_abs in starts:
        end_candidates = [end_abs for end_abs in ends if end_abs > start_abs]
        if end_candidates:
            end_abs = end_candidates[0]
        elif fallback_duration is not None:
            end_abs = start_abs + int(round(float(fallback_duration) * sfreq))
        else:
            continue
        start_local = start_abs - sample_offset
        end_local = end_abs - sample_offset
        if 0 <= start_local < end_local <= n_local_samples:
            matches.append((start_local, end_local))

    if len(matches) < occurrence:
        raise ValueError(
            f"{path}: no usable interval for {condition} triggers={start_code}->{end_code}. "
            f"sample_offset={sample_offset} local_samples={n_local_samples} matches={matches}"
        )
    start_local, end_local = matches[occurrence - 1]

    keep = keep_eeg_channels if keep_eeg_channels is not None else n_eeg_channels
    if keep > n_eeg_channels:
        raise ValueError(f"{path}: only {n_eeg_channels} EEG channels; cannot keep {keep}.")

    data = np.memmap(data_path, dtype="<f4", mode="r", shape=(n_local_samples, n_channels))
    signal = np.asarray(data[start_local:end_local, :keep].T, dtype=np.float32) * float(scale)
    return signal, sfreq, labels[:keep], (sample_offset + start_local, sample_offset + end_local, sample_offset)


def _get_eeg_picks_and_names(raw, *, keep_eeg_channels: int | None) -> tuple[np.ndarray, list[str]]:
    mne = _import_mne()
    picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, exclude=[])
    if len(picks) == 0:
        raise ValueError("No EEG channels found by MNE.")
    if keep_eeg_channels is not None:
        if len(picks) < keep_eeg_channels:
            raise ValueError(f"Only {len(picks)} EEG channels; cannot keep {keep_eeg_channels}.")
        picks = picks[:keep_eeg_channels]
    return picks, [raw.ch_names[idx] for idx in picks]


def _preprocess_signal(
    signal: np.ndarray,
    *,
    robust_scale: bool,
    center_using_first_samples: int,
    clamp_std: float | None,
) -> np.ndarray:
    if robust_scale:
        signal = _robust_scale(signal)
    signal = _center_using_first_samples(signal, center_using_first_samples)
    signal = _std_clamp(signal, clamp_std)
    return signal.astype(np.float32, copy=False)


def _load_cdt_signal(
    path: Path,
    *,
    keep_eeg_channels: int | None,
    tmin: float,
    duration: float | None,
    dst_fs: float | None,
    scale: float,
    robust_scale: bool,
    center_using_first_samples: int,
    clamp_std: float | None,
) -> tuple[np.ndarray, float, list[str]]:
    mne = _import_mne()
    raw = mne.io.read_raw_curry(str(path), preload=True, verbose="ERROR")

    if duration is not None:
        raw.crop(tmin=tmin, tmax=tmin + duration, include_tmax=False)
    elif tmin > 0:
        raw.crop(tmin=tmin, include_tmax=False)

    if dst_fs is not None and abs(float(raw.info["sfreq"]) - float(dst_fs)) > 1e-6:
        raw.resample(float(dst_fs), npad="auto", verbose="ERROR")

    try:
        picks, channel_names = _get_eeg_picks_and_names(raw, keep_eeg_channels=keep_eeg_channels)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    signal = raw.get_data(picks=picks).astype(np.float32, copy=False) * float(scale)
    signal = _preprocess_signal(
        signal,
        robust_scale=robust_scale,
        center_using_first_samples=center_using_first_samples,
        clamp_std=clamp_std,
    )
    return signal.astype(np.float32, copy=False), float(raw.info["sfreq"]), channel_names


def inspect_cdt(path: Path) -> None:
    mne = _import_mne()
    raw = mne.io.read_raw_curry(str(path), preload=False, verbose="ERROR")
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, ecg=False, exclude=[])
    print(f"file: {path}")
    print(f"sfreq: {raw.info['sfreq']}")
    print(f"n_channels: {len(raw.ch_names)}")
    print(f"n_eeg_channels: {len(eeg_picks)}")
    print(f"n_times: {raw.n_times}")
    print(f"duration_sec: {raw.n_times / float(raw.info['sfreq']):.3f}")
    print(f"first_eeg_channels: {[raw.ch_names[idx] for idx in eeg_picks[:10]]}")
    if len(raw.annotations) > 0:
        print(f"annotations: {len(raw.annotations)}")
        for ann in raw.annotations[:10]:
            print(f"  onset={ann['onset']:.3f} duration={ann['duration']:.3f} desc={ann['description']}")


def convert_cdts(args: argparse.Namespace) -> None:
    signals: list[np.ndarray] = []
    sfreqs: list[float] = []
    channel_names: list[str] | None = None

    for path in args.files:
        signal, sfreq, names = _load_cdt_signal(
            path,
            keep_eeg_channels=args.keep_eeg_channels,
            tmin=args.tmin,
            duration=args.duration,
            dst_fs=args.dst_fs,
            scale=args.scale,
            robust_scale=not args.no_robust_scale,
            center_using_first_samples=args.center_using_first_samples,
            clamp_std=args.clamp_std,
        )
        if channel_names is None:
            channel_names = names
        elif names != channel_names:
            raise ValueError(f"{path}: EEG channel names/order differ from the first file.")
        signals.append(signal)
        sfreqs.append(sfreq)
        print(f"loaded {path} shape={signal.shape} sfreq={sfreq:g}")

    min_time = min(signal.shape[1] for signal in signals)
    if len({signal.shape for signal in signals}) > 1:
        if not args.trim_to_shortest:
            shapes = [signal.shape for signal in signals]
            raise ValueError(f"Subject shapes differ: {shapes}. Rerun with --trim-to-shortest if this is expected.")
        signals = [signal[:, :min_time] for signal in signals]

    stacked = np.stack(signals, axis=2).astype(np.float32, copy=False)
    data_key = args.data_key or _infer_data_key(args.output, args.song_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        args.output,
        {
            data_key: stacked,
            "fs": np.asarray([[sfreqs[0]]], dtype=np.float32),
            "channel_names": np.asarray(channel_names or [], dtype=object),
        },
    )
    print(f"saved {args.output} key={data_key} shape={stacked.shape}")


def _condition_trigger_map(raw_specs: list[str] | None) -> dict[str, tuple[int, int]]:
    out = dict(DEFAULT_CONDITION_TRIGGERS)
    if not raw_specs:
        return out
    for spec in raw_specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid condition trigger spec {spec!r}; expected name:start:end")
        name, start, end = parts
        out[name] = (int(start), int(end))
    return out


def _find_condition_interval(
    raw,
    *,
    start_code: int,
    end_code: int,
    occurrence: int,
    fallback_duration: float | None,
) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    for ann in raw.annotations:
        code = _parse_trigger_code(ann["description"])
        if code == start_code:
            starts.append(float(ann["onset"]))
        elif code == end_code:
            ends.append(float(ann["onset"]))

    if len(starts) < occurrence:
        raise ValueError(f"Could not find occurrence {occurrence} of start trigger {start_code}. starts={starts}")
    start = starts[occurrence - 1]
    end_candidates = [end for end in ends if end > start]
    if end_candidates:
        return start, end_candidates[0]
    if fallback_duration is None:
        raise ValueError(f"Could not find end trigger {end_code} after start trigger {start_code}.")
    return start, start + float(fallback_duration)


def convert_cdt_events(args: argparse.Namespace) -> None:
    all_conditions = _condition_trigger_map(args.condition_trigger)
    selected_conditions = args.conditions or list(all_conditions)
    unknown = sorted(set(selected_conditions) - set(all_conditions))
    if unknown:
        raise ValueError(f"Unknown conditions {unknown}; available={sorted(all_conditions)}")

    by_condition: dict[str, list[np.ndarray]] = {name: [] for name in selected_conditions}
    sfreq_by_file: list[float] = []
    channel_names: list[str] | None = None

    for path in args.files:
        file_sfreq: float | None = None
        for condition in selected_conditions:
            start_code, end_code = all_conditions[condition]
            signal, sfreq, names, abs_window = _read_curry_fallback_segment(
                path,
                condition=condition,
                start_code=start_code,
                end_code=end_code,
                occurrence=args.occurrence,
                fallback_duration=args.duration,
                keep_eeg_channels=args.keep_eeg_channels,
                scale=args.scale,
            )
            if args.dst_fs is not None and abs(float(sfreq) - float(args.dst_fs)) > 1e-6:
                signal = _resample_linear(signal, src_fs=sfreq, dst_fs=float(args.dst_fs))
                sfreq = float(args.dst_fs)
            if channel_names is None:
                channel_names = names
            elif names != channel_names:
                raise ValueError(f"{path}: EEG channel names/order differ from the first file.")
            if file_sfreq is None:
                file_sfreq = float(sfreq)
            elif abs(file_sfreq - float(sfreq)) > 1e-6:
                raise ValueError(f"{path}: condition sampling rates differ unexpectedly.")
            signal = _preprocess_signal(
                signal,
                robust_scale=not args.no_robust_scale,
                center_using_first_samples=args.center_using_first_samples,
                clamp_std=args.clamp_std,
            )
            by_condition[condition].append(signal)
            print(
                f"loaded {path} condition={condition} triggers={start_code}->{end_code} "
                f"abs_samples={abs_window[0]}:{abs_window[1]} offset={abs_window[2]} shape={signal.shape}"
            )
        if file_sfreq is None:
            raise ValueError(f"{path}: no conditions were loaded.")
        sfreq_by_file.append(file_sfreq)

    for condition, signals in by_condition.items():
        min_time = min(signal.shape[1] for signal in signals)
        if len({signal.shape for signal in signals}) > 1:
            if not args.trim_to_shortest:
                shapes = [signal.shape for signal in signals]
                raise ValueError(
                    f"{condition}: subject shapes differ: {shapes}. "
                    "Rerun with --trim-to-shortest if this is expected."
                )
            signals = [signal[:, :min_time] for signal in signals]

        stacked = np.stack(signals, axis=2).astype(np.float32, copy=False)
        data_key = args.data_key or _infer_data_key(Path(args.song_name), args.song_name)
        out_path = args.output_dir / f"{args.song_name}_{condition}_{args.output_suffix}.mat"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(
            out_path,
            {
                data_key: stacked,
                "fs": np.asarray([[sfreq_by_file[0]]], dtype=np.float32),
                "channel_names": np.asarray(channel_names or [], dtype=object),
                "condition": np.asarray([[condition]], dtype=object),
            },
        )
        print(f"saved {out_path} key={data_key} condition={condition} shape={stacked.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or convert Curry/Neuroscan .cdt EEG files.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print sampling rate, EEG channel count, and annotations.")
    inspect_parser.add_argument("file", type=Path)

    convert_parser = subparsers.add_parser("convert", help="Convert one or more .cdt files to repo-compatible .mat.")
    convert_parser.add_argument("files", type=Path, nargs="+", help="One CDT per subject, already aligned to the song.")
    convert_parser.add_argument("--output", type=Path, required=True, help="Output .mat path.")
    convert_parser.add_argument("--song-name", type=str, default=None, help="Used to infer data_key, e.g. song21.")
    convert_parser.add_argument("--data-key", type=str, default=None, help="Override output key, e.g. data21.")
    convert_parser.add_argument("--keep-eeg-channels", type=int, default=None, help="Keep the first N EEG channels.")
    convert_parser.add_argument("--tmin", type=float, default=0.0, help="Crop start time in seconds.")
    convert_parser.add_argument("--duration", type=float, default=None, help="Crop duration in seconds.")
    convert_parser.add_argument("--dst-fs", type=float, default=None, help="Resample to this EEG sampling rate.")
    convert_parser.add_argument("--scale", type=float, default=1.0, help="Multiply raw MNE data before preprocessing.")
    convert_parser.add_argument("--no-robust-scale", action="store_true", help="Do not median/IQR scale each channel.")
    convert_parser.add_argument("--center-using-first-samples", type=int, default=1000)
    convert_parser.add_argument("--clamp-std", type=float, default=None)
    convert_parser.add_argument("--trim-to-shortest", action="store_true")

    events_parser = subparsers.add_parser(
        "convert-events",
        help="Cut continuous CDT files by trigger pairs and save repo-compatible .mat files.",
    )
    events_parser.add_argument("files", type=Path, nargs="+", help="One continuous CDT per subject.")
    events_parser.add_argument("--output-dir", type=Path, required=True)
    events_parser.add_argument("--song-name", type=str, required=True, help="Example: song7.")
    events_parser.add_argument("--data-key", type=str, default=None, help="Override output key, e.g. data7.")
    events_parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Conditions to export. Default: drum vocal guitar passive.",
    )
    events_parser.add_argument(
        "--condition-trigger",
        type=str,
        action="append",
        default=None,
        help="Override/add trigger pair as name:start:end, e.g. passive:24:34.",
    )
    events_parser.add_argument("--occurrence", type=int, default=1, help="Which repeated trial occurrence to export.")
    events_parser.add_argument("--duration", type=float, default=None, help="Fallback duration if end trigger is missing.")
    events_parser.add_argument("--output-suffix", type=str, default="Processed")
    events_parser.add_argument("--keep-eeg-channels", type=int, default=None)
    events_parser.add_argument("--dst-fs", type=float, default=None)
    events_parser.add_argument("--scale", type=float, default=1.0)
    events_parser.add_argument("--no-robust-scale", action="store_true")
    events_parser.add_argument("--center-using-first-samples", type=int, default=1000)
    events_parser.add_argument("--clamp-std", type=float, default=None)
    events_parser.add_argument("--trim-to-shortest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "inspect":
        inspect_cdt(args.file)
    elif args.cmd == "convert":
        convert_cdts(args)
    elif args.cmd == "convert-events":
        convert_cdt_events(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
