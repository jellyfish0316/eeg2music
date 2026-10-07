from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the temporal-shift curve from `run_paper_evaluation.py --stage summarize` outputs."
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to metrics/temporal_shift/temporal_shift_summary.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--use-seconds",
        action="store_true",
        help="Plot the x-axis in seconds instead of chunks.",
    )
    return parser.parse_args()


def load_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _offset_field(use_seconds: bool) -> str:
    return "offset_seconds" if use_seconds else "offset_chunks"


def _axis_label(use_seconds: bool) -> str:
    return "Signed EEG-audio shift (seconds)" if use_seconds else "Signed EEG-audio shift (chunks)"


def plot_temporal_shift_curve(summary: dict[str, Any], *, output_dir: Path, use_seconds: bool) -> Path:
    """Plot 1: mean CLAP vs. signed shift, one line per model, with uncertainty bands."""

    field = _offset_field(use_seconds)
    fig, ax = plt.subplots(figsize=(7, 5))
    for job in summary.get("jobs", []):
        offsets = sorted(job["offsets"], key=lambda row: row["offset_chunks"])
        points = [
            (row[field], row["mean_clap"], row.get("sem_clap"))
            for row in offsets
            if row.get("mean_clap") is not None
        ]
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        sem = [p[2] if p[2] is not None else 0.0 for p in points]
        lo = [yi - 1.96 * si for yi, si in zip(y, sem)]
        hi = [yi + 1.96 * si for yi, si in zip(y, sem)]
        ax.plot(x, y, marker="o", label=str(job["model_type"]))
        ax.fill_between(x, lo, hi, alpha=0.2)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, label="correct (offset 0)")
    ax.set_xlabel(_axis_label(use_seconds))
    ax.set_ylabel("Mean CLAP audio similarity")
    ax.set_title("Temporal-shift curve")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "temporal_shift_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_correct_minus_shift(summary: dict[str, Any], *, output_dir: Path, use_seconds: bool) -> Path:
    """Plot 2: CLAP(correct) - CLAP(shifted) vs. shift, one line per model, 95% CI band."""

    field = _offset_field(use_seconds)
    fig, ax = plt.subplots(figsize=(7, 5))
    for job in summary.get("jobs", []):
        offsets = sorted(
            (row for row in job["offsets"] if row["offset_chunks"] != 0),
            key=lambda row: row["offset_chunks"],
        )
        points = []
        for row in offsets:
            contrast = row.get("correct_minus_shift_common_support") or row.get("correct_minus_shift_max_available")
            if contrast is None:
                continue
            points.append((row[field], contrast["mean_difference"], contrast["ci95_low"], contrast["ci95_high"]))
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        lo = [p[2] for p in points]
        hi = [p[3] for p in points]
        ax.plot(x, y, marker="o", label=str(job["model_type"]))
        ax.fill_between(x, lo, hi, alpha=0.2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(_axis_label(use_seconds))
    ax.set_ylabel("CLAP(correct) - CLAP(shifted)")
    ax.set_title("Correct-minus-shift difference")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "temporal_shift_correct_minus_shift.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_participant_stratified(summary: dict[str, Any], *, output_dir: Path, use_seconds: bool) -> Path | None:
    """Plot 3: one panel per participant so subject variability is not hidden behind a pooled mean."""

    field = _offset_field(use_seconds)
    participants = sorted(
        {
            subject
            for job in summary.get("jobs", [])
            for row in job["offsets"]
            for subject in row.get("participant_stratified_mean_clap", {})
        }
    )
    if not participants:
        return None

    fig, axes = plt.subplots(1, len(participants), figsize=(5 * len(participants), 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, participant in zip(axes, participants):
        for job in summary.get("jobs", []):
            offsets = sorted(job["offsets"], key=lambda row: row["offset_chunks"])
            points = [
                (row[field], row["participant_stratified_mean_clap"].get(participant))
                for row in offsets
                if row["participant_stratified_mean_clap"].get(participant) is not None
            ]
            if not points:
                continue
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            ax.plot(x, y, marker="o", label=str(job["model_type"]))
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Participant {participant}")
        ax.set_xlabel(_axis_label(use_seconds))
    axes[0].set_ylabel("Mean CLAP audio similarity")
    axes[0].legend()
    fig.tight_layout()
    output_path = output_dir / "temporal_shift_participant_stratified.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = plot_temporal_shift_curve(summary, output_dir=output_dir, use_seconds=bool(args.use_seconds))
    difference_path = plot_correct_minus_shift(summary, output_dir=output_dir, use_seconds=bool(args.use_seconds))
    participant_path = plot_participant_stratified(summary, output_dir=output_dir, use_seconds=bool(args.use_seconds))
    for path in (curve_path, difference_path, participant_path):
        if path is not None:
            print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
