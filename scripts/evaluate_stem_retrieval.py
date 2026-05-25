from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa
import torch

from models.audioldm2_vae_wrapper import AudioLDM2VAEWrapper


DEFAULT_STEMS = ["drums", "other_bass", "vocals"]
DEFAULT_TARGET_MAP = {
    "drum": "drums",
    "drums": "drums",
    "bass": "other_bass",
    "guitar": "other_bass",
    "other": "other_bass",
    "piano": "other_bass",
    "vocal": "vocals",
    "vocals": "vocals",
}


def parse_key_values(items: list[str]) -> dict[str, list[str]]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        values = [part.strip() for part in value.replace(",", "+").split("+") if part.strip()]
        if not values:
            raise ValueError(f"Target map has no stem values: {item}")
        parsed[key.strip()] = values
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate whether generated audio retrieves the target instrument stem with CLAP similarity."
    )
    p.add_argument("--manifest", type=str, nargs="+", required=True)
    p.add_argument("--stems-root", type=str, default="data/SelfRecorded_songs/separated/htdemucs")
    p.add_argument("--stems", type=str, nargs="+", default=DEFAULT_STEMS)
    p.add_argument(
        "--target-map",
        type=str,
        nargs="*",
        default=[f"{k}={v}" for k, v in DEFAULT_TARGET_MAP.items()],
        help="Map attention condition to stem name(s), e.g. drum=drums vocal=vocals guitar=other_bass.",
    )
    p.add_argument(
        "--target-condition",
        type=str,
        default=None,
        help="Override target attention condition for every row. Useful when manifest condition_name is ambiguous.",
    )
    p.add_argument("--output-dir", type=str, default="outputs/stem_retrieval_eval")
    return p.parse_args()


def load_rows(paths: list[str]) -> list[tuple[Path, dict]]:
    rows = []
    for path_str in paths:
        path = Path(path_str)
        manifests = sorted(path.rglob("manifest.json")) if path.is_dir() else [path]
        if not manifests:
            raise FileNotFoundError(f"No manifest.json found under {path}")
        for manifest in manifests:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for row in payload.get("samples", []):
                rows.append((manifest, row))
    return rows


def resolve_audio_path(path_str: str, manifest_path: Path, subdir: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    # Some manifests may be moved with their generated/target folders. If the
    # original absolute/relative path is stale, recover by filename.
    fallback = manifest_path.parent / subdir / path.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Audio file not found: {path_str} (also tried {fallback})")


def infer_target_stems(condition_name: str, target_map: dict[str, list[str]], override: str | None) -> tuple[str, list[str]]:
    if override is not None:
        if override not in target_map:
            raise KeyError(f"--target-condition {override!r} is not in target map.")
        return override, target_map[override]

    parts = [part.strip() for part in condition_name.split("+") if part.strip()]
    matches = [(part, target_map[part]) for part in parts if part in target_map]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot infer one target stem from condition_name={condition_name!r}. "
            "Use --target-condition for ambiguous manifests."
        )
    return matches[0]


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


def load_stem_chunk(path: Path, sample_rate: int, offset: float, duration: float, n_samples: int) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True, offset=offset, duration=duration)
    if len(audio) < n_samples:
        audio = librosa.util.fix_length(audio, size=n_samples)
    elif len(audio) > n_samples:
        audio = audio[:n_samples]
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest)
    if not rows:
        raise ValueError("No generated samples found in the provided manifests.")

    target_map = parse_key_values(args.target_map)
    stems_root = Path(args.stems_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_row = rows[0][1]
    sample_rate = int(first_row["audio_sample_rate"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_helper = AudioLDM2VAEWrapper(
        model_id=str(first_row["model_id"]),
        sample_rate=sample_rate,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        freeze_vae=True,
        use_mode=False,
    )

    per_sample = []
    correct = 0
    margins = []
    confusion = {}

    for manifest_path, row in rows:
        generated_path = resolve_audio_path(row["generated_wav"], manifest_path, "generated")
        condition_name = str(row.get("condition_name", ""))
        target_condition, target_stems = infer_target_stems(condition_name, target_map, args.target_condition)
        missing_targets = [stem for stem in target_stems if stem not in args.stems]
        if missing_targets:
            raise ValueError(f"Target stems {missing_targets} are not in --stems {args.stems}")

        pred_wave = load_audio(generated_path, sample_rate)
        n_samples = int(pred_wave.shape[-1])
        chunk_sec = n_samples / float(sample_rate)
        chunk_idx = int(row["chunk_idx"])
        offset = chunk_idx * chunk_sec

        scores = {}
        for stem in args.stems:
            stem_path = stems_root / str(row["song_name"]) / f"{stem}.wav"
            if not stem_path.exists():
                raise FileNotFoundError(f"Missing stem file: {stem_path}")
            stem_wave = load_stem_chunk(stem_path, sample_rate, offset, chunk_sec, n_samples)
            scores[stem] = float(
                audio_helper.compute_audio_similarity(
                    pred_wave,
                    stem_wave,
                    sample_rate=sample_rate,
                )[0].item()
            )

        predicted_stem = max(scores, key=scores.get)
        target_score = max(scores[stem] for stem in target_stems)
        non_target_scores = [score for stem, score in scores.items() if stem not in target_stems]
        if not non_target_scores:
            raise ValueError("Need at least one non-target stem to compute target margin.")
        non_target_best = max(non_target_scores)
        margin = target_score - non_target_best
        is_correct = predicted_stem in target_stems

        correct += int(is_correct)
        margins.append(margin)
        target_label = "+".join(target_stems)
        confusion.setdefault(target_label, {stem: 0 for stem in args.stems})
        confusion[target_label][predicted_stem] += 1

        item = dict(row)
        item.update(
            {
                "generated_wav_resolved": str(generated_path),
                "target_condition": target_condition,
                "target_stems": target_stems,
                "target_stem": target_label,
                "predicted_stem": predicted_stem,
                "correct": is_correct,
                "target_stem_score": target_score,
                "best_non_target_score": non_target_best,
                "target_margin": margin,
                "stem_scores": scores,
            }
        )
        per_sample.append(item)

    n = len(per_sample)
    summary = {
        "num_samples": n,
        "stem_retrieval_accuracy": correct / n,
        "mean_target_margin": sum(margins) / n,
        "stems": args.stems,
        "target_map": target_map,
        "confusion_counts": confusion,
        "confusion_rows_normalized": {
            target: {
                pred: (count / sum(preds.values()) if sum(preds.values()) else 0.0)
                for pred, count in preds.items()
            }
            for target, preds in confusion.items()
        },
    }

    per_sample_path = output_dir / "per_sample_stem_scores.json"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "per_sample_stem_scores.csv"
    per_sample_path.write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "song_name",
        "subject_idx",
        "chunk_idx",
        "condition_name",
        "target_condition",
        "target_stem",
        "predicted_stem",
        "correct",
        "target_stem_score",
        "best_non_target_score",
        "target_margin",
    ] + [f"score_{stem}" for stem in args.stems]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_sample:
            row = {key: item.get(key) for key in fieldnames}
            for stem in args.stems:
                row[f"score_{stem}"] = item["stem_scores"][stem]
            writer.writerow(row)

    print(f"saved: {per_sample_path}", flush=True)
    print(f"saved: {summary_path}", flush=True)
    print(f"saved: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
