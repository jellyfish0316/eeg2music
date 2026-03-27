from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch

from models.audioldm2_vae_wrapper import AudioLDM2VAEWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate generated audio against targets with CLAP audio similarity")
    p.add_argument("--manifest", type=str, nargs="+", required=True)
    p.add_argument("--output-dir", type=str, default="outputs/evaluation")
    return p.parse_args()


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_dir():
            manifests = sorted(path.rglob("manifest.json"))
            if len(manifests) == 0:
                raise FileNotFoundError(f"No manifest.json found under {path}")
        else:
            manifests = [path]
        for manifest in manifests:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            rows.extend(payload.get("samples", []))
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest)
    if len(rows) == 0:
        raise ValueError("No generated samples found in the provided manifests.")

    first = rows[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_helper = AudioLDM2VAEWrapper(
        model_id=str(first["model_id"]),
        sample_rate=int(first["audio_sample_rate"]),
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        freeze_vae=True,
        use_mode=False,
    )

    per_sample = []
    scores = []
    for row in rows:
        pred_audio, pred_sr = sf.read(row["generated_wav"])
        target_audio, target_sr = sf.read(row["target_wav"])
        pred_wave = torch.tensor(pred_audio, dtype=torch.float32).unsqueeze(0)
        target_wave = torch.tensor(target_audio, dtype=torch.float32).unsqueeze(0)
        score = float(
            audio_helper.compute_audio_similarity(
                pred_wave,
                target_wave,
                sample_rate=int(target_sr),
            )[0].item()
        )
        item = dict(row)
        item["clap_audio_cosine"] = score
        item["pred_sample_rate"] = int(pred_sr)
        item["target_sample_rate"] = int(target_sr)
        per_sample.append(item)
        scores.append(score)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "per_sample_scores.json"
    summary_path = output_dir / "summary.json"
    per_sample_path.write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "num_samples": len(per_sample),
                "mean_clap_audio_cosine": float(sum(scores) / len(scores)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {per_sample_path}", flush=True)
    print(f"saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
