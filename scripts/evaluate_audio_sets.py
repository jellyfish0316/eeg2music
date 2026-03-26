from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.linalg
import soundfile as sf
import torch

from models.audioldm2_vae_wrapper import AudioLDM2VAEWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate generated audio sets with CLAP text similarity and Frechet-style CLAP audio distance"
    )
    p.add_argument(
        "--candidate",
        type=str,
        action="append",
        required=True,
        help="Candidate set in label=path form. Path can be a wav dir, manifest.json, or manifest root dir.",
    )
    p.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Reference audio set. Can be a wav dir, manifest.json, or manifest root dir.",
    )
    p.add_argument("--prompt", type=str, default="Pop music")
    p.add_argument("--output", type=str, default="outputs/audio_set_eval/summary.json")
    return p.parse_args()


def load_manifest_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("samples", []))


def resolve_audio_paths(path_str: str, *, kind: str) -> list[Path]:
    path = Path(path_str)
    if path.is_file() and path.suffix.lower() == ".json":
        rows = load_manifest_rows(path)
        key = "generated_wav" if kind == "generated" else "target_wav"
        return [Path(row[key]) for row in rows]
    if path.is_dir():
        manifests = sorted(path.rglob("manifest.json"))
        if manifests:
            rows: list[dict] = []
            for manifest in manifests:
                rows.extend(load_manifest_rows(manifest))
            key = "generated_wav" if kind == "generated" else "target_wav"
            return [Path(row[key]) for row in rows]
        wavs = sorted([p for p in path.rglob("*.wav") if p.is_file()])
        if wavs:
            return wavs
    if path.is_file() and path.suffix.lower() == ".wav":
        return [path]
    raise FileNotFoundError(f"Could not resolve audio files from {path_str}")


def load_wave(path: Path) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0), int(sr)


def stack_audio_features(
    helper: AudioLDM2VAEWrapper,
    paths: list[Path],
    *,
    normalize: bool,
) -> torch.Tensor:
    features = []
    for path in paths:
        wave, sr = load_wave(path)
        feat = helper.get_audio_features(wave, sample_rate=sr, normalize=normalize)
        features.append(feat[0])
    return torch.stack(features, dim=0)


def compute_frechet_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)
    y_np = y.detach().cpu().numpy().astype(np.float64, copy=False)

    mu_x = np.mean(x_np, axis=0)
    mu_y = np.mean(y_np, axis=0)
    sigma_x = np.cov(x_np, rowvar=False)
    sigma_y = np.cov(y_np, rowvar=False)

    diff = mu_x - mu_y
    eps = 1e-6
    sigma_x = sigma_x + np.eye(sigma_x.shape[0], dtype=np.float64) * eps
    sigma_y = sigma_y + np.eye(sigma_y.shape[0], dtype=np.float64) * eps
    covmean, _ = scipy.linalg.sqrtm(sigma_x @ sigma_y, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0.0, atol=1e-6):
            raise RuntimeError("Frechet covariance sqrt produced non-negligible imaginary components.")
        covmean = np.real(covmean)
    fid = float(diff @ diff + np.trace(sigma_x + sigma_y - 2.0 * covmean))
    return max(fid, 0.0)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    helper = AudioLDM2VAEWrapper(
        model_id="cvssp/audioldm2-music",
        sample_rate=16000,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        freeze_vae=True,
        use_mode=False,
    )

    reference_paths = resolve_audio_paths(args.reference, kind="target")
    if len(reference_paths) == 0:
        raise ValueError("Reference set resolved to zero audio files.")
    reference_features = stack_audio_features(helper, reference_paths, normalize=False)

    results = {
        "prompt": args.prompt,
        "reference": {
            "path": args.reference,
            "num_files": len(reference_paths),
        },
        "candidates": {},
    }

    for item in args.candidate:
        if "=" not in item:
            raise ValueError(f"--candidate must be label=path, got {item!r}")
        label, path_str = item.split("=", 1)
        candidate_paths = resolve_audio_paths(path_str, kind="generated")
        if len(candidate_paths) == 0:
            raise ValueError(f"Candidate {label!r} resolved to zero audio files.")

        text_scores = []
        for path in candidate_paths:
            wave, sr = load_wave(path)
            score = float(helper.compute_text_audio_similarity(args.prompt, wave, sample_rate=sr)[0].item())
            text_scores.append(score)

        candidate_features = stack_audio_features(helper, candidate_paths, normalize=False)
        frechet = compute_frechet_distance(candidate_features, reference_features)

        results["candidates"][label] = {
            "path": path_str,
            "num_files": len(candidate_paths),
            "mean_clap_text_audio": float(sum(text_scores) / len(text_scores)),
            "frechet_clap_audio_to_reference": float(frechet),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
