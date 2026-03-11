from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from models.audioldm2_wrapper import AudioLDM2MusicEncoderWrapper
from scripts.train import build_condition_jobs, build_dataloader, build_model_from_dataset, load_config
from utils.generation import generate_latents, save_waveforms
from utils.loso import create_loso_subject_splits
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate decoded music from EEG using a trained checkpoint")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--condition", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="outputs/generated_audio")
    return p.parse_args()


def resolve_fold_split(cfg: dict, fold_index: int) -> dict:
    exp_cfg = cfg.get("experiment", {})
    split_cfg = cfg.get("split", {})
    ds_probe, _ = build_dataloader(
        cfg,
        condition_type="passive_x3",
        target_instrument=None,
        subjects=None,
        shuffle=False,
    )
    total_subjects = int(ds_probe.total_subjects)
    num_folds = exp_cfg.get("num_folds", total_subjects)
    splits = create_loso_subject_splits(
        total_subjects=total_subjects,
        val_ratio=float(split_cfg.get("val_ratio", 0.1)),
        seed=int(exp_cfg.get("seed", cfg.get("seed", 42))),
        num_folds=int(num_folds),
    )
    matches = [s for s in splits if int(s["fold_index"]) == int(fold_index)]
    if len(matches) == 0:
        raise ValueError(f"No split found for fold={fold_index}")
    return matches[0]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    fold_meta = resolve_fold_split(cfg, args.fold)
    jobs = build_condition_jobs(cfg.get("experiment", {}))
    condition_matches = [j for j in jobs if j["condition_name"] == args.condition]
    if len(condition_matches) == 0:
        raise ValueError(f"Unknown condition {args.condition!r}; available={[j['condition_name'] for j in jobs]}")
    job = condition_matches[0]

    subject_key = {
        "train": "train_subjects",
        "val": "val_subjects",
        "test": "test_subjects",
    }[args.split]
    dataset, loader = build_dataloader(
        cfg,
        condition_type=job["condition_type"],
        target_instrument=job["target_instrument"] or None,
        subjects=fold_meta[subject_key],
        shuffle=False,
    )

    model = build_model_from_dataset(cfg, dataset=dataset, device=device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    audio_cfg = cfg.get("audio_encoder", {})
    data_cfg = cfg["data"]
    decoder = AudioLDM2MusicEncoderWrapper(
        model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
        sample_rate=int(audio_cfg.get("sample_rate", data_cfg["audio_fs"])),
        device=str(device),
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        freeze_vae=True,
        use_mode=bool(audio_cfg.get("use_mode", False)),
    )

    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated"
    target_dir = output_dir / "target"
    manifest_rows = []

    for step, batch in enumerate(loader):
        if args.max_batches is not None and step >= args.max_batches:
            break

        eeg = batch["eeg"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        pred_latents = generate_latents(
            model,
            eeg=eeg,
            subject_idx=subject_idx,
            num_inference_steps=int(args.num_inference_steps),
            use_control=bool(cfg.get("controlnet", {}).get("enabled", False)),
            control_scale=float(cfg.get("controlnet", {}).get("control_scale", 1.0)),
        )
        predicted_audio = decoder.decode_latents_to_waveform(pred_latents)
        target_audio = batch["audio"]

        names = []
        for i in range(predicted_audio.shape[0]):
            subj = int(batch["subject_idx"][i].item())
            chunk = int(batch["chunk_idx"][i].item())
            names.append(f"fold{args.fold:02d}_{args.condition}_{args.split}_subj{subj:02d}_chunk{chunk:04d}.wav")

        generated_paths = save_waveforms(
            predicted_audio,
            output_dir=generated_dir,
            filenames=names,
            sample_rate=decoder.vocoder_sample_rate,
        )
        target_paths = save_waveforms(
            target_audio,
            output_dir=target_dir,
            filenames=names,
            sample_rate=int(data_cfg["audio_fs"]),
        )

        for i, name in enumerate(names):
            manifest_rows.append(
                {
                    "fold_index": int(args.fold),
                    "condition_name": args.condition,
                    "split": args.split,
                    "subject_idx": int(batch["subject_idx"][i].item()),
                    "chunk_idx": int(batch["chunk_idx"][i].item()),
                    "generated_wav": generated_paths[i],
                    "target_wav": target_paths[i],
                    "checkpoint_path": str(Path(args.checkpoint).resolve()),
                    "model_id": audio_cfg.get("model_id", "cvssp/audioldm2-music"),
                    "audio_sample_rate": int(data_cfg["audio_fs"]),
                    "generated_sample_rate": int(decoder.vocoder_sample_rate),
                    "num_inference_steps": int(args.num_inference_steps),
                }
            )

    payload = {
        "meta": {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "fold_index": int(args.fold),
            "condition_name": args.condition,
            "split": args.split,
            "num_inference_steps": int(args.num_inference_steps),
            "num_rows": len(manifest_rows),
        },
        "samples": manifest_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
