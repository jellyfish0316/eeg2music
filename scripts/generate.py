from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from models.audioldm2_vae_wrapper import AudioLDM2VAEWrapper
from scripts.train import build_dataloader, build_model_from_dataset, load_config
from utils.generation import generate_latents, save_waveforms
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate decoded music from EEG using a trained checkpoint")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test", "ood_test"], default="test")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="outputs/generated_audio")
    p.add_argument(
        "--eeg-mode",
        type=str,
        choices=["real", "zero", "random"],
        default="real",
        help="Use real EEG, all-zero EEG, or randomly permuted EEG within the batch.",
    )
    p.add_argument(
        "--disable-control",
        action="store_true",
        help="Disable ControlNet conditioning during generation, even if enabled in config.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    condition_name = "passive"

    ds_probe, _ = build_dataloader(
        cfg,
        subjects=None,
        shuffle=False,
    )
    total_subjects = int(ds_probe.total_subjects)
    subject_indices = cfg.get("split", {}).get("subject_indices")
    if subject_indices is None:
        selected_subjects = list(range(total_subjects))
    else:
        selected_subjects = sorted({int(s) for s in subject_indices})

    dataset, loader = build_dataloader(
        cfg,
        subjects=selected_subjects,
        shuffle=False,
        chunk_split_name=args.split,
    )

    model = build_model_from_dataset(cfg, dataset=dataset, device=device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    audio_cfg = cfg.get("audio_encoder", {})
    data_cfg = cfg["data"]
    decoder = AudioLDM2VAEWrapper(
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
        if args.eeg_mode == "zero":
            eeg = torch.zeros_like(eeg)
        elif args.eeg_mode == "random":
            if eeg.shape[0] > 1:
                perm = torch.randperm(eeg.shape[0], device=eeg.device)
                eeg = eeg[perm]
            else:
                eeg = torch.randn_like(eeg)
        subject_idx = batch["subject_idx"].to(device)
        use_control = bool(cfg.get("controlnet", {}).get("enabled", False)) and not bool(args.disable_control)
        pred_latents = generate_latents(
            model,
            eeg=eeg,
            subject_idx=subject_idx,
            num_inference_steps=int(args.num_inference_steps),
            use_control=use_control,
            control_scale=float(cfg.get("controlnet", {}).get("control_scale", 1.0)),
        )
        predicted_audio = decoder.decode_latents_to_waveform(pred_latents)
        target_audio = batch["audio"]

        names = []
        for i in range(predicted_audio.shape[0]):
            subj = int(batch["subject_idx"][i].item())
            chunk = int(batch["chunk_idx"][i].item())
            song_idx = int(batch["song_idx"][i].item()) if "song_idx" in batch else 0
            song_name = batch["song_name"][i] if "song_name" in batch else "song"
            names.append(
                f"{condition_name}_{args.split}_{song_name}_song{song_idx:02d}_subj{subj:02d}_chunk{chunk:04d}.wav"
            )

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
                    "condition_name": condition_name,
                    "split": args.split,
                    "song_idx": int(batch["song_idx"][i].item()) if "song_idx" in batch else 0,
                    "song_name": batch["song_name"][i] if "song_name" in batch else "song",
                    "subject_idx": int(batch["subject_idx"][i].item()),
                    "chunk_idx": int(batch["chunk_idx"][i].item()),
                    "generated_wav": generated_paths[i],
                    "target_wav": target_paths[i],
                    "checkpoint_path": str(Path(args.checkpoint).resolve()),
                    "model_id": audio_cfg.get("model_id", "cvssp/audioldm2-music"),
                    "audio_sample_rate": int(data_cfg["audio_fs"]),
                    "generated_sample_rate": int(decoder.vocoder_sample_rate),
                    "num_inference_steps": int(args.num_inference_steps),
                    "eeg_mode": args.eeg_mode,
                    "use_control": bool(use_control),
                }
            )

    payload = {
        "meta": {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "condition_name": condition_name,
            "split": args.split,
            "num_inference_steps": int(args.num_inference_steps),
            "eeg_mode": args.eeg_mode,
            "use_control": bool(not args.disable_control and cfg.get("controlnet", {}).get("enabled", False)),
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
