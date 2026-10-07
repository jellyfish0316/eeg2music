from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from datasets.condition_nmedt_dataset import ConditionNMEDTDataset
from models.audioldm2_vae_wrapper import AudioLDM2VAEWrapper
from scripts.train import build_dataloader, build_model_from_dataset, load_config
from utils.generation import generate_latents, save_waveforms
from utils.evaluation_pairing import EvaluationPairingDataset, stable_generation_seed
from utils.seed import set_seed


@dataclass
class ConditionedEvaluationResources:
    dataset: object
    model: object
    decoder: AudioLDM2VAEWrapper
    device: torch.device


def prepare_conditioned_evaluation_resources(
    *,
    cfg: dict,
    checkpoint: str | Path,
    split: str,
    subjects: list[int],
) -> ConditionedEvaluationResources:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    dataset, _ = build_dataloader(
        cfg,
        subjects=[int(value) for value in subjects],
        shuffle=False,
        chunk_split_name=split,
    )
    model = build_model_from_dataset(
        cfg,
        dataset=dataset,
        device=device,
        enable_audio_encoder_override=False,
    )
    state = torch.load(str(checkpoint), map_location=device)
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
    return ConditionedEvaluationResources(dataset=dataset, model=model, decoder=decoder, device=device)


def build_full_song_pool_dataset(cfg: dict, *, subjects: list[int]) -> ConditionNMEDTDataset:
    """Build one dataset spanning every configured song at its full chunk range.

    The `correct`/`shuffled` evaluation dataset is filtered down to a single
    ood_test song (see split.ood_song_splits), so it has no other song to draw
    cross-song EEG from. Cross-song evaluation needs one flat dataset whose
    index_map covers every song for the same subjects, so that a single
    EvaluationPairingDataset mapping can address both the ood_test target rows
    and the other songs' source rows by dataset index. Using each song's full
    [0.0, 1.0] chunk range (rather than a train/val/test fraction) does not
    leak target information: the target audio is always the fixed ood_test
    song, so a mismatched-song EEG chunk cannot let the model recall its
    originally paired audio.
    """

    data_cfg = cfg["data"]
    latent_cfg = cfg.get("latent_cache", {})
    use_precomputed_latents = bool(latent_cfg.get("enabled", False))
    return ConditionNMEDTDataset(
        mat_path=data_cfg.get("mat_path"),
        audio_path=data_cfg.get("audio_path"),
        data_key=data_cfg.get("data_key", "data21"),
        songs=data_cfg.get("songs"),
        chunk_sec=float(data_cfg["chunk_sec"]),
        eeg_fs=int(data_cfg["eeg_fs"]),
        audio_fs=int(data_cfg["audio_fs"]),
        subjects=[int(value) for value in subjects],
        text_prompt=str(data_cfg.get("text_prompt", "Pop music")),
        precomputed_latents_path=latent_cfg.get("path") if use_precomputed_latents else None,
        chunk_range=(0.0, 1.0),
        eeg_chunk_cache_dir=data_cfg.get("eeg_chunk_cache_dir"),
        condition_sources=data_cfg.get("condition_sources"),
        expected_eeg_channels=data_cfg.get("expected_eeg_channels"),
    )


def prepare_cross_song_evaluation_resources(
    *,
    cfg: dict,
    checkpoint: str | Path,
    subjects: list[int],
) -> ConditionedEvaluationResources:
    """Like prepare_conditioned_evaluation_resources, but over the full song pool."""

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    dataset = build_full_song_pool_dataset(cfg, subjects=[int(value) for value in subjects])
    model = build_model_from_dataset(
        cfg,
        dataset=dataset,
        device=device,
        enable_audio_encoder_override=False,
    )
    state = torch.load(str(checkpoint), map_location=device)
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
    return ConditionedEvaluationResources(dataset=dataset, model=model, decoder=decoder, device=device)


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


def generate_conditioned_evaluation(
    *,
    cfg: dict,
    checkpoint: str | Path,
    split: str,
    subjects: list[int],
    output_dir: str | Path,
    evaluation_mode: str,
    mapping: dict[int, int],
    job_metadata: dict[str, object],
    num_inference_steps: int,
    guidance_scale: float,
    eta: float,
    generation_seed: int,
    permutation_id: int | None = None,
    permutation_seed: int | None = None,
    max_samples: int | None = None,
    resources: ConditionedEvaluationResources | None = None,
    shuffle_min_chunk_distance: int = 1,
    temporal_offset_chunks: int | None = None,
) -> Path:
    """Generate one paper-evaluation manifest while preserving target rows.

    The legacy CLI below intentionally does not call this function. Paper
    controls use explicit per-target generators and a dataset-level EEG mapping;
    the existing command retains its historical RNG stream and output schema.

    `mapping` may be a partial dataset-index -> dataset-index map (cross_song,
    temporal_shift): targets absent from `mapping` are skipped rather than
    generated, because no valid source EEG chunk exists for them.
    """

    if evaluation_mode not in {"correct", "shuffled", "cross_song", "temporal_shift"}:
        raise ValueError(
            "Conditioned evaluation mode must be correct/shuffled/cross_song/temporal_shift, "
            f"got {evaluation_mode!r}."
        )
    if evaluation_mode == "temporal_shift" and temporal_offset_chunks is None:
        raise ValueError("temporal_shift evaluation mode requires temporal_offset_chunks.")
    if resources is None:
        resources = prepare_conditioned_evaluation_resources(
            cfg=cfg,
            checkpoint=checkpoint,
            split=split,
            subjects=subjects,
        )
    dataset = resources.dataset
    model = resources.model
    decoder = resources.decoder
    device = resources.device
    condition_sources = [str(value) for value in (cfg["data"].get("condition_sources") or ["passive"])]
    paired_dataset = EvaluationPairingDataset(
        dataset,
        mapping,
        mode=evaluation_mode,
        condition_sources=condition_sources,
        base_eeg_channels=int(dataset.base_eeg_channels),
        min_chunk_distance=int(shuffle_min_chunk_distance),
        temporal_offset=temporal_offset_chunks,
    )

    audio_cfg = cfg.get("audio_encoder", {})
    data_cfg = cfg["data"]

    output_dir = Path(output_dir)
    generated_dir = output_dir / "generated"
    target_dir = output_dir / "target"
    rows: list[dict[str, object]] = []
    condition_name = "+".join(condition_sources)
    for target_index in range(len(paired_dataset)):
        if max_samples is not None and len(rows) >= int(max_samples):
            break
        sample = paired_dataset[target_index]
        song_name = str(sample.get("song_name", "song"))
        chunk_index = int(sample["chunk_idx"].item())
        subject_index = int(sample["subject_idx"].item())
        sample_seed = stable_generation_seed(
            base_seed=int(generation_seed),
            song_id=song_name,
            chunk_index=chunk_index,
        )
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        eeg = sample["eeg"].unsqueeze(0).to(device)
        subject_idx = sample["subject_idx"].reshape(1).to(device)
        pred_latents = generate_latents(
            model,
            eeg=eeg,
            subject_idx=subject_idx,
            num_inference_steps=int(num_inference_steps),
            eta=float(eta),
            generator=generator,
            use_control=bool(cfg.get("controlnet", {}).get("enabled", False)),
            control_scale=float(cfg.get("controlnet", {}).get("control_scale", 1.0)),
            guidance_scale=float(guidance_scale),
        )
        predicted_audio = decoder.decode_latents_to_waveform(pred_latents)
        target_audio = sample["audio"].unsqueeze(0)
        offset_suffix = "" if temporal_offset_chunks is None else f"_offset{temporal_offset_chunks:+03d}"
        filename = (
            f"{condition_name}_{evaluation_mode}_{split}_{song_name}_"
            f"subj{subject_index:02d}_chunk{chunk_index:04d}{offset_suffix}_seed{sample_seed}.wav"
        )
        generated_path = save_waveforms(
            predicted_audio,
            output_dir=generated_dir,
            filenames=[filename],
            sample_rate=decoder.vocoder_sample_rate,
        )[0]
        target_path = save_waveforms(
            target_audio,
            output_dir=target_dir,
            filenames=[filename],
            sample_rate=int(data_cfg["audio_fs"]),
        )[0]
        source_song_idx = int(sample["eeg_source_song_idx"].item())
        source_subject_idx = int(sample["eeg_source_subject_idx"].item())
        source_chunk_idx = int(sample["eeg_source_chunk_idx"].item())
        rows.append(
            {
                "evaluation_mode": evaluation_mode,
                "job_id": str(job_metadata["id"]),
                "model_type": str(job_metadata["model_type"]),
                "regime": str(job_metadata["regime"]),
                "subject": job_metadata.get("subject"),
                "held_out_subject": job_metadata.get("held_out_subject"),
                "condition_name": condition_name,
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_source": job_metadata.get("checkpoint_source"),
                "split": split,
                "permutation_id": permutation_id,
                "permutation_seed": permutation_seed,
                "target_dataset_idx": int(sample["target_dataset_idx"].item()),
                "target_chunk_id": chunk_index,
                "target_song_id": song_name,
                "target_subject_id": subject_index,
                "chunk_idx": chunk_index,
                "song_name": song_name,
                "subject_idx": subject_index,
                "eeg_source_dataset_idx": int(sample["eeg_source_dataset_idx"].item()),
                "eeg_source_chunk_id": source_chunk_idx,
                "eeg_source_song_id": str(dataset.song_records[source_song_idx].name),
                "eeg_source_subject_id": source_subject_idx,
                "temporal_offset_chunks": temporal_offset_chunks,
                "temporal_offset_seconds": None
                if temporal_offset_chunks is None
                else float(temporal_offset_chunks) * float(data_cfg["chunk_sec"]),
                "generation_seed": sample_seed,
                "generated_wav": generated_path,
                "target_wav": target_path,
                "model_id": audio_cfg.get("model_id", "cvssp/audioldm2-music"),
                "prompt": str(data_cfg.get("text_prompt", "Pop music")),
                "audio_sample_rate": int(data_cfg["audio_fs"]),
                "generated_sample_rate": int(decoder.vocoder_sample_rate),
                "chunk_sec": float(data_cfg["chunk_sec"]),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "eta": float(eta),
                "use_control": True,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "meta": {
                    **job_metadata,
                    "evaluation_mode": evaluation_mode,
                    "split": split,
                    "num_inference_steps": int(num_inference_steps),
                    "guidance_scale": float(guidance_scale),
                    "eta": float(eta),
                    "generation_seed": int(generation_seed),
                    "permutation_id": permutation_id,
                    "permutation_seed": permutation_seed,
                    "temporal_offset_chunks": temporal_offset_chunks,
                    "num_rows": len(rows),
                    "num_valid_targets": len(paired_dataset),
                },
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def generate_legacy_correct_regression(
    *,
    cfg: dict,
    checkpoint: str | Path,
    split: str,
    output_dir: str | Path,
    job_metadata: dict[str, object],
    num_inference_steps: int,
    batch_size: int,
    resources: ConditionedEvaluationResources,
    max_samples: int | None = None,
) -> Path:
    """Reproduce the historical global-RNG, batched correct generation path.

    This is deliberately separate from paper-control generation: it validates
    rebuilt EEG against archived results, while correct/shuffled controls use
    explicit paired per-target generators.
    """

    from torch.utils.data import DataLoader

    dataset = resources.dataset
    model = resources.model
    decoder = resources.decoder
    device = resources.device
    data_cfg = cfg["data"]
    audio_cfg = cfg.get("audio_encoder", {})
    condition_sources = [str(value) for value in (data_cfg.get("condition_sources") or ["passive"])]
    condition_name = "+".join(condition_sources)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)

    output_dir = Path(output_dir)
    generated_dir = output_dir / "generated"
    target_dir = output_dir / "target"
    rows: list[dict[str, object]] = []
    for batch_index, batch in enumerate(loader):
        if max_samples is not None and len(rows) >= int(max_samples):
            break
        eeg = batch["eeg"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        pred_latents = generate_latents(
            model,
            eeg=eeg,
            subject_idx=subject_idx,
            num_inference_steps=int(num_inference_steps),
            use_control=bool(cfg.get("controlnet", {}).get("enabled", False)),
            control_scale=float(cfg.get("controlnet", {}).get("control_scale", 1.0)),
        )
        predicted_audio = decoder.decode_latents_to_waveform(pred_latents)
        target_audio = batch["audio"]
        names: list[str] = []
        for item_index in range(predicted_audio.shape[0]):
            subject = int(batch["subject_idx"][item_index].item())
            chunk = int(batch["chunk_idx"][item_index].item())
            song_idx = int(batch["song_idx"][item_index].item()) if "song_idx" in batch else 0
            song_name = batch["song_name"][item_index] if "song_name" in batch else "song"
            names.append(
                f"{condition_name}_{split}_{song_name}_song{song_idx:02d}_"
                f"subj{subject:02d}_chunk{chunk:04d}.wav"
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
        for item_index, _name in enumerate(names):
            if max_samples is not None and len(rows) >= int(max_samples):
                break
            subject = int(batch["subject_idx"][item_index].item())
            chunk = int(batch["chunk_idx"][item_index].item())
            song_idx = int(batch["song_idx"][item_index].item()) if "song_idx" in batch else 0
            song_name = batch["song_name"][item_index] if "song_name" in batch else "song"
            rows.append(
                {
                    "evaluation_mode": "legacy_correct_regression",
                    "job_id": str(job_metadata["id"]),
                    "model_type": str(job_metadata["model_type"]),
                    "regime": str(job_metadata["regime"]),
                    "subject": job_metadata.get("subject"),
                    "held_out_subject": job_metadata.get("held_out_subject"),
                    "condition_name": condition_name,
                    "checkpoint": str(Path(checkpoint).resolve()),
                    "checkpoint_source": job_metadata.get("checkpoint_source"),
                    "split": split,
                    "target_chunk_id": chunk,
                    "target_song_id": str(song_name),
                    "target_subject_id": subject,
                    "chunk_idx": chunk,
                    "song_idx": song_idx,
                    "song_name": str(song_name),
                    "subject_idx": subject,
                    "legacy_batch_index": batch_index,
                    "rng_protocol": "legacy_global_batch_stream",
                    "generated_wav": generated_paths[item_index],
                    "target_wav": target_paths[item_index],
                    "model_id": audio_cfg.get("model_id", "cvssp/audioldm2-music"),
                    "audio_sample_rate": int(data_cfg["audio_fs"]),
                    "generated_sample_rate": int(decoder.vocoder_sample_rate),
                    "num_inference_steps": int(num_inference_steps),
                    "use_control": bool(cfg.get("controlnet", {}).get("enabled", False)),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "meta": {
                    **job_metadata,
                    "evaluation_mode": "legacy_correct_regression",
                    "split": split,
                    "num_inference_steps": int(num_inference_steps),
                    "batch_size": int(batch_size),
                    "rng_protocol": "legacy_global_batch_stream",
                    "num_rows": len(rows),
                },
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    condition_sources = cfg["data"].get("condition_sources") or ["passive"]
    condition_name = "+".join(str(source) for source in condition_sources)

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
