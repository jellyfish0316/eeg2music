from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.condition_nmedt_dataset import ConditionNMEDTDataset
from models.eeg_conditioned_audioldm2 import EEGConditionedAudioLDM2
from models.audioldm2_vae_wrapper import AudioLDM2VAEWrapper
from utils.loso import create_loso_subject_splits
from utils.generation import batch_clap_similarity, generate_latents
from utils.seed import set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_model_config(model_cfg: dict) -> None:
    if "prefer_audioldm_unet" in model_cfg:
        raise ValueError(
            "model.prefer_audioldm_unet is deprecated. "
            "The pretrained U-Net is now loaded directly from AudioLDM2Pipeline."
        )
    if "audioldm_unet_kwargs" in model_cfg:
        raise ValueError(
            "model.audioldm_unet_kwargs is deprecated. "
            "The pretrained U-Net is now loaded directly from AudioLDM2Pipeline."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Condition-aware LOSO trainer")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--fold", type=int, default=None, help="Run a single fold index")
    p.add_argument("--all-folds", action="store_true", help="Run all LOSO folds")
    p.add_argument("--max-steps", type=int, default=None, help="Optional max steps per epoch")
    return p.parse_args()


def _normalize_subject_subset(subject_subset: object | None, *, total_subjects: int) -> list[int] | None:
    if subject_subset is None:
        return None
    if not isinstance(subject_subset, (list, tuple)):
        raise ValueError(f"split.subject_indices must be a list of zero-based subject indices, got {subject_subset!r}")
    normalized = sorted({int(s) for s in subject_subset})
    for s in normalized:
        if s < 0 or s >= total_subjects:
            raise ValueError(f"Invalid subject index {s}; total_subjects={total_subjects}")
    if len(normalized) == 0:
        raise ValueError("split.subject_indices cannot be empty.")
    return normalized


def _set_trainable(module: torch.nn.Module | None, enabled: bool) -> int:
    if module is None:
        return 0
    c = 0
    for p in module.parameters():
        p.requires_grad = enabled
        c += p.numel()
    return c


def apply_freeze_policy(model: EEGConditionedAudioLDM2, control_cfg: dict) -> dict[str, int]:
    if not bool(control_cfg.get("enabled", False)):
        return {"total_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if not bool(control_cfg.get("freeze_base_unet", True)):
        return {"total_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}

    for p in model.parameters():
        p.requires_grad = False

    names = list(control_cfg.get("trainable_modules", ["subject_adapter", "projector", "control_branch"]))
    stats: dict[str, int] = {}
    for n in names:
        stats[n] = _set_trainable(getattr(model, n, None), True)
    stats["total_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return stats


def derive_latent_grid(
    cfg: dict,
    *,
    dataset: ConditionNMEDTDataset,
    device: torch.device,
) -> tuple[int, int, int]:
    latent_cfg = cfg.get("latent_cache", {})
    audio_cfg = cfg.get("audio_encoder", {})
    data_cfg = cfg["data"]
    projector_cfg = cfg["model"].get("projector", {})

    configured = projector_cfg.get("lat_grid")
    if configured is not None:
        return tuple(int(v) for v in configured)
    dataset_latent_shape = getattr(dataset, "latent_shape", None)
    if dataset_latent_shape is not None:
        return tuple(int(v) for v in dataset_latent_shape)
    if dataset.z0_by_chunk is not None:
        return tuple(int(v) for v in dataset.z0_by_chunk.shape[1:])

    encoder = AudioLDM2VAEWrapper(
        model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
        sample_rate=int(audio_cfg.get("sample_rate", data_cfg["audio_fs"])),
        device=str(device),
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        freeze_vae=bool(audio_cfg.get("freeze_vae", True)),
        use_mode=bool(latent_cfg.get("precompute_use_mode", True)),
    )
    return encoder.infer_latent_shape(int(data_cfg["audio_samples"]))


def build_condition_jobs(exp_cfg: dict) -> list[dict[str, str]]:
    active = list(exp_cfg.get("active_instruments", []))
    conditions = list(exp_cfg.get("conditions", ["multi_attention", "single_repeated", "passive_x3"]))
    jobs: list[dict[str, str]] = []

    for c in conditions:
        if c == "single_repeated":
            for inst in active:
                jobs.append(
                    {
                        "condition_name": f"single_repeated_{inst}",
                        "condition_type": "single_repeated",
                        "target_instrument": inst,
                    }
                )
        else:
            jobs.append(
                {
                    "condition_name": c,
                    "condition_type": c,
                    "target_instrument": "",
                }
            )
    return jobs


def build_dataloader(
    cfg: dict,
    *,
    condition_type: str,
    target_instrument: str | None,
    subjects: list[int],
    shuffle: bool,
    chunk_split_name: str = "train",
) -> tuple[ConditionNMEDTDataset, DataLoader]:
    data_cfg = cfg["data"]
    exp_cfg = cfg.get("experiment", {})
    latent_cfg = cfg.get("latent_cache", {})
    split_cfg = cfg.get("split", {})
    use_precomputed_latents = bool(latent_cfg.get("enabled", False))
    chunk_splits = split_cfg.get("chunk_splits", {})
    chunk_range = tuple(chunk_splits.get(chunk_split_name, [0.0, 1.0]))
    songs = data_cfg.get("songs")
    song_splits = split_cfg.get("song_splits", {})
    selected_song_names = song_splits.get(chunk_split_name)
    if songs and selected_song_names is not None:
        selected_set = {str(name) for name in selected_song_names}
        songs = [song for song in songs if str(song.get("name")) in selected_set]
        if len(songs) == 0:
            raise ValueError(
                f"split.song_splits.{chunk_split_name} selected no songs. "
                f"requested={list(selected_song_names)}"
            )

    dataset = ConditionNMEDTDataset(
        condition_type=condition_type,
        active_instruments=list(exp_cfg.get("active_instruments", ["drum", "guitar", "vocal"])),
        target_instrument=target_instrument if target_instrument else None,
        mat_path=data_cfg.get("mat_path", "data/EEG/song21_Imputed.mat"),
        audio_path=data_cfg.get("audio_path", "data/songs/song21_16k.wav"),
        data_key=data_cfg.get("data_key", "data21"),
        condition_sources=data_cfg.get("condition_sources", None),
        songs=songs,
        chunk_sec=float(data_cfg["chunk_sec"]),
        eeg_fs=int(data_cfg["eeg_fs"]),
        audio_fs=int(data_cfg["audio_fs"]),
        subjects=subjects,
        text_prompt=str(data_cfg.get("text_prompt", "Pop music")),
        precomputed_latents_path=latent_cfg.get("path") if use_precomputed_latents else None,
        chunk_range=chunk_range,
        eeg_chunk_cache_dir=data_cfg.get("eeg_chunk_cache_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    return dataset, loader


def build_model_from_dataset(
    cfg: dict,
    *,
    dataset: ConditionNMEDTDataset,
    device: torch.device,
) -> EEGConditionedAudioLDM2:
    def resolve_model_dtype(value: object | None) -> torch.dtype:
        if value is None:
            return torch.float16 if device.type == "cuda" else torch.float32
        text = str(value).strip().lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "full": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if text not in mapping:
            raise ValueError(
                f"Unsupported train.model_dtype={value!r}. "
                "Use one of: float16, float32, bfloat16."
            )
        return mapping[text]

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    audio_cfg = cfg.get("audio_encoder", {})
    latent_cfg = cfg.get("latent_cache", {})
    control_cfg = cfg.get("controlnet", {})
    train_cfg = cfg.get("train", {})

    use_precomputed_latents = bool(latent_cfg.get("enabled", False))
    latent_channels = latent_cfg.get("latent_channels")
    dataset_latent_shape = getattr(dataset, "latent_shape", None)
    if latent_channels is None and dataset_latent_shape is not None:
        latent_channels = int(dataset_latent_shape[0])
    elif latent_channels is None and dataset.z0_by_chunk is not None:
        latent_channels = int(dataset.z0_by_chunk.shape[1])
    latent_grid = derive_latent_grid(cfg, dataset=dataset, device=device)

    model = EEGConditionedAudioLDM2(
        eeg_channels=int(dataset.eeg_out_channels),
        num_subjects=int(dataset.total_subjects),
        model_dtype=resolve_model_dtype(train_cfg.get("model_dtype")),
        use_subject_adapter=bool(model_cfg.get("use_subject_adapter", True)),
        subject_emb_dim=int(model_cfg.get("subject_emb_dim", 64)),
        device=device,
        audio_model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
        audio_sample_rate=audio_cfg.get("sample_rate", data_cfg["audio_fs"]),
        audio_freeze_vae=bool(audio_cfg.get("freeze_vae", True)),
        audio_use_mode=bool(audio_cfg.get("use_mode", False)),
        text_prompt=str(data_cfg.get("text_prompt", "Pop music")),
        text_cache_path=model_cfg.get("unet", {}).get("text_cache_path"),
        enable_audio_encoder=not use_precomputed_latents,
        latent_channels=latent_channels,
        latent_grid=latent_grid,
        projector_channels=tuple(model_cfg.get("projector", {}).get("channels", [256, 512, 1024, 2048])),
        projector_strides=tuple(model_cfg.get("projector", {}).get("strides", [5, 2, 2, 2])),
        diffusion_num_steps=int(cfg["diffusion"]["num_train_timesteps"]),
        diffusion_beta_start=float(cfg["diffusion"]["beta_start"]),
        diffusion_beta_end=float(cfg["diffusion"]["beta_end"]),
        unet_cache_pipeline=bool(model_cfg.get("unet", {}).get("cache_pipeline", True)),
        controlnet_enabled=bool(control_cfg.get("enabled", False)),
        controlnet_zero_init=bool(control_cfg.get("zero_init", True)),
        controlnet_scale=float(control_cfg.get("control_scale", 1.0)),
        controlnet_copy_encoder_weights=bool(control_cfg.get("copy_encoder_weights", True)),
        controlnet_inject_middle_block=bool(control_cfg.get("inject_middle_block", True)),
        controlnet_conditioning_mode=str(control_cfg.get("conditioning_mode", "repo")),
    ).to(device)
    with torch.no_grad():
        dummy_eeg = torch.zeros(
            1,
            int(dataset.eeg_out_channels),
            int(cfg["data"]["eeg_time"]),
            device=device,
            dtype=torch.float32,
        )
        model.projector(dummy_eeg)
    print(f"training_model_dtype={str(model.model_dtype).replace('torch.', '')}", flush=True)
    return model


@torch.no_grad()
def evaluate_loss(
    model: EEGConditionedAudioLDM2,
    loader: DataLoader,
    device: torch.device,
    control_cfg: dict,
    max_steps: int | None = None,
) -> float:
    model.eval()
    losses = []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        eeg = batch["eeg"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        timesteps = model.sample_timesteps(batch_size=eeg.shape[0], device=device)
        batch_audio = None
        batch_z0 = None
        if "z0" in batch:
            batch_z0 = batch["z0"].to(device)
        else:
            batch_audio = batch["audio"].to(device)
        out = model(
            eeg=eeg,
            subject_idx=subject_idx,
            audio=batch_audio,
            z0=batch_z0,
            timesteps=timesteps,
            use_control=bool(control_cfg.get("enabled", False)),
            control_scale=float(control_cfg.get("control_scale", 1.0)),
        )
        if not torch.isfinite(out["loss"]):
            raise RuntimeError(
                "Non-finite validation loss detected: "
                f"loss={out['loss'].item()} "
                f"timesteps={timesteps.detach().cpu().tolist()} "
                f"z0_finite={bool(torch.isfinite(out['z0']).all().item())} "
                f"zt_finite={bool(torch.isfinite(out['zt']).all().item())} "
                f"noise_finite={bool(torch.isfinite(out['noise']).all().item())} "
                f"projected_finite={bool(torch.isfinite(out['projected_latent']).all().item())} "
                f"eps_pred_finite={bool(torch.isfinite(out['eps_pred']).all().item())}"
            )
        losses.append(float(out["loss"].item()))
    model.train()
    if len(losses) == 0:
        return float("nan")
    return float(sum(losses) / len(losses))


@torch.no_grad()
def evaluate_generation_clap(
    model: EEGConditionedAudioLDM2,
    loader: DataLoader,
    device: torch.device,
    control_cfg: dict,
    audio_helper: AudioLDM2VAEWrapper,
    *,
    sample_rate: int,
    num_inference_steps: int,
    max_batches: int | None = None,
) -> float:
    model.eval()
    scores = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        eeg = batch["eeg"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        target_audio = batch["audio"]
        pred_latents = generate_latents(
            model,
            eeg=eeg,
            subject_idx=subject_idx,
            num_inference_steps=num_inference_steps,
            use_control=bool(control_cfg.get("enabled", False)),
            control_scale=float(control_cfg.get("control_scale", 1.0)),
        )
        predicted_audio = audio_helper.decode_latents_to_waveform(pred_latents)
        sims = batch_clap_similarity(
            audio_helper,
            predicted_audio,
            target_audio,
            sample_rate=sample_rate,
        )
        scores.extend(float(v) for v in sims.tolist())
    model.train()
    if len(scores) == 0:
        return float("nan")
    return float(sum(scores) / len(scores))


def run_one_condition(
    cfg: dict,
    *,
    fold_meta: dict,
    condition_job: dict[str, str],
    device: torch.device,
    output_dir: Path,
    max_steps: int | None = None,
) -> dict[str, object]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    audio_cfg = cfg.get("audio_encoder", {})
    latent_cfg = cfg.get("latent_cache", {})
    control_cfg = cfg.get("controlnet", {})
    train_cfg = cfg["train"]

    condition_name = condition_job["condition_name"]
    condition_type = condition_job["condition_type"]
    target_instrument = condition_job["target_instrument"] or None

    ds_train, dl_train = build_dataloader(
        cfg,
        condition_type=condition_type,
        target_instrument=target_instrument,
        subjects=fold_meta["train_subjects"],
        shuffle=True,
        chunk_split_name="train",
    )
    ds_val, dl_val = build_dataloader(
        cfg,
        condition_type=condition_type,
        target_instrument=target_instrument,
        subjects=fold_meta["val_subjects"],
        shuffle=False,
        chunk_split_name="val",
    )
    ds_test, dl_test = build_dataloader(
        cfg,
        condition_type=condition_type,
        target_instrument=target_instrument,
        subjects=fold_meta["test_subjects"],
        shuffle=False,
        chunk_split_name="test",
    )

    model = build_model_from_dataset(cfg, dataset=ds_train, device=device)

    freeze_stats = apply_freeze_policy(model, control_cfg)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg["lr"]),
    )

    print(
        f"[fold {fold_meta['fold_index']}][{condition_name}] "
        f"train={len(ds_train)} val={len(ds_val)} test={len(ds_test)} "
        f"model_params={sum(p.numel() for p in model.parameters())} "
        f"trainable={freeze_stats['total_trainable']}",
        flush=True,
    )

    epochs = int(train_cfg["epochs"])
    log_every = int(train_cfg.get("log_every", 10))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    gradient_accumulation_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("train.gradient_accumulation_steps must be >= 1")
    validation_metric = str(train_cfg.get("validation_metric", "loss")).lower()
    validation_generate_batches = train_cfg.get("validation_generate_batches", None)
    if validation_generate_batches is not None:
        validation_generate_batches = int(validation_generate_batches)
    validation_num_inference_steps = int(train_cfg.get("validation_num_inference_steps", 20))
    history = {"train_loss": [], "val_loss": [], "val_clap": []}
    best_metric_name = "val_clap" if validation_metric == "clap" else "val_loss"
    best_metric_value = float("-inf") if validation_metric == "clap" else float("inf")
    best_ckpt_name = str(train_cfg.get("best_checkpoint_name", "best_model.pt"))
    best_ckpt_path = output_dir / best_ckpt_name
    clap_helper = None
    if validation_metric == "clap":
        clap_helper = AudioLDM2VAEWrapper(
            model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
            sample_rate=int(audio_cfg.get("sample_rate", data_cfg["audio_fs"])),
            device=str(device),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            freeze_vae=True,
            use_mode=bool(audio_cfg.get("use_mode", False)),
        )

    for epoch in range(epochs):
        model.train()
        running = []
        optimizer.zero_grad()
        for step, batch in enumerate(dl_train):
            if max_steps is not None and step >= max_steps:
                break
            t0 = time.perf_counter()
            eeg = batch["eeg"].to(device)
            subject_idx = batch["subject_idx"].to(device)
            timesteps = model.sample_timesteps(batch_size=eeg.shape[0], device=device)

            batch_audio = None
            batch_z0 = None
            if "z0" in batch:
                batch_z0 = batch["z0"].to(device)
            else:
                batch_audio = batch["audio"].to(device)

            out = model(
                eeg=eeg,
                subject_idx=subject_idx,
                audio=batch_audio,
                z0=batch_z0,
                timesteps=timesteps,
                use_control=bool(control_cfg.get("enabled", False)),
                control_scale=float(control_cfg.get("control_scale", 1.0)),
            )
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected: "
                    f"condition={condition_name} epoch={epoch} step={step} "
                    f"loss={loss.item()} "
                    f"timesteps={timesteps.detach().cpu().tolist()} "
                    f"z0_finite={bool(torch.isfinite(out['z0']).all().item())} "
                    f"zt_finite={bool(torch.isfinite(out['zt']).all().item())} "
                    f"noise_finite={bool(torch.isfinite(out['noise']).all().item())} "
                    f"projected_finite={bool(torch.isfinite(out['projected_latent']).all().item())} "
                    f"eps_pred_finite={bool(torch.isfinite(out['eps_pred']).all().item())}"
                )
            (loss / float(gradient_accumulation_steps)).backward()
            should_step = ((step + 1) % gradient_accumulation_steps == 0)
            is_last_step = (step + 1 == len(dl_train)) or (max_steps is not None and (step + 1) == max_steps)
            if should_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            running.append(float(loss.item()))

            if step % log_every == 0:
                print(
                    f"[fold {fold_meta['fold_index']}][{condition_name}] "
                    f"[ep {epoch:02d} st {step:04d}] "
                    f"loss={loss.item():.6f} step_s={time.perf_counter()-t0:.2f} "
                    f"use_control={bool(out['use_control'].item())}",
                    flush=True,
                )

        train_loss = float(sum(running) / max(1, len(running)))
        val_loss = evaluate_loss(model, dl_val, device, control_cfg, max_steps=max_steps)
        val_clap = float("nan")
        if validation_metric == "clap":
            if clap_helper is None:
                raise RuntimeError("validation_metric='clap' requires a CLAP/audio helper.")
            val_clap = evaluate_generation_clap(
                model,
                dl_val,
                device,
                control_cfg,
                clap_helper,
                sample_rate=int(data_cfg["audio_fs"]),
                num_inference_steps=validation_num_inference_steps,
                max_batches=validation_generate_batches,
            )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_clap"].append(val_clap)
        current_metric = val_clap if validation_metric == "clap" else val_loss
        is_better = current_metric > best_metric_value if validation_metric == "clap" else current_metric < best_metric_value
        if torch.isfinite(torch.tensor(current_metric)) and is_better:
            best_metric_value = float(current_metric)
            torch.save(model.state_dict(), best_ckpt_path)
        print(
            f"[fold {fold_meta['fold_index']}][{condition_name}] epoch={epoch} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_clap={val_clap:.6f}",
            flush=True,
        )

    test_loss = evaluate_loss(model, dl_test, device, control_cfg, max_steps=max_steps)
    ckpt_name = cfg["train"].get("checkpoint_name", "model.pt")
    ckpt_path = output_dir / ckpt_name
    torch.save(model.state_dict(), ckpt_path)

    return {
        "fold_index": int(fold_meta["fold_index"]),
        "condition_name": condition_name,
        "condition_type": condition_type,
        "target_instrument": target_instrument,
        "train_subjects": fold_meta["train_subjects"],
        "val_subjects": fold_meta["val_subjects"],
        "test_subjects": fold_meta["test_subjects"],
        "history": history,
        "test_loss": test_loss,
        "best_metric_name": best_metric_name,
        "best_metric_value": float(best_metric_value),
        "trainable_params": int(freeze_stats["total_trainable"]),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "unet_backend": model.control_unet.backend_name,
        "checkpoint_path": str(ckpt_path),
        "best_checkpoint_path": str(best_ckpt_path),
    }


def build_pairwise_report(results: list[dict[str, object]]) -> list[dict[str, object]]:
    report = []
    by_fold: dict[int, list[dict[str, object]]] = {}
    for r in results:
        by_fold.setdefault(int(r["fold_index"]), []).append(r)

    for fold_idx, fold_rows in by_fold.items():
        passive = [x for x in fold_rows if x["condition_name"] == "passive_x3"]
        if len(passive) == 0:
            continue
        base = passive[0]
        base_loss = float(base["test_loss"])
        for row in fold_rows:
            if row["condition_name"] == "passive_x3":
                continue
            report.append(
                {
                    "fold_index": fold_idx,
                    "compare": f"{row['condition_name']} vs passive_x3",
                    "test_loss_condition": float(row["test_loss"]),
                    "test_loss_passive": base_loss,
                    "delta_condition_minus_passive": float(row["test_loss"]) - base_loss,
                }
            )
    return report


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    validate_model_config(cfg.get("model", {}))

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(
        f"device={device} cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}",
        flush=True,
    )

    exp_cfg = cfg.get("experiment", {})
    split_cfg = cfg.get("split", {})
    data_cfg = cfg["data"]

    # Build a temporary dataset only to read total_subjects for LOSO.
    ds_probe, _ = build_dataloader(
        cfg,
        condition_type="passive_x3",
        target_instrument=None,
        subjects=None,
        shuffle=False,
    )
    total_subjects = int(ds_probe.total_subjects)
    print("total_subjects:", total_subjects, flush=True)
    subject_subset = _normalize_subject_subset(
        split_cfg.get("subject_indices"),
        total_subjects=total_subjects,
    )
    effective_subjects = list(range(total_subjects)) if subject_subset is None else subject_subset
    print("effective_subjects:", effective_subjects, flush=True)

    if bool(split_cfg.get("loso", {}).get("enabled", True)):
        num_folds = exp_cfg.get("num_folds", None)
        if num_folds is None:
            num_folds = total_subjects
        splits = create_loso_subject_splits(
            total_subjects=total_subjects,
            val_ratio=float(split_cfg.get("val_ratio", 0.1)),
            seed=int(exp_cfg.get("seed", cfg.get("seed", 42))),
            num_folds=int(num_folds),
        )
        if subject_subset is not None:
            filtered_splits = []
            for split in splits:
                train_subjects = [s for s in split["train_subjects"] if s in subject_subset]
                val_subjects = [s for s in split["val_subjects"] if s in subject_subset]
                test_subjects = [s for s in split["test_subjects"] if s in subject_subset]
                if len(train_subjects) == 0 or len(val_subjects) == 0 or len(test_subjects) == 0:
                    continue
                filtered_splits.append(
                    {
                        **split,
                        "train_subjects": train_subjects,
                        "val_subjects": val_subjects,
                        "test_subjects": test_subjects,
                    }
                )
            splits = filtered_splits
    else:
        all_subjects = effective_subjects
        splits = [
            {
                "fold_index": 0,
                "test_subject": -1,
                "train_subjects": all_subjects,
                "val_subjects": all_subjects[: max(1, int(0.1 * len(all_subjects)))],
                "test_subjects": all_subjects,
            }
        ]

    run_mode = str(cfg["train"].get("run_mode", "single_fold"))
    if args.fold is not None:
        splits = [s for s in splits if int(s["fold_index"]) == int(args.fold)]
        if len(splits) == 0:
            raise ValueError(f"No split found for fold={args.fold}")
    elif args.all_folds or run_mode == "all_folds":
        pass
    else:
        fold_index = exp_cfg.get("fold_index", 0)
        splits = [s for s in splits if int(s["fold_index"]) == int(fold_index)]
        if len(splits) == 0:
            raise ValueError(f"No split found for fold_index={fold_index}")

    condition_jobs = build_condition_jobs(exp_cfg)
    print("condition_jobs:", [x["condition_name"] for x in condition_jobs], flush=True)

    output_root = Path(cfg["train"].get("output_root", "outputs/loso_runs"))
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    for split in splits:
        fold_dir = output_root / f"fold_{int(split['fold_index']):02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for job in condition_jobs:
            cond_dir = fold_dir / job["condition_name"]
            cond_dir.mkdir(parents=True, exist_ok=True)

            result = run_one_condition(
                cfg,
                fold_meta=split,
                condition_job=job,
                device=device,
                output_dir=cond_dir,
                max_steps=args.max_steps,
            )
            results.append(result)
            # save latest model stats only (state_dict persistence can be added later if needed)
            with open(cond_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            # save path marker for compatibility
            with open(cond_dir / "checkpoint_path.txt", "w", encoding="utf-8") as f:
                f.write(str(result["checkpoint_path"]) + "\n")

    pairwise = build_pairwise_report(results)
    with open(output_root / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(output_root / "pairwise_report.json", "w", encoding="utf-8") as f:
        json.dump(pairwise, f, ensure_ascii=False, indent=2)

    print("saved:", output_root / "all_results.json", flush=True)
    print("saved:", output_root / "pairwise_report.json", flush=True)


if __name__ == "__main__":
    main()
