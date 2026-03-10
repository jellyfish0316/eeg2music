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
from models.eeg_controlnet import EEGControlNetModel
from models.audioldm2_wrapper import AudioLDM2MusicEncoderWrapper
from utils.loso import create_loso_subject_splits
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


def _set_trainable(module: torch.nn.Module | None, enabled: bool) -> int:
    if module is None:
        return 0
    c = 0
    for p in module.parameters():
        p.requires_grad = enabled
        c += p.numel()
    return c


def apply_freeze_policy(model: EEGControlNetModel, control_cfg: dict) -> dict[str, int]:
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
    if dataset.z0_by_chunk is not None:
        return tuple(int(v) for v in dataset.z0_by_chunk.shape[1:])

    encoder = AudioLDM2MusicEncoderWrapper(
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
) -> tuple[ConditionNMEDTDataset, DataLoader]:
    data_cfg = cfg["data"]
    exp_cfg = cfg.get("experiment", {})
    latent_cfg = cfg.get("latent_cache", {})
    use_precomputed_latents = bool(latent_cfg.get("enabled", False))

    dataset = ConditionNMEDTDataset(
        condition_type=condition_type,
        active_instruments=list(exp_cfg.get("active_instruments", ["drum", "guitar", "vocal"])),
        target_instrument=target_instrument if target_instrument else None,
        mat_path=data_cfg.get("mat_path", "data/EEG/song21_Imputed.mat"),
        audio_path=data_cfg.get("audio_path", "data/songs/song21_16k.wav"),
        data_key=data_cfg.get("data_key", "data21"),
        condition_sources=data_cfg.get("condition_sources", None),
        chunk_sec=float(data_cfg["chunk_sec"]),
        eeg_fs=int(data_cfg["eeg_fs"]),
        audio_fs=int(data_cfg["audio_fs"]),
        subjects=subjects,
        normalize_eeg=bool(data_cfg.get("eeg_preprocessing", {}).get("per_channel_normalization", True)),
        text_prompt=str(data_cfg.get("text_prompt", "Pop music")),
        eeg_preprocessing=data_cfg.get("eeg_preprocessing", None),
        precomputed_latents_path=latent_cfg.get("path") if use_precomputed_latents else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    return dataset, loader


@torch.no_grad()
def evaluate_loss(
    model: EEGControlNetModel,
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
    )
    ds_val, dl_val = build_dataloader(
        cfg,
        condition_type=condition_type,
        target_instrument=target_instrument,
        subjects=fold_meta["val_subjects"],
        shuffle=False,
    )
    ds_test, dl_test = build_dataloader(
        cfg,
        condition_type=condition_type,
        target_instrument=target_instrument,
        subjects=fold_meta["test_subjects"],
        shuffle=False,
    )

    use_precomputed_latents = bool(latent_cfg.get("enabled", False))
    latent_channels = latent_cfg.get("latent_channels")
    if latent_channels is None and ds_train.z0_by_chunk is not None:
        latent_channels = int(ds_train.z0_by_chunk.shape[1])
    latent_grid = derive_latent_grid(cfg, dataset=ds_train, device=device)

    model = EEGControlNetModel(
        eeg_channels=int(ds_train.eeg_out_channels),
        num_subjects=int(ds_train.total_subjects),
        use_subject_adapter=bool(model_cfg.get("use_subject_adapter", True)),
        subject_emb_dim=int(model_cfg.get("subject_emb_dim", 64)),
        device=device,
        audio_model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
        audio_sample_rate=audio_cfg.get("sample_rate", data_cfg["audio_fs"]),
        audio_freeze_vae=bool(audio_cfg.get("freeze_vae", True)),
        audio_use_mode=bool(audio_cfg.get("use_mode", False)),
        enable_audio_encoder=not use_precomputed_latents,
        latent_channels=latent_channels,
        latent_grid=latent_grid,
        projector_channels=tuple(model_cfg.get("projector", {}).get("channels", [256, 512, 1024, 2048])),
        projector_strides=tuple(model_cfg.get("projector", {}).get("strides", [5, 2, 2, 2])),
        projector_use_linear_fallback=bool(model_cfg.get("projector", {}).get("use_linear_fallback", True)),
        diffusion_num_steps=int(cfg["diffusion"]["num_train_timesteps"]),
        diffusion_beta_start=float(cfg["diffusion"]["beta_start"]),
        diffusion_beta_end=float(cfg["diffusion"]["beta_end"]),
        unet_cache_pipeline=bool(model_cfg.get("unet", {}).get("cache_pipeline", True)),
        controlnet_enabled=bool(control_cfg.get("enabled", False)),
        controlnet_zero_init=bool(control_cfg.get("zero_init", True)),
        controlnet_scale=float(control_cfg.get("control_scale", 1.0)),
        controlnet_copy_encoder_weights=bool(control_cfg.get("copy_encoder_weights", True)),
        controlnet_inject_middle_block=bool(control_cfg.get("inject_middle_block", True)),
    ).to(device)

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
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running = []
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
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
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"[fold {fold_meta['fold_index']}][{condition_name}] epoch={epoch} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
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
        "trainable_params": int(freeze_stats["total_trainable"]),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "unet_backend": model.control_unet.backend_name,
        "checkpoint_path": str(ckpt_path),
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
    else:
        all_subjects = list(range(total_subjects))
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
