from __future__ import annotations

import time
import yaml
import torch
from torch.utils.data import DataLoader

from datasets.nmedt_dataset import NMEDTDataset
from models.eeg_controlnet import EEGControlNetModel
from utils.seed import set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/train.yaml")
    set_seed(cfg["seed"])

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    log_every = int(cfg["train"].get("log_every", 10))

    print(
        f"device={device} cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()}",
        flush=True,
    )

    data_cfg = cfg["data"]
    latent_cfg = cfg.get("latent_cache", {})
    use_precomputed_latents = bool(latent_cfg.get("enabled", False))
    precomputed_latents_path = latent_cfg.get("path")

    dataset = NMEDTDataset(
        mat_path=data_cfg.get("mat_path", "data/EEG/song21_Imputed.mat"),
        audio_path=data_cfg.get("audio_path", "data/songs/song21_16k.wav"),
        data_key=data_cfg.get("data_key", "data21"),
        chunk_sec=data_cfg["chunk_sec"],
        eeg_fs=data_cfg["eeg_fs"],
        audio_fs=data_cfg["audio_fs"],
        precomputed_latents_path=precomputed_latents_path if use_precomputed_latents else None,
    )

    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
    )
    print("dataset size:", len(dataset), flush=True)

    audio_cfg = cfg.get("audio_encoder", {})
    model_cfg = cfg["model"]
    unet_kwargs = model_cfg.get("audioldm_unet_kwargs")
    if unet_kwargs is None:
        unet_kwargs = cfg.get("audioldm_unet_kwargs", {})

    latent_channels = latent_cfg.get("latent_channels")
    if latent_channels is None and dataset.z0_by_chunk is not None:
        latent_channels = int(dataset.z0_by_chunk.shape[1])

    model = EEGControlNetModel(
        eeg_channels=data_cfg["eeg_channels"],
        num_subjects=dataset.total_subjects,
        use_subject_adapter=model_cfg["use_subject_adapter"],
        subject_emb_dim=model_cfg["subject_emb_dim"],
        device=device,
        audio_model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
        audio_sample_rate=audio_cfg.get("sample_rate", data_cfg["audio_fs"]),
        audio_freeze_vae=audio_cfg.get("freeze_vae", True),
        audio_use_mode=audio_cfg.get("use_mode", False),
        enable_audio_encoder=not use_precomputed_latents,
        latent_channels=latent_channels,
        eeg_global_dim=model_cfg["eeg_global_dim"],
        diffusion_num_steps=cfg["diffusion"]["num_train_timesteps"],
        diffusion_beta_start=cfg["diffusion"]["beta_start"],
        diffusion_beta_end=cfg["diffusion"]["beta_end"],
        unet_base_channels=model_cfg["unet_base_channels"],
        prefer_audioldm_unet=model_cfg.get("prefer_audioldm_unet", False),
        audioldm_unet_kwargs=unet_kwargs,
    ).to(device)

    print("model parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    print("unet backend:", model.control_unet.backend_name, flush=True)
    print("latent source:", "precomputed z0" if use_precomputed_latents else "online VAE", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
    )

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        for step, batch in enumerate(loader):
            step_t0 = time.perf_counter()

            eeg = batch["eeg"].to(device)
            subject_idx = batch["subject_idx"].to(device)
            timesteps = model.sample_timesteps(
                batch_size=eeg.shape[0],
                device=device,
            )

            batch_audio = None
            batch_z0 = None
            if use_precomputed_latents:
                if "z0" not in batch:
                    raise KeyError(
                        "latent_cache.enabled=true but dataset batch has no 'z0'."
                    )
                batch_z0 = batch["z0"].to(device)
            else:
                batch_audio = batch["audio"].to(device)

            out = model(
                eeg=eeg,
                subject_idx=subject_idx,
                audio=batch_audio,
                z0=batch_z0,
                timesteps=timesteps,
            )

            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["train"]["grad_clip"],
            )
            optimizer.step()

            if step % log_every == 0:
                step_dt = time.perf_counter() - step_t0
                print(
                    f"[epoch {epoch:02d} step {step:04d}] "
                    f"loss={loss.item():.6f} step_s={step_dt:.2f} "
                    f"z0={tuple(out['z0'].shape)} "
                    f"eeg_global={tuple(out['eeg_global'].shape)} "
                    f"t=[{out['timesteps'].min().item()},{out['timesteps'].max().item()}]",
                    flush=True,
                )

    torch.save(model.state_dict(), cfg["train"].get("checkpoint_path", "eeg_controlnet_smoke.pt"))
    print("saved ->", cfg["train"].get("checkpoint_path", "eeg_controlnet_smoke.pt"), flush=True)


if __name__ == "__main__":
    main()
