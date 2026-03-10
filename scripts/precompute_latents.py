from __future__ import annotations

import os
from pathlib import Path

import yaml
import torch

from datasets.nmedt_dataset import NMEDTDataset
from models.audioldm2_wrapper import AudioLDM2MusicEncoderWrapper
from utils.seed import set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def main():
    cfg = load_config("configs/train.yaml")
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    audio_cfg = cfg.get("audio_encoder", {})
    latent_cfg = cfg.get("latent_cache", {})

    output_path = latent_cfg.get("path", "data/precomputed/song21_audioldm2_latents.pt")
    batch_size = int(latent_cfg.get("precompute_batch_size", 16))
    use_mode = bool(latent_cfg.get("precompute_use_mode", True))

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    dataset = NMEDTDataset(
        mat_path=data_cfg.get("mat_path", "data/EEG/song21_Imputed.mat"),
        audio_path=data_cfg.get("audio_path", "data/songs/song21_16k.wav"),
        data_key=data_cfg.get("data_key", "data21"),
        chunk_sec=data_cfg["chunk_sec"],
        eeg_fs=data_cfg["eeg_fs"],
        audio_fs=data_cfg["audio_fs"],
        subjects=[0],  # only need chunk-aligned audio once
    )

    encoder = AudioLDM2MusicEncoderWrapper(
        model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
        sample_rate=audio_cfg.get("sample_rate", data_cfg["audio_fs"]),
        device=str(device),
        dtype=dtype,
        freeze_vae=True,
        use_mode=use_mode,
    )
    encoder.eval()

    all_z0 = []
    total = dataset.n_chunks
    print(f"precompute start: chunks={total} batch_size={batch_size} device={device}", flush=True)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        wav_batch = []
        for chunk_idx in range(start, end):
            audio_start = chunk_idx * dataset.audio_chunk_len
            audio_end = audio_start + dataset.audio_chunk_len
            wav = dataset.audio[audio_start:audio_end]
            wav_batch.append(torch.tensor(wav, dtype=torch.float32))
        waveform = torch.stack(wav_batch, dim=0).to(device)

        z0 = encoder(
            waveform,
            sample_posterior=not use_mode,
        )
        all_z0.append(z0.detach().cpu())
        print(f"encoded chunks [{start:04d}, {end:04d}) -> z0 {tuple(z0.shape)}", flush=True)

    z0_by_chunk = torch.cat(all_z0, dim=0).contiguous()
    payload = {
        "z0_by_chunk": z0_by_chunk,
        "meta": {
            "model_id": audio_cfg.get("model_id", "cvssp/audioldm2-music"),
            "sample_rate": data_cfg["audio_fs"],
            "chunk_sec": data_cfg["chunk_sec"],
            "n_chunks": int(z0_by_chunk.shape[0]),
            "latent_shape": list(z0_by_chunk.shape[1:]),
            "use_mode": use_mode,
        },
    }

    out = Path(output_path)
    os.makedirs(out.parent, exist_ok=True)
    torch.save(payload, out)
    print(f"saved latent cache -> {out} shape={tuple(z0_by_chunk.shape)}", flush=True)


if __name__ == "__main__":
    main()
