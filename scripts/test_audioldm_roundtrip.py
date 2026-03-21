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

from models.audioldm2_wrapper import AudioLDM2MusicEncoderWrapper
from scripts.train import load_config
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round-trip test for AudioLDM2 VAE encode/decode.")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--audio-path", type=str, default=None)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--length-sec", type=float, default=None)
    p.add_argument("--output-dir", type=str, default="outputs/audioldm_roundtrip")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    set_seed(seed)

    data_cfg = cfg["data"]
    audio_cfg = cfg.get("audio_encoder", {})

    audio_path = Path(args.audio_path or data_cfg["audio_path"])
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    sample_rate = int(audio_cfg.get("sample_rate", data_cfg["audio_fs"]))
    chunk_sec = float(data_cfg.get("chunk_sec", 3.5) if args.length_sec is None else args.length_sec)
    start_sec = float(args.start_sec)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    wrapper = AudioLDM2MusicEncoderWrapper(
        model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
        sample_rate=sample_rate,
        device=str(device),
        dtype=dtype,
        freeze_vae=True,
        use_mode=bool(audio_cfg.get("use_mode", False)),
    )

    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        raise ValueError(
            f"Round-trip test expects sample_rate={sample_rate}, but {audio_path} has sr={sr}."
        )

    start = int(round(start_sec * sample_rate))
    length = int(round(chunk_sec * sample_rate))
    stop = start + length
    if start < 0 or stop > len(audio):
        raise ValueError(
            f"Requested slice [{start}:{stop}] is outside audio length {len(audio)}."
        )

    source = torch.tensor(audio[start:stop], dtype=torch.float32, device=device).unsqueeze(0)
    latents = wrapper(source)
    reconstructed = wrapper.decode_latents_to_waveform(latents)[0].detach().cpu().float().numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.wav"
    recon_path = output_dir / "roundtrip.wav"
    sf.write(source_path, source[start:stop].detach().cpu().numpy()[0], sample_rate)
    sf.write(recon_path, reconstructed, wrapper.vocoder_sample_rate)

    manifest = {
        "meta": {
            "config": str(Path(args.config).resolve()),
            "audio_path": str(audio_path.resolve()),
            "start_sec": start_sec,
            "length_sec": chunk_sec,
            "model_id": audio_cfg.get("model_id", "cvssp/audioldm2-music"),
            "sample_rate": sample_rate,
            "reconstructed_sample_rate": int(wrapper.vocoder_sample_rate),
            "seed": seed,
        },
        "files": {
            "source_wav": str(source_path),
            "roundtrip_wav": str(recon_path),
        },
        "latent_shape": list(latents.shape),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"saved source: {source_path}", flush=True)
    print(f"saved roundtrip: {recon_path}", flush=True)
    print(f"saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
