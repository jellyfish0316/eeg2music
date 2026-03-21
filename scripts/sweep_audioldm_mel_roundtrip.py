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


VARIANTS = {
    "baseline": {
        "mel_n_fft": 1024,
        "mel_win_length": 1024,
        "mel_hop_length": 160,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
    },
    "hop256": {
        "mel_n_fft": 1024,
        "mel_win_length": 1024,
        "mel_hop_length": 256,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
    },
    "win400_hop160": {
        "mel_n_fft": 1024,
        "mel_win_length": 400,
        "mel_hop_length": 160,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
    },
    "no_norm": {
        "mel_n_fft": 1024,
        "mel_win_length": 1024,
        "mel_hop_length": 160,
        "mel_norm": None,
        "mel_scale": "slaney",
    },
    "htk": {
        "mel_n_fft": 1024,
        "mel_win_length": 1024,
        "mel_hop_length": 160,
        "mel_norm": None,
        "mel_scale": "htk",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep AudioLDM2 mel preprocessing with round-trip tests.")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--audio-path", type=str, required=True)
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--length-sec", type=float, default=None)
    p.add_argument("--output-dir", type=str, default="outputs/audioldm_mel_sweep")
    p.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "hop256", "win400_hop160", "no_norm", "htk"],
        choices=sorted(VARIANTS.keys()),
    )
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def _official_decode_stats(wrapper: AudioLDM2MusicEncoderWrapper) -> dict[str, float | list[int]]:
    pipe = wrapper._load_full_pipeline()
    shape = (
        1,
        int(pipe.unet.config.in_channels),
        int(pipe.unet.config.sample_size),
        int(pipe.unet.config.sample_size) // 16,
    )
    latents = torch.randn(shape, device=wrapper.device, dtype=wrapper.dtype)
    latents = latents * float(pipe.scheduler.init_noise_sigma)
    mel = pipe.vae.decode(latents / float(pipe.vae.config.scaling_factor)).sample
    return {
        "shape": list(mel.shape),
        "min": float(mel.min().item()),
        "max": float(mel.max().item()),
        "mean": float(mel.mean().item()),
    }


def _load_audio_slice(audio_path: Path, sample_rate: int, start_sec: float, length_sec: float) -> torch.Tensor:
    audio, sr = sf.read(audio_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        raise ValueError(f"Expected sample_rate={sample_rate}, but {audio_path} has sr={sr}.")

    start = int(round(start_sec * sample_rate))
    stop = start + int(round(length_sec * sample_rate))
    if start < 0 or stop > len(audio):
        raise ValueError(f"Requested slice [{start}:{stop}] is outside audio length {len(audio)}.")

    return torch.tensor(audio[start:stop], dtype=torch.float32).unsqueeze(0)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    set_seed(seed)

    data_cfg = cfg["data"]
    audio_cfg = cfg.get("audio_encoder", {})
    sample_rate = int(audio_cfg.get("sample_rate", data_cfg["audio_fs"]))
    chunk_sec = float(data_cfg.get("chunk_sec", 3.5) if args.length_sec is None else args.length_sec)
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    source = _load_audio_slice(audio_path, sample_rate, float(args.start_sec), chunk_sec)
    source = source.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "source.wav", source[0].detach().cpu().numpy(), sample_rate)

    results: list[dict] = []

    for name in args.variants:
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        wrapper = AudioLDM2MusicEncoderWrapper(
            model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
            sample_rate=sample_rate,
            device=device,
            dtype=dtype,
            freeze_vae=True,
            use_mode=bool(audio_cfg.get("use_mode", False)),
            **VARIANTS[name],
        )

        source_mel = wrapper.waveform_to_mel(source)
        latents = wrapper(source)
        mel_decoded = wrapper.decode_latents_to_mel(latents)
        recon = wrapper.decode_latents_to_waveform(latents)[0].detach().cpu().float().numpy()
        sf.write(variant_dir / "roundtrip.wav", recon, wrapper.vocoder_sample_rate)

        official_stats = _official_decode_stats(wrapper)
        item = {
            "name": name,
            "mel_config": wrapper.mel_config,
            "source_mel_shape": list(source_mel.shape),
            "decoded_mel_shape": list(mel_decoded.shape),
            "source_stats": {
                "min": float(source_mel.min().item()),
                "max": float(source_mel.max().item()),
                "mean": float(source_mel.mean().item()),
            },
            "decoded_stats": {
                "min": float(mel_decoded.min().item()),
                "max": float(mel_decoded.max().item()),
                "mean": float(mel_decoded.mean().item()),
            },
            "official_decode_stats": official_stats,
            "roundtrip_wav": str(variant_dir / "roundtrip.wav"),
        }
        results.append(item)
        print(
            f"[{name}] source_mean={item['source_stats']['mean']:.4f} "
            f"decoded_mean={item['decoded_stats']['mean']:.4f} "
            f"official_mean={official_stats['mean']:.4f}",
            flush=True,
        )

    manifest = {
        "audio_path": str(audio_path.resolve()),
        "sample_rate": sample_rate,
        "start_sec": float(args.start_sec),
        "length_sec": chunk_sec,
        "variants": results,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
