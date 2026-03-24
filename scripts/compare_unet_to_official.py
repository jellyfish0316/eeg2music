from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from models.audioldm2_unet_wrapper import AudioLDM2UNetWrapper
from scripts.train import load_config
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare wrapper U-Net output to official AudioLDM2 U-Net output.")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--guidance", action="store_true", help="Use CFG text conditioning instead of plain cached text conditioning.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=str, default="outputs/unet_compare.json")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    set_seed(seed)

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    audio_cfg = cfg.get("audio_encoder", {})
    model_cfg = cfg["model"]

    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    wrapper = AudioLDM2UNetWrapper(
        model_id=audio_cfg.get("model_id", "cvssp/audioldm2-music"),
        device=device,
        dtype=dtype,
        cache_pipeline=True,
        text_prompt=str(data_cfg.get("text_prompt", "Pop music")),
        text_cache_path=model_cfg.get("unet", {}).get("text_cache_path"),
    )
    pipe = wrapper.pipeline
    if pipe is None:
        raise RuntimeError("Wrapper pipeline was not cached.")

    batch_size = int(args.batch_size)
    latent_shape = (
        batch_size,
        int(pipe.unet.config.in_channels),
        int(pipe.unet.config.sample_size) // int(pipe.vae_scale_factor),
        int(pipe.vocoder.config.model_in_dim) // int(pipe.vae_scale_factor),
    )
    sample = torch.randn(latent_shape, device=device, dtype=dtype)
    timesteps = torch.full((batch_size,), int(args.timestep), device=device, dtype=torch.long)

    if args.guidance:
        text_cond = wrapper.get_text_conditioning_with_guidance(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            guidance_scale=3.5,
            negative_prompt="",
        )
    else:
        text_cond = wrapper.get_text_conditioning(batch_size=batch_size, device=device, dtype=dtype)

    official = pipe.unet(
        sample,
        timesteps,
        encoder_hidden_states=text_cond["encoder_hidden_states"],
        encoder_hidden_states_1=text_cond["encoder_hidden_states_1"],
        encoder_attention_mask_1=text_cond["attention_mask"],
        return_dict=False,
    )[0]
    ours = wrapper(
        x=sample,
        timesteps=timesteps,
        encoder_hidden_states=text_cond["encoder_hidden_states"],
        encoder_hidden_states_1=text_cond["encoder_hidden_states_1"],
    )

    abs_diff = (official - ours).abs()
    rel = abs_diff / official.abs().clamp_min(1e-6)

    payload = {
        "batch_size": batch_size,
        "timestep": int(args.timestep),
        "guidance": bool(args.guidance),
        "shape": list(official.shape),
        "official_mean": float(official.mean().item()),
        "ours_mean": float(ours.mean().item()),
        "max_abs_diff": float(abs_diff.max().item()),
        "mean_abs_diff": float(abs_diff.mean().item()),
        "mean_rel_diff": float(rel.mean().item()),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
