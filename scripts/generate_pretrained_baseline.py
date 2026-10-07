from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch
from diffusers import AudioLDM2Pipeline

from utils.evaluation_pairing import stable_generation_seed


DEFAULT_SONGS = ["song3", "song6", "song7", "song9", "song10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a text-only pretrained AudioLDM2 baseline paired with target chunks."
    )
    parser.add_argument("--target-dir", default="data/SelfRecorded_songs/wav_16k")
    parser.add_argument("--songs", nargs="+", default=DEFAULT_SONGS)
    parser.add_argument("--prompt", default="Pop music")
    parser.add_argument("--chunk-sec", type=float, default=3.5)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="Generate one baseline sample for every complete target chunk.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--model-id", default="cvssp/audioldm2-music")
    parser.add_argument("--output-dir", default="outputs/pretrained_baseline")
    return parser.parse_args()


def load_target_chunk(path: Path, *, chunk_index: int, chunk_sec: float) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    chunk_samples = round(float(chunk_sec) * int(sample_rate))
    start = int(chunk_index) * chunk_samples
    end = start + chunk_samples
    if end > len(audio):
        raise ValueError(
            f"Target chunk exceeds {path}: chunk_index={chunk_index}, "
            f"required_end={end}, samples={len(audio)}"
        )
    return torch.from_numpy(audio[start:end].copy()), int(sample_rate)


def count_complete_chunks(path: Path, *, chunk_sec: float) -> int:
    info = sf.info(path)
    chunk_samples = round(float(chunk_sec) * int(info.samplerate))
    return int(math.floor(int(info.frames) / chunk_samples))


def generate_pretrained_from_targets(
    *,
    target_rows: list[dict],
    output_dir: str | Path,
    model_id: str,
    prompt: str,
    chunk_sec: float,
    generation_seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    max_samples: int | None = None,
) -> Path:
    """Generate the unconditional pretrained prior from a canonical target list."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    output_dir = Path(output_dir)
    generated_dir = output_dir / "generated"
    target_dir = output_dir / "target"
    generated_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_targets: set[tuple[str, int]] = set()
    for target_row in target_rows:
        song_name = str(target_row["target_song_id"])
        chunk_index = int(target_row["target_chunk_id"])
        target_key = (song_name, chunk_index)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        if max_samples is not None and len(rows) >= int(max_samples):
            break
        source_path = Path(str(target_row["target_audio_path"]))
        target, target_sr = load_target_chunk(
            source_path,
            chunk_index=chunk_index,
            chunk_sec=chunk_sec,
        )
        sample_seed = stable_generation_seed(
            base_seed=int(generation_seed),
            song_id=song_name,
            chunk_index=chunk_index,
        )
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        result = pipe(
            prompt,
            audio_length_in_s=float(chunk_sec),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )
        generated = result.audios[0]
        filename = f"pretrained_{song_name}_chunk{chunk_index:04d}_seed{sample_seed}.wav"
        generated_path = generated_dir / filename
        target_path = target_dir / filename
        generated_sr = int(pipe.vocoder.config.sampling_rate)
        sf.write(generated_path, generated, generated_sr)
        sf.write(target_path, target.numpy(), target_sr)
        rows.append(
            {
                "evaluation_mode": "pretrained",
                "job_id": "pretrained_prior",
                "model_type": "AudioLDM2",
                "regime": "pretrained",
                "subject": None,
                "held_out_subject": None,
                "condition_name": "Unconditional pretrained prior baseline",
                "checkpoint": None,
                "permutation_id": None,
                "permutation_seed": None,
                "target_chunk_id": chunk_index,
                "target_song_id": song_name,
                "target_subject_id": None,
                "chunk_idx": chunk_index,
                "song_name": song_name,
                "subject_idx": -1,
                "eeg_source_dataset_idx": None,
                "eeg_source_chunk_id": None,
                "eeg_source_song_id": None,
                "eeg_source_subject_id": None,
                "generation_seed": sample_seed,
                "generated_wav": str(generated_path),
                "target_wav": str(target_path),
                "model_id": model_id,
                "prompt": prompt,
                "audio_sample_rate": int(target_sr),
                "generated_sample_rate": generated_sr,
                "chunk_sec": float(chunk_sec),
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "use_control": False,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "meta": {
                    "evaluation_mode": "pretrained",
                    "baseline_name": "Unconditional pretrained prior baseline",
                    "model_id": model_id,
                    "prompt": prompt,
                    "device": str(device),
                    "chunk_sec": float(chunk_sec),
                    "generation_seed": int(generation_seed),
                    "num_inference_steps": int(num_inference_steps),
                    "guidance_scale": float(guidance_scale),
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
    if args.chunk_index < 0:
        raise ValueError("--chunk-index must be non-negative.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AudioLDM2Pipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated"
    target_dir = output_dir / "target"
    generated_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for song_name in args.songs:
        source_path = Path(args.target_dir) / f"{song_name}.wav"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if args.all_chunks:
            chunk_indices = range(count_complete_chunks(source_path, chunk_sec=args.chunk_sec))
        else:
            chunk_indices = [int(args.chunk_index)]

        for chunk_index in chunk_indices:
            target, target_sr = load_target_chunk(
                source_path,
                chunk_index=chunk_index,
                chunk_sec=args.chunk_sec,
            )
            target_name = f"{song_name}_chunk{chunk_index:04d}.wav"
            target_path = target_dir / target_name
            sf.write(target_path, target.numpy(), target_sr)

            for base_seed in args.seeds:
                effective_seed = int(base_seed)
                if args.all_chunks:
                    effective_seed += int(chunk_index) * 1000
                generator = torch.Generator(device=device).manual_seed(effective_seed)
                result = pipe(
                    args.prompt,
                    audio_length_in_s=float(args.chunk_sec),
                    num_inference_steps=int(args.num_inference_steps),
                    guidance_scale=float(args.guidance_scale),
                    generator=generator,
                )
                generated = result.audios[0]
                generated_name = (
                    f"{song_name}_chunk{chunk_index:04d}_seed{effective_seed}.wav"
                )
                generated_path = generated_dir / generated_name
                generated_sr = int(pipe.vocoder.config.sampling_rate)
                sf.write(generated_path, generated, generated_sr)
                rows.append(
                    {
                        "condition_name": "pretrained_text_only",
                        "split": "ood_test" if args.all_chunks else "baseline",
                        "song_name": song_name,
                        "song_idx": int(song_name.removeprefix("song")),
                        "subject_idx": -1,
                        "chunk_idx": int(chunk_index),
                        "seed": effective_seed,
                        "base_seed": int(base_seed),
                        "generated_wav": str(generated_path),
                        "target_wav": str(target_path),
                        "checkpoint_path": None,
                        "model_id": args.model_id,
                        "prompt": args.prompt,
                        "audio_sample_rate": target_sr,
                        "generated_sample_rate": generated_sr,
                        "chunk_sec": float(args.chunk_sec),
                        "num_inference_steps": int(args.num_inference_steps),
                        "guidance_scale": float(args.guidance_scale),
                        "use_control": False,
                    }
                )

    manifest = {
        "meta": {
            "model_id": args.model_id,
            "prompt": args.prompt,
            "device": str(device),
            "songs": args.songs,
            "chunk_index": int(args.chunk_index),
            "all_chunks": bool(args.all_chunks),
            "chunk_sec": float(args.chunk_sec),
            "seeds": args.seeds,
            "num_rows": len(rows),
        },
        "samples": rows,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
