from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa
import numpy as np
import soundfile as sf
import torch

from models.audioldm2_vae_wrapper import AudioLDM2VAEWrapper
from utils.vendor_audioldm2_audio import TacotronSTFT, normalize_wav, pad_wav


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check wav->mel->vocoder / VAE roundtrip compatibility.")
    p.add_argument("--audio", type=str, required=True, help="Input wav path")
    p.add_argument("--model-id", type=str, default="cvssp/audioldm2-music")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--output-dir", type=str, default="outputs/mel_vocoder_check")
    return p.parse_args()


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio.astype(np.float32, copy=False)


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(reference.shape[-1], estimate.shape[-1])
    ref = reference[:length].astype(np.float64, copy=False)
    est = estimate[:length].astype(np.float64, copy=False)
    noise = ref - est
    ref_power = np.mean(ref ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    return float(10.0 * np.log10(ref_power / noise_power))


def aligned_abs_diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    length = min(a.shape[-1], b.shape[-1])
    x = a[:length].astype(np.float64, copy=False)
    y = b[:length].astype(np.float64, copy=False)
    diff = np.abs(x - y)
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
    }


def mel_from_wrapper(wrapper: AudioLDM2VAEWrapper, audio: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(audio).unsqueeze(0)
    return wrapper.waveform_to_mel(waveform)


def mel_manual(
    *,
    audio: np.ndarray,
    wrapper: AudioLDM2VAEWrapper,
    do_normalize: bool,
    do_pad: bool,
) -> torch.Tensor:
    sample = audio.astype(np.float32, copy=False)
    if do_normalize:
        sample = normalize_wav(sample).astype(np.float32, copy=False)
    if do_pad:
        sample = pad_wav(sample[None, ...], target_length=sample.shape[-1])[0]

    waveform = torch.from_numpy(sample).unsqueeze(0).to(device=wrapper.device, dtype=torch.float32)
    mel_log, _, _, _ = wrapper.tacotron_stft.mel_spectrogram(waveform, normalize_fun=torch.log)
    mel_log = mel_log.transpose(-1, -2).contiguous()
    return mel_log.unsqueeze(1).to(dtype=wrapper.dtype)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    wrapper = AudioLDM2VAEWrapper(
        model_id=args.model_id,
        sample_rate=int(args.sample_rate),
        device=device,
        dtype=dtype,
        freeze_vae=True,
        use_mode=True,
    )
    pipe = wrapper._load_full_pipeline()

    audio = load_audio(args.audio, int(args.sample_rate))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "wrapper_default": mel_from_wrapper(wrapper, audio),
        "no_normalize": mel_manual(audio=audio, wrapper=wrapper, do_normalize=False, do_pad=True),
        "no_pad": mel_manual(audio=audio, wrapper=wrapper, do_normalize=True, do_pad=False),
        "raw": mel_manual(audio=audio, wrapper=wrapper, do_normalize=False, do_pad=False),
    }

    summary: dict[str, object] = {
        "audio_path": str(Path(args.audio).resolve()),
        "sample_rate": int(args.sample_rate),
        "vocoder_sample_rate": int(pipe.vocoder.config.sampling_rate),
        "vocoder_model_in_dim": int(pipe.vocoder.config.model_in_dim),
        "variants": {},
    }

    ref_path = out_dir / "reference.wav"
    sf.write(ref_path, audio, int(args.sample_rate))

    for name, mel in variants.items():
        waveform_vocoder = pipe.mel_spectrogram_to_waveform(mel)
        if waveform_vocoder.dim() == 1:
            waveform_vocoder = waveform_vocoder.unsqueeze(0)
        vocoder_audio = waveform_vocoder[0].detach().cpu().float().numpy()

        latents = wrapper.encode_mel(mel, sample_posterior=False).latents
        decoded_mel = wrapper.decode_latents_to_mel(latents)
        vae_audio = wrapper.decode_latents_to_waveform(latents)[0].detach().cpu().float().numpy()

        mel_abs_diff = (decoded_mel.float().cpu() - mel.float().cpu()).abs()

        vocoder_path = out_dir / f"{name}_mel_to_vocoder.wav"
        vae_path = out_dir / f"{name}_vae_roundtrip.wav"
        sf.write(vocoder_path, vocoder_audio, int(wrapper.vocoder_sample_rate))
        sf.write(vae_path, vae_audio, int(wrapper.vocoder_sample_rate))

        summary["variants"][name] = {
            "mel_shape": list(mel.shape),
            "vocoder_snr_db": snr_db(audio, vocoder_audio),
            "vae_roundtrip_snr_db": snr_db(audio, vae_audio),
            "vocoder_vs_vae_waveform": aligned_abs_diff_stats(vocoder_audio, vae_audio),
            "mel_vs_decoded_mel": {
                "mean_abs_diff": float(mel_abs_diff.mean().item()),
                "max_abs_diff": float(mel_abs_diff.max().item()),
            },
            "vocoder_wav": str(vocoder_path),
            "vae_roundtrip_wav": str(vae_path),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
