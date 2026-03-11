from __future__ import annotations

import copy
from pathlib import Path

import soundfile as sf
import torch

from models.audioldm2_wrapper import AudioLDM2MusicEncoderWrapper
from models.eeg_controlnet import EEGControlNetModel


def get_scheduler_from_model(model: EEGControlNetModel):
    pipe = getattr(model.control_unet, "pipeline", None)
    if pipe is None or not hasattr(pipe, "scheduler"):
        raise RuntimeError("The pretrained U-Net wrapper must keep a live pipeline with a scheduler for generation.")
    return copy.deepcopy(pipe.scheduler)


@torch.no_grad()
def generate_latents(
    model: EEGControlNetModel,
    *,
    eeg: torch.Tensor,
    subject_idx: torch.Tensor,
    num_inference_steps: int,
    scheduler=None,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    use_control: bool = True,
    control_scale: float | None = None,
) -> torch.Tensor:
    model.eval()
    device = eeg.device
    batch_size = int(eeg.shape[0])

    if scheduler is None:
        scheduler = get_scheduler_from_model(model)
    scheduler.set_timesteps(int(num_inference_steps), device=device)

    latent_shape = (
        batch_size,
        int(model.latent_grid[0]),
        int(model.latent_grid[1]),
        int(model.latent_grid[2]),
    )
    latents = torch.randn(
        latent_shape,
        device=device,
        dtype=model.control_unet.dtype,
        generator=generator,
    )

    extra_step_kwargs = {}
    pipe = getattr(model.control_unet, "pipeline", None)
    if pipe is not None and hasattr(pipe, "prepare_extra_step_kwargs"):
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    for timestep in scheduler.timesteps:
        latent_model_input = latents
        if hasattr(scheduler, "scale_model_input"):
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        timestep_batch = torch.full(
            (batch_size,),
            int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
            device=device,
            dtype=torch.long,
        )
        pred = model.predict_noise(
            eeg=eeg,
            subject_idx=subject_idx,
            zt=latent_model_input,
            timesteps=timestep_batch,
            use_control=use_control,
            control_scale=control_scale,
        )
        step_out = scheduler.step(pred["eps_pred"], timestep, latents, **extra_step_kwargs)
        latents = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

    return latents


@torch.no_grad()
def batch_clap_similarity(
    audio_helper: AudioLDM2MusicEncoderWrapper,
    predicted_waveforms: torch.Tensor,
    target_waveforms: torch.Tensor,
    *,
    sample_rate: int,
) -> torch.Tensor:
    return audio_helper.compute_audio_similarity(
        predicted_waveforms,
        target_waveforms,
        sample_rate=sample_rate,
    )


def save_waveforms(
    waveforms: torch.Tensor,
    *,
    output_dir: Path,
    filenames: list[str],
    sample_rate: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    waveforms = waveforms.detach().cpu().float()
    for waveform, name in zip(waveforms, filenames):
        path = output_dir / name
        sf.write(path, waveform.numpy(), samplerate=sample_rate)
        written.append(str(path))
    return written
