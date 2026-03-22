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


def _prepare_official_latents(
    model: EEGControlNetModel,
    *,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    pipe = getattr(model.control_unet, "pipeline", None)
    if pipe is None or not hasattr(pipe, "prepare_latents"):
        raise RuntimeError("The pretrained U-Net wrapper must keep a live pipeline with prepare_latents for generation.")

    # Official AudioLDM2 prepare_latents expects latents laid out as [B, C, T/4, F/4].
    # Our training cache stores latents in [B, C, F/4, T/4], so we transpose back after
    # sampling to stay consistent with the trained ControlNet/projector path.
    official_height = int(model.latent_grid[2]) * int(pipe.vae_scale_factor)
    latents = pipe.prepare_latents(
        batch_size,
        int(model.latent_grid[0]),
        official_height,
        model.control_unet.dtype,
        device,
        generator,
    )
    return latents.transpose(-1, -2).contiguous()


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
    guidance_scale: float = 3.5,
    negative_prompt: str = "",
) -> torch.Tensor:
    model.eval()
    device = eeg.device
    batch_size = int(eeg.shape[0])

    if scheduler is None:
        scheduler = get_scheduler_from_model(model)
    scheduler.set_timesteps(int(num_inference_steps), device=device)
    do_classifier_free_guidance = float(guidance_scale) > 1.0

    latents = _prepare_official_latents(
        model,
        batch_size=batch_size,
        device=device,
        generator=generator,
    )

    extra_step_kwargs = {}
    pipe = getattr(model.control_unet, "pipeline", None)
    if pipe is not None and hasattr(pipe, "prepare_extra_step_kwargs"):
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    text_conditioning = model.control_unet.get_text_conditioning_with_guidance(
        batch_size=batch_size,
        device=device,
        dtype=model.control_unet.dtype,
        guidance_scale=float(guidance_scale),
        negative_prompt=negative_prompt,
    )

    for timestep in scheduler.timesteps:
        latent_model_input = latents
        eeg_input = eeg
        subject_input = subject_idx
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents, latents], dim=0)
            eeg_input = torch.cat([eeg, eeg], dim=0)
            subject_input = torch.cat([subject_idx, subject_idx], dim=0)
        if hasattr(scheduler, "scale_model_input"):
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        timestep_batch = torch.full(
            (latent_model_input.shape[0],),
            int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
            device=device,
            dtype=torch.long,
        )
        pred = model.predict_noise(
            eeg=eeg_input,
            subject_idx=subject_input,
            zt=latent_model_input,
            timesteps=timestep_batch,
            use_control=use_control,
            control_scale=control_scale,
            text_conditioning=text_conditioning,
        )
        noise_pred = pred["eps_pred"]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)
        step_out = scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs)
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
