from __future__ import annotations

import copy
from pathlib import Path

import soundfile as sf
import torch

from models.audioldm2_vae_wrapper import AudioLDM2VAEWrapper
from models.eeg_conditioned_audioldm2 import EEGConditionedAudioLDM2


def get_scheduler_from_model(model: EEGConditionedAudioLDM2):
    pipe = getattr(model.control_unet, "pipeline", None)
    if pipe is None or not hasattr(pipe, "scheduler"):
        raise RuntimeError("The pretrained U-Net wrapper must keep a live pipeline with a scheduler for generation.")
    return copy.deepcopy(pipe.scheduler)


def _prepare_official_latents(
    model: EEGConditionedAudioLDM2,
    *,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    pipe = getattr(model.control_unet, "pipeline", None)
    if pipe is None or not hasattr(pipe, "prepare_latents"):
        raise RuntimeError("The pretrained U-Net wrapper must keep a live pipeline with prepare_latents for generation.")

    # Keep official AudioLDM2 latent layout [B, C, T/4, F/4] throughout the repo.
    official_height = int(model.latent_grid[1]) * int(pipe.vae_scale_factor)
    latents = pipe.prepare_latents(
        batch_size,
        int(model.latent_grid[0]),
        official_height,
        model.control_unet.dtype,
        device,
        generator,
    )
    return latents


@torch.no_grad()
def _generate_latents_official_backbone(
    model: EEGConditionedAudioLDM2,
    *,
    batch_size: int,
    device: torch.device,
    num_inference_steps: int,
    scheduler,
    eta: float,
    generator: torch.Generator | None,
    guidance_scale: float,
    negative_prompt: str,
) -> torch.Tensor:
    pipe = getattr(model.control_unet, "pipeline", None)
    if pipe is None:
        raise RuntimeError("Official backbone generation requires a live AudioLDM2 pipeline.")

    scheduler.set_timesteps(int(num_inference_steps), device=device)
    do_classifier_free_guidance = float(guidance_scale) > 1.0
    height = int(model.latent_grid[1]) * int(pipe.vae_scale_factor)
    latents = pipe.prepare_latents(
        batch_size,
        int(model.latent_grid[0]),
        height,
        model.control_unet.dtype,
        device,
        generator,
    )
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
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents, latents], dim=0)
        if hasattr(scheduler, "scale_model_input"):
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        timestep_batch = torch.full(
            (latent_model_input.shape[0],),
            int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
            device=device,
            dtype=torch.long,
        )
        noise_pred = model.control_unet.backbone(
            sample=latent_model_input,
            timestep=timestep_batch,
            encoder_hidden_states=text_conditioning["encoder_hidden_states"],
            encoder_hidden_states_1=text_conditioning["encoder_hidden_states_1"],
            encoder_attention_mask_1=text_conditioning["attention_mask"],
        )
        noise_pred = noise_pred.sample if hasattr(noise_pred, "sample") else noise_pred
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + float(guidance_scale) * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents, **extra_step_kwargs).prev_sample

    return latents


@torch.no_grad()
def generate_latents(
    model: EEGConditionedAudioLDM2,
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

    if not use_control:
        return _generate_latents_official_backbone(
            model,
            batch_size=batch_size,
            device=device,
            num_inference_steps=num_inference_steps,
            scheduler=scheduler,
            eta=eta,
            generator=generator,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        )

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
    audio_helper: AudioLDM2VAEWrapper,
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
