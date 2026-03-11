from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.audioldm2_wrapper as audio_wrapper_module
from models.audioldm2_wrapper import AudioLDM2MusicEncoderWrapper
from utils.generation import generate_latents


class FakeDecodeOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class FakeLatentDist:
    def __init__(self, mel: torch.Tensor) -> None:
        pooled = F.avg_pool2d(mel, kernel_size=(4, 4), stride=(4, 4))
        self._z = pooled.repeat(1, 4, 1, 1)

    def mode(self) -> torch.Tensor:
        return self._z

    def sample(self) -> torch.Tensor:
        return self._z


class FakeEncodeOutput:
    def __init__(self, mel: torch.Tensor) -> None:
        self.latent_dist = FakeLatentDist(mel)


class FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"scaling_factor": 0.5, "latent_channels": 4})()

    def encode(self, mel: torch.Tensor) -> FakeEncodeOutput:
        return FakeEncodeOutput(mel)

    def decode(self, latents: torch.Tensor) -> FakeDecodeOutput:
        mel = latents.mean(dim=1, keepdim=True)
        mel = F.interpolate(mel, size=(64, 32), mode="bilinear", align_corners=False)
        return FakeDecodeOutput(mel)


class FakeVocoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"sampling_rate": 16000})()

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return mel.mean(dim=1).mean(dim=1)


class FakeFeatureExtractor:
    sampling_rate = 16000

    def __call__(self, audio_list, return_tensors: str, sampling_rate: int):
        del return_tensors, sampling_rate
        max_len = max(len(x) for x in audio_list)
        padded = []
        for x in audio_list:
            tensor = torch.tensor(x, dtype=torch.float32)
            if tensor.numel() < max_len:
                tensor = F.pad(tensor, (0, max_len - tensor.numel()))
            padded.append(tensor)
        features = torch.stack(padded, dim=0).unsqueeze(1)
        return type("Out", (), {"input_features": features})()


class FakeClapModel(nn.Module):
    dtype = torch.float32

    def get_audio_features(self, input_features: torch.Tensor) -> torch.Tensor:
        pooled = input_features.mean(dim=-1)
        return torch.cat([pooled, pooled], dim=-1)


class FakeAudioPipeline:
    def __init__(self) -> None:
        self.vae = FakeVAE()
        self.vocoder = FakeVocoder()
        self.text_encoder = FakeClapModel()
        self.feature_extractor = FakeFeatureExtractor()

    @classmethod
    def from_pretrained(cls, model_id: str, torch_dtype: torch.dtype):
        del model_id, torch_dtype
        return cls()

    def to(self, device):
        del device
        return self

    def mel_spectrogram_to_waveform(self, mel: torch.Tensor) -> torch.Tensor:
        return self.vocoder(mel).cpu().float()


class DummyControlUNet:
    dtype = torch.float32


class DummyScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([], dtype=torch.long)

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
        del device
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1, dtype=torch.long)

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return sample

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, **kwargs):
        del timestep, kwargs
        return type("StepOut", (), {"prev_sample": sample - 0.1 * model_output})()


class DummyModel:
    def __init__(self) -> None:
        self.control_unet = DummyControlUNet()
        self.latent_grid = (4, 8, 8)
        self.calls = 0

    def eval(self):
        return self

    def predict_noise(self, eeg, subject_idx, zt, timesteps, control_scale=None, use_control=True):
        del eeg, subject_idx, timesteps, control_scale, use_control
        self.calls += 1
        return {"eps_pred": torch.zeros_like(zt)}


def test_decode_latents_and_audio_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_wrapper_module, "AudioLDM2Pipeline", FakeAudioPipeline)
    wrapper = AudioLDM2MusicEncoderWrapper(
        model_id="fake/audioldm2",
        sample_rate=16000,
        device="cpu",
        dtype=torch.float32,
        freeze_vae=True,
        use_mode=True,
    )
    latents = torch.randn(2, 4, 8, 8)
    mel = wrapper.decode_latents_to_mel(latents)
    waveform = wrapper.decode_latents_to_waveform(latents)
    assert mel.shape == (2, 1, 64, 32)
    assert waveform.dim() == 2
    sims = wrapper.compute_audio_similarity(waveform, waveform, sample_rate=16000)
    assert sims.shape == (2,)
    assert torch.allclose(sims, torch.ones_like(sims), atol=1e-4)


def test_generate_latents_runs_fixed_number_of_steps() -> None:
    model = DummyModel()
    scheduler = DummyScheduler()
    eeg = torch.randn(3, 12, 437)
    subject_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    latents = generate_latents(
        model,
        eeg=eeg,
        subject_idx=subject_idx,
        num_inference_steps=5,
        scheduler=scheduler,
        use_control=True,
    )
    assert latents.shape == (3, 4, 8, 8)
    assert model.calls == 5
