from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.audioldm_unet_wrapper as unet_wrapper_module
from models.eeg_projector import EEGProjector
from models.eeg_controlnet import EEGControlNetModel
from scripts.train import apply_freeze_policy, derive_latent_grid, validate_model_config


class FakePipelineOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class FakeResnet(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(x))


class FakeDownsampler(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(x))


class FakeDownBlock(nn.Module):
    def __init__(self, channels: int, has_cross_attention: bool) -> None:
        super().__init__()
        self.has_cross_attention = has_cross_attention
        self.resnets = nn.ModuleList([FakeResnet(channels), FakeResnet(channels)])
        self.downsamplers = nn.ModuleList([FakeDownsampler(channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_1: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask_1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        del temb
        del encoder_hidden_states_1
        del attention_mask
        del cross_attention_kwargs
        del encoder_attention_mask
        del encoder_attention_mask_1
        if self.has_cross_attention and encoder_hidden_states is None:
            raise AssertionError("cross-attention block requires encoder_hidden_states")
        res_samples = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
            res_samples.append(hidden_states)
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
            res_samples.append(hidden_states)
        return hidden_states, tuple(res_samples)


class FakeMidBlock(nn.Module):
    has_cross_attention = True

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([FakeResnet(channels)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_1: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask_1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del temb
        del encoder_hidden_states_1
        del attention_mask
        del cross_attention_kwargs
        del encoder_attention_mask
        del encoder_attention_mask_1
        if encoder_hidden_states is None:
            raise AssertionError("mid block requires encoder_hidden_states")
        return self.resnets[0](hidden_states)


class FakeUpBlock(nn.Module):
    has_cross_attention = False

    def __init__(self, channels: int, num_resnets: int = 3) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([FakeResnet(channels) for _ in range(num_resnets)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        res_hidden_states_tuple: tuple[torch.Tensor, ...],
        upsample_size: tuple[int, int] | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        cross_attention_kwargs: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_hidden_states_1: torch.Tensor | None = None,
        encoder_attention_mask_1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del temb
        del upsample_size
        del encoder_hidden_states
        del cross_attention_kwargs
        del attention_mask
        del encoder_attention_mask
        del encoder_hidden_states_1
        del encoder_attention_mask_1
        for resnet, skip in zip(self.resnets, res_hidden_states_tuple):
            hidden_states = hidden_states + skip
            hidden_states = resnet(hidden_states)
        return hidden_states


class FakeTimeProj(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = int(out_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float().unsqueeze(-1).repeat(1, self.out_dim)


class FakeTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, timestep_cond: torch.Tensor | None = None) -> torch.Tensor:
        del timestep_cond
        return self.proj(x)


class FakeUNet(nn.Module):
    def __init__(self, channels: int = 8, model_channels: int = 8) -> None:
        super().__init__()
        self.dtype = torch.float32
        self.conv_in = nn.Conv2d(channels, model_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                FakeDownBlock(model_channels, has_cross_attention=True),
                FakeDownBlock(model_channels, has_cross_attention=False),
            ]
        )
        self.mid_block = FakeMidBlock(model_channels)
        self.out = nn.Conv2d(model_channels, channels, kernel_size=3, padding=1)
        self.time_proj = FakeTimeProj(model_channels)
        self.time_embedding = FakeTimeEmbedding(model_channels)
        self.class_embedding = None
        self.class_embeddings_concat = False
        self.time_embed_act = None
        self.conv_norm_out = None
        self.conv_act = None
        self.conv_out = self.out
        self.up_blocks = nn.ModuleList([FakeUpBlock(model_channels, num_resnets=3)])
        self.num_upsamplers = 0
        self.config = type(
            "FakeConfig",
            (),
            {
                "block_out_channels": [model_channels, model_channels],
                "cross_attention_dim": model_channels,
                "in_channels": channels,
                "out_channels": channels,
                "class_embed_type": None,
                "class_embeddings_concat": False,
            },
        )()

    def eval(self):
        super().eval()
        return self

    def to(self, device=None, dtype=None):
        super().to(device=device, dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
        return self

    def forward(
        self,
        *,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: tuple[torch.Tensor, ...] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
    ) -> FakePipelineOutput:
        temb = self.time_embedding(self.time_proj(timestep).to(dtype=sample.dtype))
        hidden_states = self.conv_in(sample)
        if down_block_additional_residuals is not None:
            hidden_states = hidden_states + down_block_additional_residuals[0].to(hidden_states.dtype)

        residual_index = 1
        for block in self.down_blocks:
            hidden_states, res_samples = block(
                hidden_states=hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
            )
            if down_block_additional_residuals is not None:
                for res in res_samples:
                    hidden_states = hidden_states + down_block_additional_residuals[residual_index].to(res.dtype)
                    residual_index += 1

        hidden_states = self.mid_block(
            hidden_states=hidden_states,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
        )
        if mid_block_additional_residual is not None:
            hidden_states = hidden_states + mid_block_additional_residual.to(hidden_states.dtype)
        return FakePipelineOutput(self.out(hidden_states))


class FakeAudioLDM2Pipeline:
    def __init__(self, unet: nn.Module) -> None:
        self.unet = unet

    @classmethod
    def from_pretrained(cls, model_id: str, torch_dtype: torch.dtype):
        del model_id, torch_dtype
        return cls(FakeUNet())


def install_fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unet_wrapper_module, "AudioLDM2Pipeline", FakeAudioLDM2Pipeline)


def build_test_model(monkeypatch: pytest.MonkeyPatch) -> EEGControlNetModel:
    install_fake_pipeline(monkeypatch)
    return EEGControlNetModel(
        eeg_channels=12,
        num_subjects=5,
        device="cpu",
        enable_audio_encoder=False,
        latent_channels=8,
        latent_grid=(8, 16, 87),
        projector_use_linear_fallback=True,
        audio_model_id="fake/audioldm2",
        controlnet_enabled=True,
        controlnet_copy_encoder_weights=True,
        controlnet_inject_middle_block=True,
    )


def test_pretrained_unet_wrapper_uses_pipeline_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pipeline(monkeypatch)
    wrapper = unet_wrapper_module.AudioLDMUNetWrapper(
        model_id="fake/audioldm2",
        device="cpu",
        dtype=torch.float32,
        cache_pipeline=True,
    )
    assert wrapper.backend_name == "diffusers_pretrained_unet"
    assert wrapper.pipeline is not None
    assert isinstance(wrapper.backbone, FakeUNet)


def test_pretrained_unet_load_failure_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenPipeline:
        @classmethod
        def from_pretrained(cls, model_id: str, torch_dtype: torch.dtype):
            del cls, model_id, torch_dtype
            raise RuntimeError("load failed")

    monkeypatch.setattr(unet_wrapper_module, "AudioLDM2Pipeline", BrokenPipeline)
    with pytest.raises(RuntimeError, match="load failed"):
        unet_wrapper_module.AudioLDMUNetWrapper(
            model_id="broken/model",
            device="cpu",
            dtype=torch.float32,
        )


def test_deprecated_unet_config_keys_fail_cleanly() -> None:
    with pytest.raises(ValueError, match="prefer_audioldm_unet is deprecated"):
        validate_model_config({"prefer_audioldm_unet": True})
    with pytest.raises(ValueError, match="audioldm_unet_kwargs is deprecated"):
        validate_model_config({"audioldm_unet_kwargs": {}})


def test_paper_aligned_forward_backward_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    model = build_test_model(monkeypatch)

    eeg = torch.randn(2, 12, 437)
    z0 = torch.randn(2, 8, 16, 87)
    subject_idx = torch.tensor([0, 1], dtype=torch.long)
    timesteps = model.sample_timesteps(batch_size=2, device=torch.device("cpu"))

    out = model(eeg=eeg, subject_idx=subject_idx, z0=z0, timesteps=timesteps, use_control=True)
    assert out["projected_latent"].shape == (2, 8, 16, 87)
    assert out["eps_pred"].shape == (2, 8, 16, 87)
    assert torch.isfinite(out["loss"])
    assert out["control_residuals"] is not None
    assert len(out["control_residuals"]["down_block_residuals"]) == len(
        model.control_unet.control_specs["input_block_channels"]
    )

    out["loss"].backward()
    trainable_with_grad = [
        name for name, param in model.named_parameters() if param.requires_grad and param.grad is not None
    ]
    assert trainable_with_grad


def test_freeze_policy_only_projector_subject_adapter_and_control_branch_trainable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_test_model(monkeypatch)
    stats = apply_freeze_policy(
        model,
        {
            "enabled": True,
            "freeze_base_unet": True,
            "trainable_modules": ["subject_adapter", "projector", "control_branch"],
        },
    )
    assert stats["total_trainable"] > 0

    trainable_prefixes = ("subject_adapter.", "projector.", "control_branch.")
    saw_trainable = False
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            assert param.requires_grad, name
            saw_trainable = True
        else:
            assert not param.requires_grad, name
    assert saw_trainable


def test_derive_latent_grid_prefers_config_then_cache_then_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDataset:
        def __init__(self, z0_by_chunk):
            self.z0_by_chunk = z0_by_chunk

    cfg = {
        "data": {"audio_fs": 16000, "audio_samples": 56000},
        "audio_encoder": {"model_id": "dummy", "sample_rate": 16000, "freeze_vae": True},
        "latent_cache": {"precompute_use_mode": True},
        "model": {"projector": {"lat_grid": [8, 12, 34]}},
    }
    ds = DummyDataset(z0_by_chunk=torch.randn(4, 8, 16, 87))
    assert derive_latent_grid(cfg, dataset=ds, device=torch.device("cpu")) == (8, 12, 34)

    cfg["model"]["projector"]["lat_grid"] = None
    assert derive_latent_grid(cfg, dataset=ds, device=torch.device("cpu")) == (8, 16, 87)

    ds.z0_by_chunk = None

    class DummyEncoder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def infer_latent_shape(self, num_audio_samples: int) -> tuple[int, int, int]:
            assert num_audio_samples == 56000
            return (8, 9, 10)

    monkeypatch.setattr("scripts.train.AudioLDM2MusicEncoderWrapper", DummyEncoder)
    assert derive_latent_grid(cfg, dataset=ds, device=torch.device("cpu")) == (8, 9, 10)


def test_projector_uses_linear_fallback_only_when_needed() -> None:
    projector = EEGProjector(
        in_channels=12,
        latent_grid=(8, 16, 87),
        use_linear_fallback=True,
    )
    eeg = torch.randn(2, 12, 437)
    out = projector(eeg)
    assert out.shape == (2, 8, 16, 87)
    assert projector.linear_fallback is not None


def test_projector_raises_when_fallback_disabled_and_temporal_length_mismatches() -> None:
    projector = EEGProjector(
        in_channels=12,
        latent_grid=(8, 16, 87),
        use_linear_fallback=False,
    )
    eeg = torch.randn(2, 12, 437)
    with pytest.raises(RuntimeError, match="linear fallback is disabled"):
        projector(eeg)


if __name__ == "__main__":
    test_projector_uses_linear_fallback_only_when_needed()
    print("paper alignment smoke ok")
