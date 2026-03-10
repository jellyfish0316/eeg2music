from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.eeg_projector import EEGProjector
from models.eeg_controlnet import EEGControlNetModel
from models.audioldm_unet_wrapper import AudioLDMUNetModel
from scripts.train import apply_freeze_policy, derive_latent_grid


def build_test_model() -> EEGControlNetModel:
    return EEGControlNetModel(
        eeg_channels=12,
        num_subjects=5,
        device="cpu",
        enable_audio_encoder=False,
        latent_channels=8,
        latent_grid=(8, 16, 87),
        projector_use_linear_fallback=True,
        prefer_audioldm_unet=True,
        audioldm_unet_kwargs={
            "image_size": 16,
            "in_channels": 8,
            "model_channels": 32,
            "out_channels": 8,
            "num_res_blocks": 1,
            "attention_resolutions": [1, 2, 4],
            "dropout": 0.0,
            "channel_mult": [1, 2],
            "conv_resample": True,
            "dims": 2,
            "extra_sa_layer": False,
            "extra_film_condition_dim": None,
            "use_checkpoint": False,
            "use_fp16": False,
            "num_heads": 2,
            "num_head_channels": -1,
            "num_heads_upsample": -1,
            "use_scale_shift_norm": False,
            "resblock_updown": False,
            "use_new_attention_order": False,
            "use_spatial_transformer": False,
            "transformer_depth": 1,
            "context_dim": None,
            "n_embed": None,
            "legacy": True,
        },
        controlnet_enabled=True,
        controlnet_copy_encoder_weights=True,
        controlnet_inject_middle_block=True,
    )


def test_paper_aligned_forward_backward_smoke() -> None:
    torch.manual_seed(0)
    model = build_test_model()

    eeg = torch.randn(2, 12, 437)
    z0 = torch.randn(2, 8, 16, 87)
    subject_idx = torch.tensor([0, 1], dtype=torch.long)
    timesteps = model.sample_timesteps(batch_size=2, device=torch.device("cpu"))

    out = model(eeg=eeg, subject_idx=subject_idx, z0=z0, timesteps=timesteps, use_control=True)
    assert out["projected_latent"].shape == (2, 8, 16, 87)
    assert out["eps_pred"].shape == (2, 8, 16, 87)
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    trainable_with_grad = [
        name for name, param in model.named_parameters() if param.requires_grad and param.grad is not None
    ]
    frozen_with_grad = [
        name for name, param in model.named_parameters() if (not param.requires_grad) and param.grad is not None
    ]
    assert trainable_with_grad
    assert not frozen_with_grad


def test_freeze_policy_only_projector_subject_adapter_and_control_branch_trainable() -> None:
    model = build_test_model()
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


def test_audioldm_backend_comes_from_local_vendor_package() -> None:
    assert AudioLDMUNetModel is not None
    assert AudioLDMUNetModel.__module__.startswith("vendor.audioldm_min.")

    model = build_test_model()
    assert model.control_unet.backend_name == "audioldm_unet"


if __name__ == "__main__":
    test_paper_aligned_forward_backward_smoke()
    print("paper alignment smoke ok")
