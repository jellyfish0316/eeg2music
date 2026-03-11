from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.train as train_module


class DummyDataset:
    eeg_out_channels = 12
    total_subjects = 5
    z0_by_chunk = torch.randn(4, 8, 16, 87)

    def __len__(self) -> int:
        return 2


class DummyTrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.control_unet = type("DummyControlUNet", (), {"backend_name": "fake_backend"})()
        self.projector = nn.Identity()

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, dtype=torch.long, device=device)

    def forward(
        self,
        *,
        eeg: torch.Tensor,
        subject_idx: torch.Tensor,
        audio: torch.Tensor | None = None,
        z0: torch.Tensor | None = None,
        timesteps: torch.Tensor,
        use_control: bool,
        control_scale: float,
    ) -> dict[str, torch.Tensor]:
        del eeg, subject_idx, audio, z0, timesteps, use_control, control_scale
        loss = self.weight.square()
        latent = torch.ones(1, 1, 1, 1, dtype=loss.dtype, device=loss.device)
        return {
            "loss": loss,
            "z0": latent,
            "zt": latent,
            "noise": latent,
            "projected_latent": latent,
            "eps_pred": latent,
            "use_control": torch.tensor(True),
        }


def make_cfg(*, epochs: int = 1, validation_metric: str = "clap") -> dict:
    return {
        "seed": 42,
        "experiment": {
            "active_instruments": ["drum", "guitar", "vocal"],
            "conditions": ["multi_attention", "passive_x3"],
        },
        "data": {
            "audio_fs": 16000,
            "eeg_time": 437,
            "text_prompt": "Pop music",
            "batch_size": 2,
        },
        "model": {
            "projector": {"channels": [256, 512, 1024, 2048], "strides": [5, 2, 2, 2], "use_linear_fallback": True},
            "unet": {"cache_pipeline": True, "text_cache_path": None},
            "use_subject_adapter": True,
            "subject_emb_dim": 64,
        },
        "audio_encoder": {
            "model_id": "fake/audioldm2",
            "sample_rate": 16000,
            "freeze_vae": True,
            "use_mode": False,
        },
        "latent_cache": {"enabled": True, "latent_channels": 8, "path": "unused.pt", "precompute_use_mode": True},
        "controlnet": {
            "enabled": True,
            "control_scale": 1.0,
            "freeze_base_unet": True,
            "trainable_modules": ["subject_adapter", "projector", "control_branch"],
        },
        "diffusion": {"num_train_timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02},
        "train": {
            "lr": 1e-4,
            "epochs": epochs,
            "grad_clip": 1.0,
            "log_every": 10,
            "validation_metric": validation_metric,
            "validation_generate_batches": 1,
            "validation_num_inference_steps": 5,
            "best_checkpoint_name": "best_model.pt",
            "checkpoint_name": "model.pt",
        },
    }


def install_fake_training_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    batches = [
        {
            "eeg": torch.randn(2, 12, 437),
            "subject_idx": torch.tensor([0, 1], dtype=torch.long),
            "audio": torch.randn(2, 56000),
        }
    ]

    def fake_build_dataloader(cfg, *, condition_type, target_instrument, subjects, shuffle):
        del cfg, condition_type, target_instrument, subjects, shuffle
        return DummyDataset(), batches

    monkeypatch.setattr(train_module, "build_dataloader", fake_build_dataloader)
    monkeypatch.setattr(train_module, "build_model_from_dataset", lambda cfg, dataset, device: DummyTrainModel())
    monkeypatch.setattr(train_module, "apply_freeze_policy", lambda model, control_cfg: {"total_trainable": 1})


def test_run_one_condition_clap_validation_writes_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_training_stack(monkeypatch)
    monkeypatch.setattr(train_module, "evaluate_loss", lambda *args, **kwargs: 0.25)
    monkeypatch.setattr(train_module, "evaluate_generation_clap", lambda *args, **kwargs: 0.42)
    monkeypatch.setattr(train_module, "AudioLDM2MusicEncoderWrapper", lambda *args, **kwargs: object())

    result = train_module.run_one_condition(
        make_cfg(),
        fold_meta={"fold_index": 0, "train_subjects": [1], "val_subjects": [2], "test_subjects": [3]},
        condition_job={"condition_name": "multi_attention", "condition_type": "multi_attention", "target_instrument": ""},
        device=torch.device("cpu"),
        output_dir=tmp_path,
        max_steps=1,
    )

    assert result["history"]["val_clap"] == [0.42]
    assert result["best_metric_name"] == "val_clap"
    assert result["best_metric_value"] == pytest.approx(0.42)
    assert Path(result["best_checkpoint_path"]).name == "best_model.pt"
    assert Path(result["best_checkpoint_path"]).exists()
    assert Path(result["checkpoint_path"]).exists()


def test_run_one_condition_prefers_higher_clap_over_lower_val_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_training_stack(monkeypatch)
    val_losses = iter([0.10, 0.01, 0.30])
    val_claps = iter([0.20, 0.35])

    monkeypatch.setattr(train_module, "evaluate_loss", lambda *args, **kwargs: next(val_losses))
    monkeypatch.setattr(train_module, "evaluate_generation_clap", lambda *args, **kwargs: next(val_claps))
    monkeypatch.setattr(train_module, "AudioLDM2MusicEncoderWrapper", lambda *args, **kwargs: object())

    result = train_module.run_one_condition(
        make_cfg(epochs=2),
        fold_meta={"fold_index": 0, "train_subjects": [1], "val_subjects": [2], "test_subjects": [3]},
        condition_job={"condition_name": "passive_x3", "condition_type": "passive_x3", "target_instrument": ""},
        device=torch.device("cpu"),
        output_dir=tmp_path,
        max_steps=1,
    )

    assert result["history"]["val_loss"] == [0.10, 0.01]
    assert result["history"]["val_clap"] == [0.20, 0.35]
    assert result["best_metric_name"] == "val_clap"
    assert result["best_metric_value"] == pytest.approx(0.35)
    assert Path(result["best_checkpoint_path"]).exists()


def test_run_one_condition_clap_validation_requires_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_training_stack(monkeypatch)
    monkeypatch.setattr(train_module, "evaluate_loss", lambda *args, **kwargs: 0.25)

    def broken_helper(*args, **kwargs):
        raise RuntimeError("CLAP unavailable")

    monkeypatch.setattr(train_module, "AudioLDM2MusicEncoderWrapper", broken_helper)

    with pytest.raises(RuntimeError, match="CLAP unavailable"):
        train_module.run_one_condition(
            make_cfg(),
            fold_meta={"fold_index": 0, "train_subjects": [1], "val_subjects": [2], "test_subjects": [3]},
            condition_job={"condition_name": "multi_attention", "condition_type": "multi_attention", "target_instrument": ""},
            device=torch.device("cpu"),
            output_dir=tmp_path,
            max_steps=1,
        )
