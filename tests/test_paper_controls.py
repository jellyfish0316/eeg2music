from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.generate_pretrained_baseline as pretrained_module
from scripts.prepare_cdt_eeg import _fallback_curry_paths
from scripts.run_paper_evaluation import verify_correct_regressions
from utils.evaluation_analysis import (
    hierarchical_cluster_bootstrap,
    paired_bootstrap,
    summarize_paper_records,
    write_paper_outputs,
)
from utils.evaluation_pairing import (
    EvaluationPairingDataset,
    assert_passive3_tensor,
    build_cyclic_derangements,
    identity_mapping,
    keys_from_index_map,
    stable_generation_seed,
    validate_mapping,
)


class SyntheticDataset:
    def __init__(self, *, subjects: tuple[int, ...] = (0, 1), chunks: int = 48, passive3: bool = False) -> None:
        self.index_map = [(0, subject, chunk) for subject in subjects for chunk in range(chunks)]
        self.base_eeg_channels = 2
        self.passive3 = passive3

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):
        song, subject, chunk = self.index_map[index]
        base = torch.full((self.base_eeg_channels, 5), float(index))
        eeg = torch.cat([base, base, base], dim=0) if self.passive3 else torch.cat([base, base + 1, base + 2], dim=0)
        return {
            "eeg": eeg,
            "audio": torch.full((16,), float(chunk)),
            "song_idx": torch.tensor(song),
            "subject_idx": torch.tensor(subject),
            "chunk_idx": torch.tensor(chunk),
            "song_name": "song7",
        }


def test_cyclic_derangements_are_deterministic_unique_and_constrained() -> None:
    dataset = SyntheticDataset()
    keys = keys_from_index_map(dataset.index_map)
    mappings = build_cyclic_derangements(keys, num_permutations=10, seed=0, min_chunk_distance=5)
    repeated = build_cyclic_derangements(keys, num_permutations=10, seed=0, min_chunk_distance=5)
    assert mappings == repeated
    assert len({tuple(sorted(mapping.items())) for mapping in mappings}) == 10
    for mapping in mappings:
        validate_mapping(keys, mapping, min_chunk_distance=5, allow_identity=False)


def test_derangement_fails_when_group_cannot_satisfy_distance() -> None:
    keys = keys_from_index_map(SyntheticDataset(subjects=(0,), chunks=8).index_map)
    with pytest.raises(ValueError, match="Not enough distinct legal"):
        build_cyclic_derangements(keys, num_permutations=5, seed=0, min_chunk_distance=4)


def test_multicond_moves_whole_source_sample() -> None:
    dataset = SyntheticDataset(subjects=(0,), chunks=48, passive3=False)
    keys = keys_from_index_map(dataset.index_map)
    mapping = build_cyclic_derangements(keys, num_permutations=1, seed=3, min_chunk_distance=5)[0]
    paired = EvaluationPairingDataset(
        dataset,
        mapping,
        mode="shuffled",
        condition_sources=["guitar", "vocal", "drum"],
        base_eeg_channels=2,
    )
    target_index = 0
    source_index = mapping[target_index]
    assert torch.equal(paired[target_index]["eeg"], dataset[source_index]["eeg"])
    assert torch.equal(paired[target_index]["audio"], dataset[target_index]["audio"])


def test_passive3_remains_exactly_triplicated_after_shuffle() -> None:
    dataset = SyntheticDataset(subjects=(0,), chunks=48, passive3=True)
    keys = keys_from_index_map(dataset.index_map)
    mapping = build_cyclic_derangements(keys, num_permutations=1, seed=9, min_chunk_distance=5)[0]
    paired = EvaluationPairingDataset(
        dataset,
        mapping,
        mode="shuffled",
        condition_sources=["passive", "passive", "passive"],
        base_eeg_channels=2,
    )
    assert_passive3_tensor(paired[0]["eeg"], base_channels=2)


def test_correct_mode_is_identity_and_seed_ignores_subject() -> None:
    dataset = SyntheticDataset()
    keys = keys_from_index_map(dataset.index_map)
    paired = EvaluationPairingDataset(
        dataset,
        identity_mapping(keys),
        mode="correct",
        condition_sources=["guitar", "vocal", "drum"],
        base_eeg_channels=2,
    )
    assert torch.equal(paired[5]["eeg"], dataset[5]["eeg"])
    assert stable_generation_seed(base_seed=42, song_id="song7", chunk_index=4) == stable_generation_seed(
        base_seed=42, song_id="song7", chunk_index=4
    )
    assert stable_generation_seed(base_seed=42, song_id="song7", chunk_index=4) != stable_generation_seed(
        base_seed=42, song_id="song7", chunk_index=5
    )


class FakePretrainedPipeline:
    def __init__(self) -> None:
        self.vocoder = type("Vocoder", (), {"config": type("Config", (), {"sampling_rate": 16000})()})()

    @classmethod
    def from_pretrained(cls, model_id: str, torch_dtype: torch.dtype):
        assert model_id == "fake/model"
        del torch_dtype
        return cls()

    def to(self, device):
        del device
        return self

    def __call__(self, prompt: str, **kwargs):
        assert prompt == "Pop music"
        assert kwargs["generator"] is not None
        return type("Output", (), {"audios": [np.zeros(56000, dtype=np.float32)]})()


def test_pretrained_generation_uses_target_manifest_without_eeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(pretrained_module, "AudioLDM2Pipeline", FakePretrainedPipeline)
    target = tmp_path / "song7.wav"
    sf.write(target, np.zeros(112000, dtype=np.float32), 16000)
    manifest = pretrained_module.generate_pretrained_from_targets(
        target_rows=[{"target_song_id": "song7", "target_chunk_id": 0, "target_audio_path": str(target)}],
        output_dir=tmp_path / "out",
        model_id="fake/model",
        prompt="Pop music",
        chunk_sec=3.5,
        generation_seed=42,
        num_inference_steps=1,
        guidance_scale=3.5,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    row = payload["samples"][0]
    assert row["evaluation_mode"] == "pretrained"
    assert row["checkpoint"] is None
    assert row["eeg_source_chunk_id"] is None
    assert row["use_control"] is False


def _record(mode: str, score: float, chunk: int, *, permutation: int | None = None, model="MULTICOND"):
    return {
        "evaluation_mode": mode,
        "job_id": "multi" if model == "MULTICOND" else "passive",
        "model_type": model,
        "checkpoint": "model.pt",
        "regime": "train_all_s013",
        "subject": "S0",
        "held_out_subject": None,
        "permutation_id": permutation,
        "permutation_seed": permutation,
        "target_chunk_id": chunk,
        "target_song_id": "song7",
        "target_subject_id": 0,
        "eeg_source_chunk_id": chunk + 5 if mode == "shuffled" else chunk,
        "eeg_source_song_id": "song7" if mode != "pretrained" else None,
        "eeg_source_subject_id": 0 if mode != "pretrained" else None,
        "generation_seed": chunk + 100,
        "overall_clap": score,
        "drum_clap": score,
        "guitar_proxy_other_bass_clap": score,
        "vocal_clap": score,
    }


def test_summary_uses_per_target_shuffle_mean_and_writes_outputs(tmp_path: Path) -> None:
    records = []
    for chunk in (0, 1):
        records.append(_record("correct", 0.7, chunk))
        records.append(_record("correct", 0.6, chunk, model="PASSIVE3"))
        records.append(_record("shuffled", 0.4, chunk, permutation=0))
        records.append(_record("shuffled", 0.5, chunk, permutation=1))
        records.append(_record("shuffled", 0.35, chunk, permutation=0, model="PASSIVE3"))
        records.append(_record("shuffled", 0.45, chunk, permutation=1, model="PASSIVE3"))
        pretrained = _record("pretrained", 0.3, chunk)
        pretrained.update({"job_id": "pretrained_prior", "model_type": "AudioLDM2", "target_subject_id": None})
        records.append(pretrained)
    summary = summarize_paper_records(records, num_resamples=200, seed=0)
    multi = next(row for row in summary["jobs"] if row["job_id"] == "multi")
    assert multi["correct_mean"] == pytest.approx(0.7)
    assert multi["shuffled_mean"] == pytest.approx(0.45)
    assert multi["correct_minus_shuffled"]["mean_difference"] == pytest.approx(0.25)
    assert multi["correct_minus_pretrained"]["mean_difference"] == pytest.approx(0.4)
    assert summary["protocol_gain_multicond_minus_passive3"][0]["mean_difference"] == pytest.approx(0.1)
    write_paper_outputs(records, summary, output_dir=tmp_path / "metrics")
    assert (tmp_path / "metrics/per_chunk_scores.csv").exists()
    assert (tmp_path / "metrics/summary.csv").exists()
    assert (tmp_path / "metrics/summary.json").exists()


def test_hierarchical_bootstrap_resamples_subjects_and_reports_subject_means() -> None:
    result = hierarchical_cluster_bootstrap(
        {0: [0.1, 0.2], 1: [0.3, 0.4], 3: [0.5, 0.6]},
        num_resamples=500,
        seed=4,
    )
    assert result["n_clusters"] == 3
    assert result["mean_difference"] == pytest.approx(0.35)
    assert result["cluster_means"] == pytest.approx({"0": 0.15, "1": 0.35, "3": 0.55})
    assert result["ci95_low"] < result["mean_difference"] < result["ci95_high"]


def test_hierarchical_bootstrap_refuses_population_inference_for_one_subject() -> None:
    result = hierarchical_cluster_bootstrap({0: [0.1, 0.2]}, num_resamples=20, seed=0)
    assert result["n_clusters"] == 1
    assert result["p_value"] is None
    assert result["ci95_low"] is None


def test_yaml_registry_contains_all_table1_jobs() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/evaluate_paper_controls.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(config["jobs"]) == 16
    assert all(job.get("archived_manifest", "").endswith("/manifest.json") for job in config["jobs"])
    train_all = [job for job in config["jobs"] if job["regime"] == "train_all_s013"]
    assert len(train_all) == 2
    assert all(job["evaluation_subjects"] == [0, 1, 3] for job in train_all)
    subject_s2 = [job for job in config["jobs"] if job.get("subject") == "S2"]
    assert len(subject_s2) == 2
    assert all(job["preprocessing_variant"] == "interpolated" for job in subject_s2)


def test_correct_regression_gate_uses_cached_scores(tmp_path: Path) -> None:
    config = {
        "run_id": "run",
        "paths": {"output_root": str(tmp_path)},
        "preprocessing": {"regression_tolerance": 0.01},
        "regression_references": {"job": {"num_rows": 2, "mean_clap": 0.5}},
    }
    correct_dir = tmp_path / "run/correct/job"
    correct_dir.mkdir(parents=True)
    (correct_dir / "manifest.json").write_text('{"samples": []}', encoding="utf-8")
    legacy_dir = tmp_path / "run/legacy_correct/job"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "manifest.json").write_text('{"samples": []}', encoding="utf-8")
    score_dir = tmp_path / "run/metrics/legacy_correct_regression/job"
    score_dir.mkdir(parents=True)
    (score_dir / "per_sample_scores.json").write_text(
        json.dumps([{"overall_clap": 0.49}, {"overall_clap": 0.51}]),
        encoding="utf-8",
    )

    gates = verify_correct_regressions(config, [{"id": "job"}])

    assert gates[0]["passed"] is True


def test_curry_sidecars_support_both_naming_conventions(tmp_path: Path) -> None:
    data = tmp_path / "song7.cdt"
    data.touch()
    legacy_dpa = tmp_path / "song7.dpa"
    legacy_ceo = tmp_path / "song7.ceo"
    legacy_dpa.touch()
    legacy_ceo.touch()
    assert _fallback_curry_paths(data) == (data, legacy_dpa, legacy_ceo)

    modern_dpa = tmp_path / "song7.cdt.dpa"
    modern_ceo = tmp_path / "song7.cdt.ceo"
    modern_dpa.touch()
    modern_ceo.touch()
    assert _fallback_curry_paths(data) == (data, modern_dpa, modern_ceo)
