from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluation_analysis import holm_correction, summarize_paper_records, summarize_temporal_shift
from utils.evaluation_pairing import (
    EvaluationPairingDataset,
    PairingKey,
    assert_passive3_tensor,
    build_cross_song_mappings,
    build_temporal_shift_mapping,
    chunk_timestamp_seconds,
    common_support_target_indices,
    keys_from_index_map,
    stable_generation_seed,
    validate_cross_song_mapping,
    validate_temporal_shift_mapping,
)


class MultiSongSyntheticDataset:
    """A dataset spanning several songs per subject, for cross-song pairing tests."""

    def __init__(self, *, num_songs: int = 3, subjects: tuple[int, ...] = (0, 1), chunks_per_song: int = 20, passive3: bool = False) -> None:
        self.index_map = [
            (song, subject, chunk)
            for song in range(num_songs)
            for subject in subjects
            for chunk in range(chunks_per_song)
        ]
        self.base_eeg_channels = 2
        self.passive3 = passive3
        self.song_records = [type("Rec", (), {"name": f"song{idx}"})() for idx in range(num_songs)]

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
            "song_name": f"song{song}",
        }


# ---------------------------------------------------------------------------
# Cross-song tests
# ---------------------------------------------------------------------------


def test_cross_song_source_always_differs_from_target_song() -> None:
    dataset = MultiSongSyntheticDataset(num_songs=4, subjects=(0, 1), chunks_per_song=15)
    keys = keys_from_index_map(dataset.index_map)
    mappings, excluded = build_cross_song_mappings(keys, num_permutations=5, seed=0)
    assert not excluded
    for mapping in mappings:
        for target_index, source_index in mapping.items():
            assert keys[target_index].song_index != keys[source_index].song_index


def test_cross_song_participant_unchanged() -> None:
    dataset = MultiSongSyntheticDataset(num_songs=3, subjects=(0, 1, 2), chunks_per_song=10)
    keys = keys_from_index_map(dataset.index_map)
    mappings, _ = build_cross_song_mappings(keys, num_permutations=3, seed=1)
    for mapping in mappings:
        for target_index, source_index in mapping.items():
            assert keys[target_index].subject_index == keys[source_index].subject_index


def test_cross_song_mapping_is_deterministic_for_fixed_seed() -> None:
    dataset = MultiSongSyntheticDataset(num_songs=3, subjects=(0, 1), chunks_per_song=12)
    keys = keys_from_index_map(dataset.index_map)
    mappings_a, excluded_a = build_cross_song_mappings(keys, num_permutations=4, seed=7)
    mappings_b, excluded_b = build_cross_song_mappings(keys, num_permutations=4, seed=7)
    assert mappings_a == mappings_b
    assert excluded_a == excluded_b


def test_cross_song_invalid_targets_are_reported_not_silently_paired() -> None:
    # Subject 0 has two songs (eligible); subject 1 has only one song (must be excluded).
    keys = [
        PairingKey(dataset_index=0, song_index=0, subject_index=0, chunk_index=0),
        PairingKey(dataset_index=1, song_index=1, subject_index=0, chunk_index=0),
        PairingKey(dataset_index=2, song_index=0, subject_index=1, chunk_index=0),
    ]
    mappings, excluded = build_cross_song_mappings(keys, num_permutations=2, seed=0)
    assert len(excluded) == 1
    assert excluded[0]["target_dataset_index"] == 2
    assert excluded[0]["reason"] == "no_other_song_for_subject"
    for mapping in mappings:
        assert 2 not in mapping
        assert 0 in mapping and 1 in mapping


def test_cross_song_restricts_targets_to_requested_song_indices() -> None:
    dataset = MultiSongSyntheticDataset(num_songs=3, subjects=(0,), chunks_per_song=10)
    keys = keys_from_index_map(dataset.index_map)
    mappings, excluded = build_cross_song_mappings(keys, num_permutations=2, seed=0, target_song_indices={1})
    assert not excluded
    for mapping in mappings:
        assert all(keys[target].song_index == 1 for target in mapping)
        assert len(mapping) == 10


def test_cross_song_multicond_uses_coherent_source_song() -> None:
    dataset = MultiSongSyntheticDataset(num_songs=3, subjects=(0,), chunks_per_song=10, passive3=False)
    keys = keys_from_index_map(dataset.index_map)
    mapping = build_cross_song_mappings(keys, num_permutations=1, seed=2)[0][0]
    paired = EvaluationPairingDataset(
        dataset,
        mapping,
        mode="cross_song",
        condition_sources=["guitar", "vocal", "drum"],
        base_eeg_channels=2,
    )
    target_index = paired.target_indices[0]
    source_index = mapping[target_index]
    # The whole [guitar,vocal,drum] EEG tensor is swapped from one source row, so
    # all three attention components come from the same source song/chunk.
    assert torch.equal(paired[0]["eeg"], dataset[source_index]["eeg"])
    assert torch.equal(paired[0]["audio"], dataset[target_index]["audio"])


def test_cross_song_passive3_replication_preserved() -> None:
    dataset = MultiSongSyntheticDataset(num_songs=3, subjects=(0,), chunks_per_song=10, passive3=True)
    keys = keys_from_index_map(dataset.index_map)
    mapping = build_cross_song_mappings(keys, num_permutations=1, seed=5)[0][0]
    paired = EvaluationPairingDataset(
        dataset,
        mapping,
        mode="cross_song",
        condition_sources=["passive", "passive", "passive"],
        base_eeg_channels=2,
    )
    assert_passive3_tensor(paired[0]["eeg"], base_channels=2)


def test_cross_song_shares_generation_seed_with_correct() -> None:
    seed_correct = stable_generation_seed(base_seed=42, song_id="song7", chunk_index=4)
    seed_cross_song = stable_generation_seed(base_seed=42, song_id="song7", chunk_index=4)
    assert seed_correct == seed_cross_song


def test_validate_cross_song_mapping_rejects_same_song() -> None:
    keys = [
        PairingKey(dataset_index=0, song_index=0, subject_index=0, chunk_index=0),
        PairingKey(dataset_index=1, song_index=0, subject_index=0, chunk_index=1),
    ]
    with pytest.raises(AssertionError, match="different song"):
        validate_cross_song_mapping(keys, {0: 1})


# ---------------------------------------------------------------------------
# Temporal-shift tests
# ---------------------------------------------------------------------------


def _single_group_keys(n_chunks: int = 20) -> list[PairingKey]:
    return [PairingKey(dataset_index=i, song_index=0, subject_index=0, chunk_index=i) for i in range(n_chunks)]


def test_temporal_shift_offset_zero_reproduces_correct_mapping() -> None:
    keys = _single_group_keys()
    mapping, excluded = build_temporal_shift_mapping(keys, offset=0)
    assert not excluded
    assert mapping == {key.dataset_index: key.dataset_index for key in keys}


def test_temporal_shift_direction_matches_documented_convention() -> None:
    keys = _single_group_keys()
    mapping, _ = build_temporal_shift_mapping(keys, offset=3)
    # source_eeg_index = target_chunk_index + offset
    assert mapping[5] == 8
    mapping_neg, _ = build_temporal_shift_mapping(keys, offset=-3)
    assert mapping_neg[8] == 5


def test_temporal_shift_never_crosses_song_boundary() -> None:
    keys = _single_group_keys(10) + [
        PairingKey(dataset_index=10 + i, song_index=1, subject_index=0, chunk_index=i) for i in range(10)
    ]
    mapping, excluded = build_temporal_shift_mapping(keys, offset=2)
    by_index = {key.dataset_index: key for key in keys}
    for target_index, source_index in mapping.items():
        assert by_index[target_index].song_index == by_index[source_index].song_index
    validate_temporal_shift_mapping(keys, mapping, offset=2)


def test_temporal_shift_excludes_boundary_chunks() -> None:
    keys = _single_group_keys(10)
    mapping, excluded = build_temporal_shift_mapping(keys, offset=-5)
    excluded_targets = {row["target_dataset_index"] for row in excluded}
    assert excluded_targets == {0, 1, 2, 3, 4}
    assert set(mapping) == {5, 6, 7, 8, 9}
    for row in excluded:
        assert row["reason"] == "boundary"


def test_temporal_shift_common_support_uses_identical_targets_for_every_offset() -> None:
    keys = _single_group_keys(20)
    offsets = [-5, -2, -1, 0, 1, 2, 5]
    mappings_by_offset = {offset: build_temporal_shift_mapping(keys, offset=offset)[0] for offset in offsets}
    common = common_support_target_indices(mappings_by_offset)
    # Chunks 5..14 support every offset in [-5, 5] within a 20-chunk song.
    assert common == set(range(5, 15))
    for offset, mapping in mappings_by_offset.items():
        assert common.issubset(set(mapping))


def test_temporal_shift_multicond_components_receive_same_offset() -> None:
    dataset = MultiSongSyntheticDataset(num_songs=1, subjects=(0,), chunks_per_song=20, passive3=False)
    keys = keys_from_index_map(dataset.index_map)
    mapping, _ = build_temporal_shift_mapping(keys, offset=3)
    paired = EvaluationPairingDataset(
        dataset,
        mapping,
        mode="temporal_shift",
        condition_sources=["guitar", "vocal", "drum"],
        base_eeg_channels=2,
        temporal_offset=3,
    )
    target_index = paired.target_indices[0]
    source_index = mapping[target_index]
    assert torch.equal(paired[0]["eeg"], dataset[source_index]["eeg"])
    assert int(dataset[source_index]["chunk_idx"]) - int(dataset[target_index]["chunk_idx"]) == 3


def test_temporal_shift_passive3_uses_shifted_chunk_and_replication() -> None:
    dataset = MultiSongSyntheticDataset(num_songs=1, subjects=(0,), chunks_per_song=20, passive3=True)
    keys = keys_from_index_map(dataset.index_map)
    mapping, _ = build_temporal_shift_mapping(keys, offset=-2)
    paired = EvaluationPairingDataset(
        dataset,
        mapping,
        mode="temporal_shift",
        condition_sources=["passive", "passive", "passive"],
        base_eeg_channels=2,
        temporal_offset=-2,
    )
    assert_passive3_tensor(paired[0]["eeg"], base_channels=2)


def test_temporal_shift_rejects_non_contiguous_chunk_ordering() -> None:
    keys = [
        PairingKey(dataset_index=0, song_index=0, subject_index=0, chunk_index=0),
        PairingKey(dataset_index=1, song_index=0, subject_index=0, chunk_index=2),  # gap: missing chunk 1
    ]
    with pytest.raises(ValueError, match="Non-contiguous"):
        build_temporal_shift_mapping(keys, offset=1)


def test_temporal_shift_shares_generation_seed_with_correct() -> None:
    seed_a = stable_generation_seed(base_seed=42, song_id="song7", chunk_index=10)
    seed_b = stable_generation_seed(base_seed=42, song_id="song7", chunk_index=10)
    assert seed_a == seed_b


def test_chunk_timestamp_seconds_uses_verified_stride() -> None:
    assert chunk_timestamp_seconds(0, chunk_sec=3.5) == 0.0
    assert chunk_timestamp_seconds(4, chunk_sec=3.5) == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------------


def test_holm_correction_matches_manual_step_down() -> None:
    adjusted = holm_correction({1: 0.01, 2: 0.02, 3: 0.5})
    assert adjusted[1] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.04)
    assert adjusted[3] == pytest.approx(0.5)


def test_holm_correction_is_monotone_non_decreasing_by_rank() -> None:
    adjusted = holm_correction({1: 0.2, 2: 0.01, 3: 0.03})
    ordered = sorted(adjusted.items(), key=lambda kv: {1: 0.2, 2: 0.01, 3: 0.03}[kv[0]])
    values = [value for _, value in ordered]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# Analysis-layer integration: cross_song in summarize_paper_records
# ---------------------------------------------------------------------------


def _record(mode: str, score: float, chunk: int, *, permutation: int | None = None, model: str = "MULTICOND"):
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
        "eeg_source_chunk_id": chunk if mode != "cross_song" else chunk + 100,
        "eeg_source_song_id": "song3" if mode == "cross_song" else ("song7" if mode != "pretrained" else None),
        "eeg_source_subject_id": 0 if mode != "pretrained" else None,
        "generation_seed": chunk + 100,
        "overall_clap": score,
        "drum_clap": score,
        "guitar_proxy_other_bass_clap": score,
        "vocal_clap": score,
    }


def test_summary_computes_cross_song_contrasts() -> None:
    records = []
    for chunk in (0, 1):
        records.append(_record("correct", 0.7, chunk))
        records.append(_record("correct", 0.6, chunk, model="PASSIVE3"))
        records.append(_record("cross_song", 0.5, chunk, permutation=0))
        records.append(_record("cross_song", 0.55, chunk, permutation=1))
        records.append(_record("cross_song", 0.4, chunk, permutation=0, model="PASSIVE3"))
        records.append(_record("cross_song", 0.42, chunk, permutation=1, model="PASSIVE3"))
        pretrained = _record("pretrained", 0.3, chunk)
        pretrained.update({"job_id": "pretrained_prior", "model_type": "AudioLDM2", "target_subject_id": None})
        records.append(pretrained)

    summary = summarize_paper_records(records, num_resamples=200, seed=0)
    multi = next(row for row in summary["jobs"] if row["job_id"] == "multi")
    passive = next(row for row in summary["jobs"] if row["job_id"] == "passive")

    assert multi["cross_song_mean"] == pytest.approx(0.525)
    assert multi["correct_minus_cross_song"]["mean_difference"] == pytest.approx(0.7 - 0.525)
    assert multi["cross_song_minus_pretrained"]["mean_difference"] == pytest.approx(0.525 - 0.3)
    assert passive["cross_song_mean"] == pytest.approx(0.41)

    cross_gain = summary["protocol_gain_multicond_minus_passive3_cross_song"][0]
    assert cross_gain["mean_difference"] == pytest.approx(0.525 - 0.41)


def test_summary_cross_song_missing_targets_do_not_raise() -> None:
    # Only chunk 0 has a cross_song result (e.g. chunk 1's subject had no eligible source song).
    records = [
        _record("correct", 0.7, 0),
        _record("correct", 0.7, 1),
        _record("cross_song", 0.5, 0, permutation=0),
    ]
    summary = summarize_paper_records(records, num_resamples=50, seed=0)
    multi = summary["jobs"][0]
    assert multi["num_cross_song_targets"] == 1
    assert multi["correct_minus_cross_song"]["n_pairs"] == 1


# ---------------------------------------------------------------------------
# Analysis-layer integration: summarize_temporal_shift
# ---------------------------------------------------------------------------


def _temporal_record(mode: str, score: float, chunk: int, *, offset: int | None = None):
    return {
        "evaluation_mode": mode,
        "job_id": "multi",
        "model_type": "MULTICOND",
        "target_chunk_id": chunk,
        "target_song_id": "song7",
        "target_subject_id": 0,
        "eeg_source_chunk_id": chunk if offset is None else chunk + offset,
        "temporal_offset_chunks": offset,
        "permutation_id": 0 if mode == "shuffled" else None,
        "generation_seed": 100 + chunk,
        "overall_clap": score,
    }


def test_summarize_temporal_shift_common_support_and_direction() -> None:
    records = [_temporal_record("correct", 1.0, chunk) for chunk in range(10)]
    for offset, valid_chunks, score in [
        (-2, range(2, 10), 0.90),
        (-1, range(1, 10), 0.95),
        (1, range(0, 9), 0.93),
        (2, range(0, 8), 0.85),
    ]:
        for chunk in valid_chunks:
            records.append(_temporal_record("temporal_shift", score, chunk, offset=offset))

    summary = summarize_temporal_shift(
        records,
        offsets=[-2, -1, 0, 1, 2],
        chunk_sec=3.5,
        include_random_within_song=False,
        analysis_mode="common_support",
        num_resamples=200,
        seed=0,
    )
    job = summary["jobs"][0]
    # Intersection of {2..9}, {1..9}, {0..9}, {0..8}, {0..7} == {2..7}.
    assert job["common_support_n"] == 6
    by_offset = {row["offset_chunks"]: row for row in job["offsets"]}
    assert by_offset[0]["mean_clap"] == pytest.approx(1.0)
    assert by_offset[0]["holm_adjusted_p_value"] == 0.0
    assert by_offset[-2]["n_max_available"] == 8
    assert by_offset[-2]["n_common_support"] == 6
    contrast = by_offset[-2]["correct_minus_shift_common_support"]
    assert contrast["mean_difference"] == pytest.approx(1.0 - 0.90)
    for offset in (-2, -1, 1, 2):
        raw_contrast = by_offset[offset]["correct_minus_shift_common_support"]
        assert raw_contrast is not None
        assert by_offset[offset]["holm_adjusted_p_value"] >= raw_contrast["p_value"]

    # 10 correct rows + valid nonzero-offset rows.
    assert len(summary["tidy_rows"]) == 10 + 8 + 9 + 9 + 8
    shifted_row = next(
        row for row in summary["tidy_rows"] if row["offset_chunks"] == -2 and row["target_chunk_id"] == 5
    )
    assert shifted_row["eeg_source_chunk_id"] == 3


def test_summarize_temporal_shift_reuses_shuffled_as_random_within_song() -> None:
    records = [_temporal_record("correct", 1.0, chunk) for chunk in range(5)]
    for chunk in range(5):
        records.append({**_temporal_record("shuffled", 0.6, chunk), "permutation_id": 0})
        records.append({**_temporal_record("shuffled", 0.7, chunk), "permutation_id": 1})
    summary = summarize_temporal_shift(
        records,
        offsets=[0],
        chunk_sec=3.5,
        include_random_within_song=True,
        analysis_mode="common_support",
        num_resamples=100,
        seed=0,
    )
    job = summary["jobs"][0]
    assert job["random_within_song"]["n"] == 5
    assert job["random_within_song"]["mean_clap"] == pytest.approx(0.65)
    labels = {row["condition_label"] for row in summary["tidy_rows"]}
    assert "random_within_song" in labels


# ---------------------------------------------------------------------------
# Regression: existing config keys remain valid and new keys are well-formed
# ---------------------------------------------------------------------------


def test_paper_controls_config_has_valid_cross_song_and_temporal_shift_blocks() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/evaluate_paper_controls.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(config["jobs"]) == 16  # unchanged from the existing registry

    cross_song = config["cross_song"]
    assert isinstance(cross_song["num_permutations"], int) and cross_song["num_permutations"] > 0
    assert isinstance(cross_song["mapping_seed"], int)

    temporal_shift = config["temporal_shift"]
    offsets = temporal_shift["offsets"]
    assert 0 in offsets
    assert offsets == sorted(offsets)
    assert temporal_shift["analysis_mode"] in {"common_support", "max_available"}
