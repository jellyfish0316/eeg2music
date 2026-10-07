from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairingKey:
    dataset_index: int
    song_index: int
    subject_index: int
    chunk_index: int


def keys_from_index_map(index_map: Sequence[tuple[int, int, int]]) -> list[PairingKey]:
    return [
        PairingKey(
            dataset_index=index,
            song_index=int(song_index),
            subject_index=int(subject_index),
            chunk_index=int(chunk_index),
        )
        for index, (song_index, subject_index, chunk_index) in enumerate(index_map)
    ]


def identity_mapping(keys: Sequence[PairingKey]) -> dict[int, int]:
    return {key.dataset_index: key.dataset_index for key in keys}


def _valid_cyclic_shifts(group: Sequence[PairingKey], min_chunk_distance: int) -> list[int]:
    ordered = sorted(group, key=lambda item: (item.chunk_index, item.dataset_index))
    valid: list[int] = []
    for shift in range(1, len(ordered)):
        if all(
            abs(target.chunk_index - ordered[(position + shift) % len(ordered)].chunk_index)
            >= min_chunk_distance
            for position, target in enumerate(ordered)
        ):
            valid.append(shift)
    return valid


def build_cyclic_derangements(
    keys: Sequence[PairingKey],
    *,
    num_permutations: int,
    seed: int,
    min_chunk_distance: int,
) -> list[dict[int, int]]:
    """Build deterministic, unique, same-subject/song constrained derangements.

    Each subject/song group is independently rotated by a randomly ordered legal
    cyclic shift. Requiring a different shift for each permutation guarantees
    that the full mappings are different without retry-based fallbacks.
    """

    if num_permutations <= 0:
        raise ValueError("num_permutations must be positive.")
    if min_chunk_distance < 1:
        raise ValueError("min_chunk_distance must be at least 1.")
    if not keys:
        raise ValueError("Cannot shuffle an empty evaluation set.")

    groups: dict[tuple[int, int], list[PairingKey]] = defaultdict(list)
    for key in keys:
        groups[(key.subject_index, key.song_index)].append(key)

    rng = np.random.default_rng(int(seed))
    shifts_by_group: dict[tuple[int, int], list[int]] = {}
    ordered_by_group: dict[tuple[int, int], list[PairingKey]] = {}
    for group_id in sorted(groups):
        ordered = sorted(groups[group_id], key=lambda item: (item.chunk_index, item.dataset_index))
        valid = _valid_cyclic_shifts(ordered, min_chunk_distance)
        if len(valid) < num_permutations:
            raise ValueError(
                "Not enough distinct legal cyclic derangements for "
                f"subject={group_id[0]} song={group_id[1]}: "
                f"chunks={len(ordered)} min_chunk_distance={min_chunk_distance} "
                f"legal_shifts={len(valid)} requested={num_permutations}."
            )
        shifts_by_group[group_id] = [int(value) for value in rng.permutation(valid)[:num_permutations]]
        ordered_by_group[group_id] = ordered

    mappings: list[dict[int, int]] = []
    for permutation_id in range(num_permutations):
        mapping: dict[int, int] = {}
        for group_id in sorted(ordered_by_group):
            ordered = ordered_by_group[group_id]
            shift = shifts_by_group[group_id][permutation_id]
            for position, target in enumerate(ordered):
                source = ordered[(position + shift) % len(ordered)]
                mapping[target.dataset_index] = source.dataset_index
        validate_mapping(
            keys,
            mapping,
            min_chunk_distance=min_chunk_distance,
            allow_identity=False,
        )
        mappings.append(mapping)

    signatures = {tuple(sorted(mapping.items())) for mapping in mappings}
    if len(signatures) != len(mappings):
        raise AssertionError("Generated shuffle mappings are not unique.")
    return mappings


def chunk_timestamp_seconds(chunk_index: int, *, chunk_sec: float) -> float:
    """Convert a local chunk index to a start timestamp in seconds.

    Chunks are sliced back-to-back with no overlap (see
    ConditionNMEDTDataset._build_song_records / __getitem__: absolute_chunk_idx *
    audio_chunk_len), so the true stride equals chunk_sec exactly. This is a
    verified property of this dataset, not an assumption.
    """

    return float(chunk_index) * float(chunk_sec)


def validate_mapping(
    keys: Sequence[PairingKey],
    mapping: Mapping[int, int],
    *,
    min_chunk_distance: int,
    allow_identity: bool,
) -> None:
    by_index = {key.dataset_index: key for key in keys}
    expected = set(by_index)
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise AssertionError(f"Incomplete mapping: missing={missing} extra={extra}")
    if set(mapping.values()) != expected:
        raise AssertionError("Mapping sources must form a permutation of the evaluation rows.")

    for target_index, source_index in mapping.items():
        target = by_index[target_index]
        source = by_index[source_index]
        if not allow_identity and target_index == source_index:
            raise AssertionError(f"Fixed point at dataset index {target_index}.")
        if target.subject_index != source.subject_index:
            raise AssertionError(
                f"Cross-subject mapping: target={target.subject_index} source={source.subject_index}."
            )
        if target.song_index != source.song_index:
            raise AssertionError(f"Cross-song mapping: target={target.song_index} source={source.song_index}.")
        if not allow_identity and abs(target.chunk_index - source.chunk_index) < min_chunk_distance:
            raise AssertionError(
                f"Chunk distance violation: target={target.chunk_index} source={source.chunk_index} "
                f"minimum={min_chunk_distance}."
            )


def validate_cross_song_mapping(keys: Sequence[PairingKey], mapping: Mapping[int, int]) -> None:
    """Validate a cross-song mapping: same participant, source song differs from target song.

    Unlike within-song shuffled mappings, cross-song mappings are not required to
    form a permutation of the evaluation set: a target may be excluded (no
    eligible source song for its subject) and a source chunk may be reused by
    more than one target.
    """

    by_index = {key.dataset_index: key for key in keys}
    unknown_targets = sorted(set(mapping) - set(by_index))
    if unknown_targets:
        raise AssertionError(f"Cross-song mapping references unknown target indices: {unknown_targets}")
    for target_index, source_index in mapping.items():
        target = by_index[target_index]
        source = by_index.get(int(source_index))
        if source is None:
            raise AssertionError(f"Cross-song mapping references unknown source index {source_index}.")
        if target.subject_index != source.subject_index:
            raise AssertionError(
                f"Cross-subject mapping: target={target.subject_index} source={source.subject_index}."
            )
        if target.song_index == source.song_index:
            raise AssertionError(
                f"Cross-song mapping requires a different song: target={target.song_index} "
                f"source={source.song_index} (dataset index {target_index})."
            )


def build_cross_song_mappings(
    keys: Sequence[PairingKey],
    *,
    num_permutations: int,
    seed: int,
    target_song_indices: set[int] | None = None,
) -> tuple[list[dict[int, int]], list[dict[str, Any]]]:
    """Build deterministic same-subject, different-song EEG source mappings.

    For every (subject, song) target group, eligible sources are drawn from all
    other songs belonging to the same subject. Source songs are cycled in a
    permutation-specific shuffled order so that target chunks are distributed
    across eligible source songs rather than concentrated on one song, and
    source chunks within a song are drawn from a shuffled pool so that a single
    source chunk is not reused before the pool cycles. Targets whose subject has
    no other song are reported as excluded rather than silently paired.

    `target_song_indices` restricts which songs are treated as evaluation
    targets (e.g. only the canonical ood_test song); every song still counts as
    an eligible EEG source for other songs' targets. `None` treats every song
    present in `keys` as a target, which is convenient for tests over a single
    synthetic multi-song pool.
    """

    if num_permutations <= 0:
        raise ValueError("num_permutations must be positive.")
    if not keys:
        raise ValueError("Cannot build cross-song mappings for an empty evaluation set.")

    groups: dict[tuple[int, int], list[PairingKey]] = defaultdict(list)
    songs_by_subject: dict[int, set[int]] = defaultdict(set)
    for key in keys:
        groups[(key.subject_index, key.song_index)].append(key)
        songs_by_subject[key.subject_index].add(key.song_index)

    target_group_ids = [
        group_id
        for group_id in sorted(groups)
        if target_song_indices is None or group_id[1] in target_song_indices
    ]

    excluded: list[dict[str, Any]] = []
    eligible_groups: dict[tuple[int, int], tuple[list[PairingKey], list[int]]] = {}
    for group_id in target_group_ids:
        subject_index, song_index = group_id
        ordered = sorted(groups[group_id], key=lambda item: (item.chunk_index, item.dataset_index))
        other_songs = sorted(songs_by_subject[subject_index] - {song_index})
        if not other_songs:
            for target in ordered:
                excluded.append(
                    {
                        "target_dataset_index": target.dataset_index,
                        "target_subject_index": target.subject_index,
                        "target_song_index": target.song_index,
                        "target_chunk_index": target.chunk_index,
                        "reason": "no_other_song_for_subject",
                    }
                )
            continue
        eligible_groups[group_id] = (ordered, other_songs)

    rng = np.random.default_rng(int(seed))
    mappings: list[dict[int, int]] = []
    for _ in range(int(num_permutations)):
        mapping: dict[int, int] = {}
        for group_id in sorted(eligible_groups):
            ordered_targets, other_songs = eligible_groups[group_id]
            song_cycle = [int(value) for value in rng.permutation(other_songs)]
            source_pools: dict[int, list[PairingKey]] = {}
            pool_cursors: dict[int, int] = {}
            for song_id in song_cycle:
                pool = sorted(groups[(group_id[0], song_id)], key=lambda item: (item.chunk_index, item.dataset_index))
                permuted = [pool[index] for index in rng.permutation(len(pool))]
                source_pools[song_id] = permuted
                pool_cursors[song_id] = 0
            for position, target in enumerate(ordered_targets):
                song_choice = song_cycle[position % len(song_cycle)]
                pool = source_pools[song_choice]
                cursor = pool_cursors[song_choice]
                source = pool[cursor % len(pool)]
                pool_cursors[song_choice] = cursor + 1
                mapping[target.dataset_index] = source.dataset_index
        validate_cross_song_mapping(keys, mapping)
        mappings.append(mapping)

    return mappings, excluded


def validate_temporal_shift_mapping(
    keys: Sequence[PairingKey],
    mapping: Mapping[int, int],
    *,
    offset: int,
) -> None:
    """Validate a temporal-shift mapping: same participant/song, exact chunk offset."""

    by_index = {key.dataset_index: key for key in keys}
    unknown_targets = sorted(set(mapping) - set(by_index))
    if unknown_targets:
        raise AssertionError(f"Temporal-shift mapping references unknown target indices: {unknown_targets}")
    for target_index, source_index in mapping.items():
        target = by_index[target_index]
        source = by_index.get(int(source_index))
        if source is None:
            raise AssertionError(f"Temporal-shift mapping references unknown source index {source_index}.")
        if target.subject_index != source.subject_index:
            raise AssertionError(
                f"Cross-subject temporal shift: target={target.subject_index} source={source.subject_index}."
            )
        if target.song_index != source.song_index:
            raise AssertionError(
                f"Temporal shift crossed a song boundary: target={target.song_index} source={source.song_index}."
            )
        if int(source.chunk_index) - int(target.chunk_index) != int(offset):
            raise AssertionError(
                f"Temporal shift offset mismatch at dataset index {target_index}: expected {offset}, "
                f"got {int(source.chunk_index) - int(target.chunk_index)}."
            )


def build_temporal_shift_mapping(
    keys: Sequence[PairingKey],
    *,
    offset: int,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Build a same-subject, same-song mapping shifted by a fixed signed chunk offset.

    source_eeg_index = target_chunk_index + offset. Targets whose shifted chunk
    would fall outside their song's valid chunk range are excluded (no
    wraparound across song boundaries and no borrowing from another song).
    Also validates that chunk indices within each (subject, song) group are
    unique and contiguous (0..n-1) before trusting array position as
    chronological order.
    """

    if not keys:
        raise ValueError("Cannot build a temporal-shift mapping for an empty evaluation set.")

    groups: dict[tuple[int, int], list[PairingKey]] = defaultdict(list)
    for key in keys:
        groups[(key.subject_index, key.song_index)].append(key)

    mapping: dict[int, int] = {}
    excluded: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        ordered = sorted(groups[group_id], key=lambda item: (item.chunk_index, item.dataset_index))
        chunk_indices = [item.chunk_index for item in ordered]
        if len(set(chunk_indices)) != len(chunk_indices):
            raise ValueError(f"Duplicate chunk indices for subject={group_id[0]} song={group_id[1]}: {chunk_indices}")
        if chunk_indices != list(range(len(chunk_indices))):
            raise ValueError(
                f"Non-contiguous chunk indices for subject={group_id[0]} song={group_id[1]}: {chunk_indices}. "
                "Chronological ordering could not be established from array position."
            )
        by_chunk = {item.chunk_index: item for item in ordered}
        for target in ordered:
            source_chunk = int(target.chunk_index) + int(offset)
            if source_chunk not in by_chunk:
                excluded.append(
                    {
                        "target_dataset_index": target.dataset_index,
                        "target_subject_index": target.subject_index,
                        "target_song_index": target.song_index,
                        "target_chunk_index": target.chunk_index,
                        "offset": int(offset),
                        "reason": "boundary",
                    }
                )
                continue
            mapping[target.dataset_index] = by_chunk[source_chunk].dataset_index

    validate_temporal_shift_mapping(keys, mapping, offset=offset)
    return mapping, excluded


def common_support_target_indices(mappings_by_offset: Mapping[int, Mapping[int, int]]) -> set[int]:
    """Intersect valid target dataset indices across every requested offset."""

    sets = [set(mapping) for mapping in mappings_by_offset.values()]
    if not sets:
        return set()
    common = set(sets[0])
    for other in sets[1:]:
        common &= other
    return common


def restrict_mapping(mapping: Mapping[int, int], target_indices: set[int]) -> dict[int, int]:
    return {target: source for target, source in mapping.items() if target in target_indices}


def assert_passive3_tensor(eeg: torch.Tensor, *, base_channels: int) -> None:
    if eeg.ndim != 2:
        raise AssertionError(f"PASSIVE3 EEG must have shape [C,T], got {tuple(eeg.shape)}")
    if int(eeg.shape[0]) != int(base_channels) * 3:
        raise AssertionError(
            f"PASSIVE3 expected {base_channels * 3} channels, got {int(eeg.shape[0])}."
        )
    branches = eeg.reshape(3, int(base_channels), int(eeg.shape[1]))
    if not torch.equal(branches[0], branches[1]) or not torch.equal(branches[0], branches[2]):
        raise AssertionError("PASSIVE3 branches are not exact copies of the same passive chunk.")


class EvaluationPairingDataset(Dataset):
    """Read target metadata/audio from one row and EEG from a mapped source row."""

    def __init__(
        self,
        dataset: Dataset,
        mapping: Mapping[int, int],
        *,
        mode: str,
        condition_sources: Sequence[str],
        base_eeg_channels: int,
        min_chunk_distance: int = 1,
        temporal_offset: int | None = None,
    ) -> None:
        if mode not in {"correct", "shuffled", "cross_song", "temporal_shift"}:
            raise ValueError(f"Unsupported pairing mode: {mode}")
        self.dataset = dataset
        self.mapping = {int(target): int(source) for target, source in mapping.items()}
        self.mode = mode
        self.condition_sources = [str(value) for value in condition_sources]
        self.base_eeg_channels = int(base_eeg_channels)
        self.temporal_offset = None if temporal_offset is None else int(temporal_offset)

        index_map = getattr(dataset, "index_map", None)
        if index_map is None:
            raise TypeError("EvaluationPairingDataset requires a dataset.index_map.")
        keys = keys_from_index_map(index_map)
        if mode in ("correct", "shuffled"):
            validate_mapping(
                keys,
                self.mapping,
                min_chunk_distance=0 if mode == "correct" else int(min_chunk_distance),
                allow_identity=mode == "correct",
            )
            if mode == "correct" and any(target != source for target, source in self.mapping.items()):
                raise AssertionError("Correct mode must use an identity mapping.")
        elif mode == "cross_song":
            validate_cross_song_mapping(keys, self.mapping)
        else:
            if self.temporal_offset is None:
                raise ValueError("temporal_shift mode requires temporal_offset.")
            validate_temporal_shift_mapping(keys, self.mapping, offset=self.temporal_offset)

        if not self.mapping:
            raise ValueError(f"{mode} mapping has no valid targets to evaluate.")
        self.target_indices = sorted(self.mapping)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        target_index = self.target_indices[position]
        target = dict(self.dataset[target_index])
        source_index = self.mapping[target_index]
        source = self.dataset[source_index]
        target["eeg"] = source["eeg"]
        target["eeg_source_dataset_idx"] = torch.tensor(source_index, dtype=torch.long)
        target["eeg_source_song_idx"] = source["song_idx"].clone()
        target["eeg_source_subject_idx"] = source["subject_idx"].clone()
        target["eeg_source_chunk_idx"] = source["chunk_idx"].clone()
        target["target_dataset_idx"] = torch.tensor(int(target_index), dtype=torch.long)
        target["evaluation_mode"] = self.mode

        if self.condition_sources == ["passive", "passive", "passive"]:
            assert_passive3_tensor(target["eeg"], base_channels=self.base_eeg_channels)
        elif self.condition_sources != ["guitar", "vocal", "drum"]:
            raise AssertionError(
                "Paper controls support only MULTICOND [guitar,vocal,drum] or "
                "PASSIVE3 [passive,passive,passive]."
            )
        return target


def stable_generation_seed(
    *,
    base_seed: int,
    song_id: str,
    chunk_index: int,
    replicate_index: int = 0,
) -> int:
    payload = f"{int(base_seed)}|{song_id}|{int(chunk_index)}|{int(replicate_index)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def mapping_records(
    keys: Sequence[PairingKey],
    mapping: Mapping[int, int],
) -> list[dict[str, int]]:
    by_index = {key.dataset_index: key for key in keys}
    rows = []
    for target_index in sorted(mapping):
        target = by_index[target_index]
        source = by_index[mapping[target_index]]
        rows.append(
            {
                "target_dataset_index": target.dataset_index,
                "target_song_index": target.song_index,
                "target_subject_index": target.subject_index,
                "target_chunk_index": target.chunk_index,
                "source_dataset_index": source.dataset_index,
                "source_song_index": source.song_index,
                "source_subject_index": source.subject_index,
                "source_chunk_index": source.chunk_index,
            }
        )
    return rows


def unique_target_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, int, int]]:
    return {
        (str(row["target_song_id"]), int(row["target_chunk_id"]), int(row["generation_seed"]))
        for row in rows
    }
