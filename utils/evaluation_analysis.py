from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from utils.evaluation_pairing import chunk_timestamp_seconds


TARGET_KEY_FIELDS = ("target_song_id", "target_subject_id", "target_chunk_id", "generation_seed")
PRETRAINED_KEY_FIELDS = ("target_song_id", "target_chunk_id", "generation_seed")
SCORE_FIELDS = ("overall_clap", "drum_clap", "guitar_proxy_other_bass_clap", "vocal_clap")


def _target_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in TARGET_KEY_FIELDS)


def _pretrained_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in PRETRAINED_KEY_FIELDS)


def _pretrained_key_from_target_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    # TARGET_KEY_FIELDS = (song, subject, chunk, seed); PRETRAINED_KEY_FIELDS = (song, chunk, seed).
    return (key[0], key[2], key[3])


def _permutation_target_means(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[Any, ...], list[float]], dict[int, list[float]]]:
    """Group per-permutation scores by target key and by permutation id."""

    target_scores: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    permutation_scores: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = _available_score(row, "overall_clap")
        if value is None:
            continue
        target_scores[_target_key(row)].append(value)
        permutation_scores[int(row["permutation_id"])].append(value)
    return target_scores, permutation_scores


def paired_bootstrap(
    differences: Sequence[float],
    *,
    num_resamples: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired_bootstrap requires at least one finite difference.")
    if not np.isfinite(values).all():
        raise ValueError("paired_bootstrap received non-finite differences.")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive.")

    rng = np.random.default_rng(int(seed))
    observed = float(values.mean())
    sample_indices = rng.integers(0, values.size, size=(int(num_resamples), values.size))
    bootstrap_means = values[sample_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

    centered = values - observed
    null_indices = rng.integers(0, values.size, size=(int(num_resamples), values.size))
    null_means = centered[null_indices].mean(axis=1)
    p_value = (1 + int(np.count_nonzero(np.abs(null_means) >= abs(observed)))) / (int(num_resamples) + 1)
    return {
        "n_pairs": int(values.size),
        "mean_difference": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": float(p_value),
        "bootstrap_resamples": int(num_resamples),
        "bootstrap_seed": int(seed),
    }


def hierarchical_cluster_bootstrap(
    differences_by_cluster: Mapping[Any, Sequence[float]],
    *,
    num_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap subjects first and chunks second for population-level inference."""

    clusters = {
        str(cluster): np.asarray(values, dtype=np.float64)
        for cluster, values in differences_by_cluster.items()
        if len(values) > 0
    }
    if not clusters:
        raise ValueError("hierarchical_cluster_bootstrap requires at least one non-empty cluster.")
    if any(values.ndim != 1 or not np.isfinite(values).all() for values in clusters.values()):
        raise ValueError("hierarchical_cluster_bootstrap received invalid differences.")
    cluster_ids = sorted(clusters)
    cluster_means = {cluster: float(clusters[cluster].mean()) for cluster in cluster_ids}
    observed = float(np.mean(list(cluster_means.values())))
    result: dict[str, Any] = {
        "n_clusters": len(cluster_ids),
        "cluster_field": "target_subject_id",
        "cluster_means": cluster_means,
        "mean_difference": observed,
        "bootstrap_resamples": int(num_resamples),
        "bootstrap_seed": int(seed),
        "inference_scope": "subjects_resampled_then_chunks_within_subject",
    }
    if len(cluster_ids) < 2:
        result.update(
            {
                "ci95_low": None,
                "ci95_high": None,
                "p_value": None,
                "warning": "Fewer than two subjects; population-level cluster inference is not available.",
            }
        )
        return result

    rng = np.random.default_rng(int(seed))

    def resample_means(center: bool) -> np.ndarray:
        sampled_means = np.empty(int(num_resamples), dtype=np.float64)
        for resample_index in range(int(num_resamples)):
            sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            within_means = []
            for cluster in sampled_clusters:
                values = clusters[str(cluster)]
                if center:
                    values = values - observed
                indices = rng.integers(0, len(values), size=len(values))
                within_means.append(float(values[indices].mean()))
            sampled_means[resample_index] = float(np.mean(within_means))
        return sampled_means

    bootstrap_means = resample_means(center=False)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    null_means = resample_means(center=True)
    p_value = (1 + int(np.count_nonzero(np.abs(null_means) >= abs(observed)))) / (int(num_resamples) + 1)
    result.update(
        {
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "p_value": float(p_value),
        }
    )
    return result


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return float(sum(materialized) / len(materialized)) if materialized else float("nan")


def _mean_or_none(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return float(sum(materialized) / len(materialized)) if materialized else None


def _available_score(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def summarize_metric(
    correct_rows: Sequence[Mapping[str, Any]],
    shuffled_rows: Sequence[Mapping[str, Any]],
    pretrained_by_key: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    field: str,
    num_resamples: int,
    seed: int,
) -> dict[str, Any]:
    correct_by_key = {
        _target_key(row): row for row in correct_rows if _available_score(row, field) is not None
    }
    shuffled_scores: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    permutation_scores: dict[int, list[float]] = defaultdict(list)
    for row in shuffled_rows:
        value = _available_score(row, field)
        if value is None:
            continue
        shuffled_scores[_target_key(row)].append(value)
        permutation_scores[int(row["permutation_id"])].append(value)

    shuffled_target_means = {key: _mean(values) for key, values in shuffled_scores.items()}
    correct_shuffled = [
        float(correct_by_key[key][field]) - shuffled_target_means[key]
        for key in correct_by_key
        if key in shuffled_target_means
    ]
    pretrained_values: list[float] = []
    correct_pretrained: list[float] = []
    correct_pretrained_by_subject: dict[Any, list[float]] = defaultdict(list)
    for row in correct_by_key.values():
        baseline = pretrained_by_key.get(_pretrained_key(row))
        if baseline is None:
            continue
        baseline_value = _available_score(baseline, field)
        if baseline_value is None:
            continue
        pretrained_values.append(baseline_value)
        difference = float(row[field]) - baseline_value
        correct_pretrained.append(difference)
        correct_pretrained_by_subject[row.get("target_subject_id")].append(difference)
    permutation_means = [_mean(values) for _, values in sorted(permutation_scores.items())]
    correct_values = [float(row[field]) for row in correct_by_key.values()]
    return {
        "num_correct_rows": len(correct_values),
        "num_shuffle_rows": sum(len(values) for values in permutation_scores.values()),
        "correct_mean": _mean_or_none(correct_values),
        "shuffled_mean": _mean_or_none(permutation_means),
        "shuffled_sd": float(statistics.stdev(permutation_means)) if len(permutation_means) > 1 else 0.0,
        "pretrained_mean": _mean_or_none(pretrained_values),
        "correct_minus_shuffled": paired_bootstrap(
            correct_shuffled, num_resamples=num_resamples, seed=seed
        )
        if correct_shuffled
        else None,
        "correct_minus_pretrained": paired_bootstrap(
            correct_pretrained, num_resamples=num_resamples, seed=seed + 1
        )
        if correct_pretrained
        else None,
        "correct_minus_pretrained_cluster_bootstrap": hierarchical_cluster_bootstrap(
            correct_pretrained_by_subject,
            num_resamples=num_resamples,
            seed=seed + 2,
        )
        if correct_pretrained_by_subject
        else None,
    }


def summarize_paper_records(
    records: Sequence[Mapping[str, Any]],
    *,
    num_resamples: int,
    seed: int,
) -> dict[str, Any]:
    correct_by_job: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    shuffled_by_job: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cross_song_by_job: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pretrained = []
    for row in records:
        mode = str(row["evaluation_mode"])
        if mode == "correct":
            correct_by_job[str(row["job_id"])].append(row)
        elif mode == "shuffled":
            shuffled_by_job[str(row["job_id"])].append(row)
        elif mode == "cross_song":
            cross_song_by_job[str(row["job_id"])].append(row)
        elif mode == "pretrained":
            pretrained.append(row)
        else:
            raise ValueError(f"Unknown evaluation_mode={mode!r}")

    pretrained_by_key = {_pretrained_key(row): row for row in pretrained}
    cross_song_target_means_by_job: dict[str, dict[tuple[Any, ...], float]] = {}
    summaries: list[dict[str, Any]] = []
    for job_id in sorted(correct_by_job):
        correct_rows = correct_by_job[job_id]
        shuffled_rows = shuffled_by_job.get(job_id, [])
        cross_song_rows = cross_song_by_job.get(job_id, [])
        correct_by_key = {_target_key(row): row for row in correct_rows}
        shuffled_scores, permutation_scores = _permutation_target_means(shuffled_rows)
        cross_song_scores, cross_song_permutation_scores = _permutation_target_means(cross_song_rows)

        missing = sorted(set(correct_by_key) - set(shuffled_scores), key=str)
        if shuffled_rows and missing:
            raise ValueError(f"{job_id}: shuffled rows missing {len(missing)} correct target keys.")

        correct_values = [float(row["overall_clap"]) for row in correct_rows]
        shuffled_target_means = {
            key: _mean(values) for key, values in shuffled_scores.items()
        }
        correct_shuffled_differences = [
            float(correct_by_key[key]["overall_clap"]) - shuffled_target_means[key]
            for key in correct_by_key
            if key in shuffled_target_means
        ]

        # Cross-song mappings may legitimately exclude targets (no eligible source
        # song for that subject), so unlike shuffled this is not an error condition.
        cross_song_target_means = {key: _mean(values) for key, values in cross_song_scores.items()}
        cross_song_target_means_by_job[job_id] = cross_song_target_means
        correct_cross_song_differences = [
            float(correct_by_key[key]["overall_clap"]) - cross_song_target_means[key]
            for key in correct_by_key
            if key in cross_song_target_means
        ]

        pretrained_differences = []
        pretrained_values = []
        pretrained_differences_by_subject: dict[Any, list[float]] = defaultdict(list)
        for row in correct_rows:
            baseline = pretrained_by_key.get(_pretrained_key(row))
            if baseline is None:
                continue
            value = float(baseline["overall_clap"])
            pretrained_values.append(value)
            difference = float(row["overall_clap"]) - value
            pretrained_differences.append(difference)
            pretrained_differences_by_subject[row.get("target_subject_id")].append(difference)

        cross_song_pretrained_differences = []
        cross_song_pretrained_differences_by_subject: dict[Any, list[float]] = defaultdict(list)
        for key, mean_value in cross_song_target_means.items():
            baseline = pretrained_by_key.get(_pretrained_key_from_target_key(key))
            if baseline is None:
                continue
            baseline_value = _available_score(baseline, "overall_clap")
            if baseline_value is None:
                continue
            difference = mean_value - baseline_value
            cross_song_pretrained_differences.append(difference)
            cross_song_pretrained_differences_by_subject[key[1]].append(difference)

        permutation_means = [_mean(values) for _, values in sorted(permutation_scores.items())]
        cross_song_permutation_means = [_mean(values) for _, values in sorted(cross_song_permutation_scores.items())]
        first = correct_rows[0]
        summary: dict[str, Any] = {
            "job_id": job_id,
            "model_type": first.get("model_type"),
            "regime": first.get("regime"),
            "subject": first.get("subject"),
            "held_out_subject": first.get("held_out_subject"),
            "num_correct_rows": len(correct_rows),
            "num_shuffle_rows": len(shuffled_rows),
            "num_cross_song_rows": len(cross_song_rows),
            "num_cross_song_targets": len(cross_song_target_means),
            "correct_mean": _mean(correct_values),
            "shuffled_mean": _mean_or_none(permutation_means),
            "shuffled_sd": float(statistics.stdev(permutation_means)) if len(permutation_means) > 1 else 0.0,
            "cross_song_mean": _mean_or_none(cross_song_permutation_means),
            "cross_song_sd": float(statistics.stdev(cross_song_permutation_means))
            if len(cross_song_permutation_means) > 1
            else 0.0,
            "pretrained_mean": _mean_or_none(pretrained_values),
            "metric_summaries": {
                field: summarize_metric(
                    correct_rows,
                    shuffled_rows,
                    pretrained_by_key,
                    field=field,
                    num_resamples=num_resamples,
                    seed=seed + metric_index * 10,
                )
                for metric_index, field in enumerate(SCORE_FIELDS)
            },
        }
        if correct_shuffled_differences:
            summary["correct_minus_shuffled"] = paired_bootstrap(
                correct_shuffled_differences,
                num_resamples=num_resamples,
                seed=seed,
            )
        else:
            summary["correct_minus_shuffled"] = None
        if pretrained_differences:
            summary["correct_minus_pretrained"] = paired_bootstrap(
                pretrained_differences,
                num_resamples=num_resamples,
                seed=seed + 1,
            )
        else:
            summary["correct_minus_pretrained"] = None
        summary["correct_minus_pretrained_cluster_bootstrap"] = (
            hierarchical_cluster_bootstrap(
                pretrained_differences_by_subject,
                num_resamples=num_resamples,
                seed=seed + 3,
            )
            if pretrained_differences_by_subject
            else None
        )
        summary["correct_minus_cross_song"] = (
            paired_bootstrap(correct_cross_song_differences, num_resamples=num_resamples, seed=seed + 4)
            if correct_cross_song_differences
            else None
        )
        summary["cross_song_minus_pretrained"] = (
            paired_bootstrap(cross_song_pretrained_differences, num_resamples=num_resamples, seed=seed + 5)
            if cross_song_pretrained_differences
            else None
        )
        summary["cross_song_minus_pretrained_cluster_bootstrap"] = (
            hierarchical_cluster_bootstrap(
                cross_song_pretrained_differences_by_subject,
                num_resamples=num_resamples,
                seed=seed + 6,
            )
            if cross_song_pretrained_differences_by_subject
            else None
        )
        summaries.append(summary)

    def _protocol_comparisons(
        by_key_per_job: Mapping[str, Mapping[tuple[Any, ...], Any]],
        *,
        value_of,
        bootstrap_seed_offset: int,
    ) -> list[dict[str, Any]]:
        comparisons = []
        jobs_by_context: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
        for summary in summaries:
            context = (summary["regime"], summary["subject"], summary["held_out_subject"])
            jobs_by_context[context][str(summary["model_type"]).upper()] = str(summary["job_id"])
        for context, model_jobs in sorted(jobs_by_context.items(), key=str):
            if not {"MULTICOND", "PASSIVE3"}.issubset(model_jobs):
                continue
            multi = by_key_per_job.get(model_jobs["MULTICOND"], {})
            passive = by_key_per_job.get(model_jobs["PASSIVE3"], {})
            common = sorted(set(multi) & set(passive), key=str)
            if not common:
                continue
            differences = [value_of(multi[key]) - value_of(passive[key]) for key in common]
            comparisons.append(
                {
                    "regime": context[0],
                    "subject": context[1],
                    "held_out_subject": context[2],
                    "multicond_job": model_jobs["MULTICOND"],
                    "passive3_job": model_jobs["PASSIVE3"],
                    **paired_bootstrap(differences, num_resamples=num_resamples, seed=seed + bootstrap_seed_offset),
                }
            )
        return comparisons

    protocol_comparisons = []
    jobs_by_context: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for summary in summaries:
        context = (summary["regime"], summary["subject"], summary["held_out_subject"])
        jobs_by_context[context][str(summary["model_type"]).upper()] = str(summary["job_id"])
    for context, model_jobs in sorted(jobs_by_context.items(), key=str):
        if not {"MULTICOND", "PASSIVE3"}.issubset(model_jobs):
            continue
        multi = {_target_key(row): row for row in correct_by_job[model_jobs["MULTICOND"]]}
        passive = {_target_key(row): row for row in correct_by_job[model_jobs["PASSIVE3"]]}
        common = sorted(set(multi) & set(passive), key=str)
        differences = [float(multi[key]["overall_clap"]) - float(passive[key]["overall_clap"]) for key in common]
        protocol_comparisons.append(
            {
                "regime": context[0],
                "subject": context[1],
                "held_out_subject": context[2],
                "multicond_job": model_jobs["MULTICOND"],
                "passive3_job": model_jobs["PASSIVE3"],
                **paired_bootstrap(differences, num_resamples=num_resamples, seed=seed + 2),
                "metric_comparisons": {
                    field: paired_bootstrap(
                        [
                            float(multi[key][field]) - float(passive[key][field])
                            for key in common
                            if _available_score(multi[key], field) is not None
                            and _available_score(passive[key], field) is not None
                        ],
                        num_resamples=num_resamples,
                        seed=seed + 2 + metric_index * 10,
                    )
                    if any(
                        _available_score(multi[key], field) is not None
                        and _available_score(passive[key], field) is not None
                        for key in common
                    )
                    else None
                    for metric_index, field in enumerate(SCORE_FIELDS)
                },
            }
        )

    protocol_gain_cross_song = _protocol_comparisons(
        cross_song_target_means_by_job,
        value_of=lambda value: float(value),
        bootstrap_seed_offset=7,
    )

    pretrained_summary: dict[str, dict[str, Any]] = {}
    for field in SCORE_FIELDS:
        values = []
        for row in pretrained:
            value = _available_score(row, field)
            if value is not None:
                values.append(value)
        pretrained_summary[field] = {"num_rows": len(values), "mean": _mean_or_none(values)}
    return {
        "num_records": len(records),
        "jobs": summaries,
        "pretrained": pretrained_summary,
        "protocol_gain_multicond_minus_passive3": protocol_comparisons,
        "protocol_gain_multicond_minus_passive3_cross_song": protocol_gain_cross_song,
        "bootstrap": {"num_resamples": int(num_resamples), "seed": int(seed)},
    }


def holm_correction(p_values: Mapping[Any, float]) -> dict[Any, float]:
    """Holm step-down adjusted p-values (monotone, family-wise error control)."""

    items = sorted(p_values.items(), key=lambda kv: kv[1])
    total = len(items)
    adjusted: dict[Any, float] = {}
    running_max = 0.0
    for rank, (key, p_value) in enumerate(items):
        candidate = min(1.0, (total - rank) * float(p_value))
        running_max = max(running_max, candidate)
        adjusted[key] = running_max
    return adjusted


def summarize_temporal_shift(
    records: Sequence[Mapping[str, Any]],
    *,
    offsets: Sequence[int],
    chunk_sec: float,
    include_random_within_song: bool,
    analysis_mode: str,
    num_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Build the temporal-shift tidy table and paired contrasts against offset 0.

    `records` should contain `correct` rows (used to synthesize the offset-0
    condition, so it is never regenerated), `temporal_shift` rows for the
    nonzero offsets, and optionally `shuffled` rows (used to synthesize the
    reused random-within-song condition when `include_random_within_song`).
    """

    if analysis_mode not in {"common_support", "max_available"}:
        raise ValueError(f"Unsupported temporal_shift analysis_mode: {analysis_mode!r}")

    nonzero_offsets = sorted({int(value) for value in offsets if int(value) != 0})
    all_offsets = sorted({int(value) for value in offsets} | {0})

    by_job_mode: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_job_mode[(str(row["job_id"]), str(row["evaluation_mode"]))].append(row)
    job_ids = sorted({job_id for job_id, mode in by_job_mode if mode == "correct"})

    tidy_rows: list[dict[str, Any]] = []
    job_reports: list[dict[str, Any]] = []

    for job_id in job_ids:
        correct_rows = by_job_mode.get((job_id, "correct"), [])
        shuffled_rows = by_job_mode.get((job_id, "shuffled"), [])
        temporal_rows = by_job_mode.get((job_id, "temporal_shift"), [])
        model_type = correct_rows[0].get("model_type")

        rows_by_offset: dict[int, dict[tuple[Any, ...], Mapping[str, Any]]] = defaultdict(dict)
        for row in correct_rows:
            rows_by_offset[0][_target_key(row)] = row
        for row in temporal_rows:
            offset = row.get("temporal_offset_chunks")
            if offset is None:
                continue
            rows_by_offset[int(offset)][_target_key(row)] = row

        for offset in all_offsets:
            for key, row in rows_by_offset.get(offset, {}).items():
                target_chunk = row.get("target_chunk_id")
                source_chunk = row.get("eeg_source_chunk_id")
                tidy_rows.append(
                    {
                        "condition_label": f"offset_{offset:+d}" if offset != 0 else "correct",
                        "model_type": model_type,
                        "job_id": job_id,
                        "target_subject_id": row.get("target_subject_id"),
                        "target_song_id": row.get("target_song_id"),
                        "target_chunk_id": target_chunk,
                        "target_timestamp_seconds": None
                        if target_chunk is None
                        else chunk_timestamp_seconds(int(target_chunk), chunk_sec=chunk_sec),
                        "eeg_source_chunk_id": source_chunk,
                        "eeg_source_timestamp_seconds": None
                        if source_chunk is None
                        else chunk_timestamp_seconds(int(source_chunk), chunk_sec=chunk_sec),
                        "offset_chunks": offset,
                        "offset_seconds": float(offset) * float(chunk_sec),
                        "generation_seed": row.get("generation_seed"),
                        "overall_clap": row.get("overall_clap"),
                        "validity": "valid",
                    }
                )

        random_summary: dict[str, Any] | None = None
        if include_random_within_song and shuffled_rows:
            shuffled_target_scores, _ = _permutation_target_means(shuffled_rows)
            shuffled_target_means = {key: _mean(values) for key, values in shuffled_target_scores.items()}
            shuffled_rows_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
            for row in shuffled_rows:
                shuffled_rows_by_key.setdefault(_target_key(row), row)
            for key, mean_value in shuffled_target_means.items():
                representative = shuffled_rows_by_key[key]
                target_chunk = representative.get("target_chunk_id")
                tidy_rows.append(
                    {
                        "condition_label": "random_within_song",
                        "model_type": model_type,
                        "job_id": job_id,
                        "target_subject_id": representative.get("target_subject_id"),
                        "target_song_id": representative.get("target_song_id"),
                        "target_chunk_id": target_chunk,
                        "target_timestamp_seconds": None
                        if target_chunk is None
                        else chunk_timestamp_seconds(int(target_chunk), chunk_sec=chunk_sec),
                        "eeg_source_chunk_id": None,
                        "eeg_source_timestamp_seconds": None,
                        "offset_chunks": None,
                        "offset_seconds": None,
                        "generation_seed": representative.get("generation_seed"),
                        "overall_clap": mean_value,
                        "validity": "valid",
                    }
                )
            zero_map = rows_by_offset.get(0, {})
            usable = sorted(set(zero_map) & set(shuffled_target_means), key=str)
            differences = [float(zero_map[key]["overall_clap"]) - shuffled_target_means[key] for key in usable]
            random_summary = {
                "n": len(usable),
                "mean_clap": _mean_or_none(list(shuffled_target_means.values())),
                "correct_minus_random_within_song": paired_bootstrap(differences, num_resamples=num_resamples, seed=seed)
                if differences
                else None,
            }

        # Intersecting every requested offset (including any with zero rows so far)
        # means common support is honestly empty until every offset has been
        # generated and scored, rather than silently ignoring missing offsets.
        common_support_keys = set.intersection(*(set(rows_by_offset.get(offset, {})) for offset in all_offsets))

        def paired_contrast(offset: int, *, restrict_to: set | None) -> dict[str, Any] | None:
            zero_map = rows_by_offset.get(0, {})
            offset_map = rows_by_offset.get(offset, {})
            usable = set(zero_map) & set(offset_map)
            if restrict_to is not None:
                usable &= restrict_to
            usable = sorted(usable, key=str)
            if not usable:
                return None
            differences = [float(zero_map[key]["overall_clap"]) - float(offset_map[key]["overall_clap"]) for key in usable]
            # Offsets can be negative; keep the derived per-offset seed non-negative
            # while still deterministic and distinct per offset.
            return paired_bootstrap(differences, num_resamples=num_resamples, seed=seed + 1000 + offset)

        contrasts_common = {offset: paired_contrast(offset, restrict_to=common_support_keys) for offset in nonzero_offsets}
        contrasts_max = {offset: paired_contrast(offset, restrict_to=None) for offset in nonzero_offsets}
        primary_contrasts = contrasts_common if analysis_mode == "common_support" else contrasts_max
        primary_p_values = {offset: contrast["p_value"] for offset, contrast in primary_contrasts.items() if contrast is not None}
        holm_adjusted = holm_correction(primary_p_values) if primary_p_values else {}

        pooled_abs_offsets: list[float] = []
        pooled_degradation: list[float] = []
        zero_map = rows_by_offset.get(0, {})
        for offset in nonzero_offsets:
            offset_map = rows_by_offset.get(offset, {})
            usable = sorted(set(zero_map) & set(offset_map) & common_support_keys, key=str)
            for key in usable:
                pooled_abs_offsets.append(float(abs(offset)))
                pooled_degradation.append(float(zero_map[key]["overall_clap"]) - float(offset_map[key]["overall_clap"]))
        degradation_correlation = (
            float(np.corrcoef(pooled_abs_offsets, pooled_degradation)[0, 1]) if len(pooled_abs_offsets) >= 2 else None
        )

        aggregate_rows = []
        for offset in all_offsets:
            offset_map = rows_by_offset.get(offset, {})
            values = [float(row["overall_clap"]) for row in offset_map.values() if _available_score(row, "overall_clap") is not None]
            n = len(values)
            sd = float(statistics.stdev(values)) if n > 1 else 0.0
            subject_means: dict[Any, list[float]] = defaultdict(list)
            for row in offset_map.values():
                value = _available_score(row, "overall_clap")
                if value is not None:
                    subject_means[row.get("target_subject_id")].append(value)
            aggregate_rows.append(
                {
                    "offset_chunks": offset,
                    "offset_seconds": float(offset) * float(chunk_sec),
                    "n_max_available": n,
                    "n_common_support": len(set(offset_map) & common_support_keys),
                    "mean_clap": _mean_or_none(values),
                    "sd_clap": sd,
                    "sem_clap": (sd / math.sqrt(n)) if n > 0 else None,
                    "correct_minus_shift_common_support": contrasts_common.get(offset),
                    "correct_minus_shift_max_available": contrasts_max.get(offset),
                    "holm_adjusted_p_value": (0.0 if offset == 0 else holm_adjusted.get(offset)),
                    "participant_stratified_mean_clap": {
                        str(subject): _mean_or_none(subject_values) for subject, subject_values in subject_means.items()
                    },
                }
            )

        job_reports.append(
            {
                "job_id": job_id,
                "model_type": model_type,
                "analysis_mode_primary": analysis_mode,
                "common_support_n": len(common_support_keys),
                "offsets": aggregate_rows,
                "random_within_song": random_summary,
                "degradation_vs_abs_offset_pearson_r": degradation_correlation,
                "degradation_vs_abs_offset_n": len(pooled_abs_offsets),
                "p_value_correction": "holm",
            }
        )

    return {
        "chunk_sec": float(chunk_sec),
        "offsets_requested": all_offsets,
        "analysis_mode_primary": analysis_mode,
        "include_random_within_song": bool(include_random_within_song),
        "jobs": job_reports,
        "tidy_rows": tidy_rows,
        "bootstrap": {"num_resamples": int(num_resamples), "seed": int(seed)},
    }


def write_temporal_shift_outputs(summary: Mapping[str, Any], *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tidy_rows = summary.get("tidy_rows", [])
    if tidy_rows:
        fieldnames = sorted(set().union(*(row.keys() for row in tidy_rows)))
        with (output_dir / "temporal_shift_tidy.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tidy_rows)

    aggregate_rows = []
    for job in summary.get("jobs", []):
        for offset_row in job.get("offsets", []):
            flat = {
                "job_id": job["job_id"],
                "model_type": job["model_type"],
                "analysis_mode_primary": job["analysis_mode_primary"],
            }
            flat.update({key: value for key, value in offset_row.items() if not isinstance(value, (dict, list))})
            for prefix in ("correct_minus_shift_common_support", "correct_minus_shift_max_available"):
                detail = offset_row.get(prefix)
                if isinstance(detail, Mapping):
                    flat.update({f"{prefix}_{key}": value for key, value in detail.items()})
            aggregate_rows.append(flat)
    if aggregate_rows:
        fieldnames = sorted(set().union(*(row.keys() for row in aggregate_rows)))
        with (output_dir / "temporal_shift_aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(aggregate_rows)

    (output_dir / "temporal_shift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_paper_outputs(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_chunk_path = output_dir / "per_chunk_scores.csv"
    if records:
        preferred = [
            "evaluation_mode",
            "job_id",
            "model_type",
            "checkpoint",
            "regime",
            "subject",
            "held_out_subject",
            "permutation_id",
            "permutation_seed",
            "target_chunk_id",
            "target_song_id",
            "target_subject_id",
            "eeg_source_chunk_id",
            "eeg_source_song_id",
            "eeg_source_subject_id",
            "generation_seed",
            "overall_clap",
            "drum_clap",
            "guitar_proxy_other_bass_clap",
            "vocal_clap",
        ]
        extras = sorted(set().union(*(row.keys() for row in records)) - set(preferred))
        with per_chunk_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=preferred + extras, extrasaction="ignore")
            writer.writeheader()
            for row in records:
                writer.writerow(dict(row))

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary_csv = output_dir / "summary.csv"
    flat_rows = []
    for row in summary.get("jobs", []):
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for prefix in ("correct_minus_shuffled", "correct_minus_pretrained"):
            detail = row.get(prefix)
            if isinstance(detail, Mapping):
                flat.update({f"{prefix}_{key}": value for key, value in detail.items()})
        flat_rows.append(flat)
    if flat_rows:
        fieldnames = sorted(set().union(*(row.keys() for row in flat_rows)))
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def load_json_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records.extend(dict(row) for row in payload)
        elif isinstance(payload, Mapping):
            records.extend(dict(row) for row in payload.get("samples", []))
        else:
            raise TypeError(f"Unsupported records payload in {path}: {type(payload)}")
    return records
