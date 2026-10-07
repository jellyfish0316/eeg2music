from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch
import yaml

from scripts.evaluate_generation import evaluate_rows, load_rows
from scripts.generate import (
    build_full_song_pool_dataset,
    generate_conditioned_evaluation,
    generate_legacy_correct_regression,
    prepare_conditioned_evaluation_resources,
    prepare_cross_song_evaluation_resources,
)
from scripts.generate_pretrained_baseline import generate_pretrained_from_targets
from scripts.prepare_cdt_eeg import _fallback_curry_paths
from scripts.train import build_dataloader
from utils.evaluation_analysis import (
    summarize_paper_records,
    summarize_temporal_shift,
    write_paper_outputs,
    write_temporal_shift_outputs,
)
from utils.evaluation_pairing import (
    build_cross_song_mappings,
    build_cyclic_derangements,
    build_temporal_shift_mapping,
    chunk_timestamp_seconds,
    common_support_target_indices,
    identity_mapping,
    keys_from_index_map,
    mapping_records,
    stable_generation_seed,
)


STAGES = (
    "preflight",
    "prepare-data",
    "prepare-stems",
    "legacy-regression",
    "verify-regression",
    "dry-run",
    "generate",
    "score",
    "summarize",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-driven paper control evaluations.")
    parser.add_argument("--config", default="configs/evaluate_paper_controls.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--mode", choices=["correct", "shuffled", "cross_song", "temporal_shift", "pretrained"])
    parser.add_argument("--jobs", nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-permutations",
        type=int,
        default=None,
        help="Limit shuffled permutations for a --max-samples smoke run.",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing a stage's existing outputs.")
    parser.add_argument("--hash-checkpoints", action="store_true", help="Hash every selected SMB checkpoint in preflight.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping in {path}.")
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    shuffle = config["shuffle"]
    if not bool(shuffle.get("same_subject")) or not bool(shuffle.get("same_song")):
        raise ValueError("Paper controls require same_subject=true and same_song=true.")
    if str(shuffle.get("on_failure")) != "error":
        raise ValueError("Paper controls require shuffle.on_failure=error.")
    if int(config["evaluation"].get("generation_replicates", 1)) != 1:
        raise ValueError("This canonical target index currently requires generation_replicates=1.")
    subject_indices = [int(row["index"]) for row in config["raw_subjects"]]
    if subject_indices != [0, 1, 2, 3]:
        raise ValueError("raw_subjects must remain ordered S0, S1, S2, S3.")


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def run_dir(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["output_root"]) / str(config["run_id"])


def selected_jobs(config: Mapping[str, Any], requested: Sequence[str] | None) -> list[dict[str, Any]]:
    jobs = [dict(job) for job in config.get("jobs", []) if bool(job.get("enabled", True))]
    ids = [str(job["id"]) for job in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate job IDs in paper controls config.")
    if requested is None or len(requested) == 0:
        return jobs
    unknown = sorted(set(requested) - set(ids))
    if unknown:
        raise KeyError(f"Unknown jobs: {unknown}")
    requested_set = set(requested)
    return [job for job in jobs if str(job["id"]) in requested_set]


def resolve_checkpoint(config: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    path = Path(str(job["checkpoint"]))
    if path.is_absolute():
        return path
    return Path(config["paths"]["checkpoint_root"]) / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inventory_entry(
    path: Path,
    *,
    root: Path,
    job_id: str,
    role: str,
    hash_file: bool,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "job_id": job_id,
        "role": role,
        "relative_path": str(path.relative_to(root)),
        "path": str(path),
        "exists": path.exists(),
    }
    if not entry["exists"]:
        return entry
    stat = path.stat()
    entry.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    entry["sha256"] = sha256_file(path) if hash_file else "deferred-until-staging"
    return entry


def inventory_job_files(
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    hash_checkpoint: bool,
) -> list[dict[str, Any]]:
    checkpoint_root = Path(config["paths"]["checkpoint_root"])
    checkpoint = resolve_checkpoint(config, job)
    entries = [
        inventory_entry(
            checkpoint,
            root=checkpoint_root,
            job_id=str(job["id"]),
            role="checkpoint",
            hash_file=hash_checkpoint,
        )
    ]
    inventory_config = config.get("smb_inventory", {})
    result_path = checkpoint.parent / str(inventory_config.get("archived_result", "result.json"))
    entries.append(
        inventory_entry(
            result_path,
            root=checkpoint_root,
            job_id=str(job["id"]),
            role="archived_result",
            hash_file=True,
        )
    )
    if "archived_manifest" not in job:
        raise KeyError(f"Job {job['id']} is missing archived_manifest in the paper controls registry.")
    manifest_path = checkpoint.parent / str(job["archived_manifest"])
    entries.append(
        inventory_entry(
            manifest_path,
            root=checkpoint_root,
            job_id=str(job["id"]),
            role="archived_manifest",
            hash_file=True,
        )
    )
    return entries


def stage_checkpoint(config: Mapping[str, Any], job: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    source = resolve_checkpoint(config, job)
    staging_dir = Path(config["paths"]["checkpoint_staging_dir"])
    staging_dir.mkdir(parents=True, exist_ok=True)
    destination = staging_dir / f"{job['id']}{source.suffix}"
    checksum_path = destination.with_suffix(destination.suffix + ".sha256")

    # Trust an already-staged copy with a recorded checksum without touching the
    # archival source at all. The source (often a flaky SMB/GVFS mount) is only
    # needed to freshly stage a copy or to recompute a missing checksum, never to
    # re-verify a copy that already carries its own recorded sha256.
    if destination.exists() and checksum_path.exists():
        checksum = checksum_path.read_text(encoding="utf-8").strip().split()[0]
        verify = bool(config.get("runtime", {}).get("verify_checkpoint_sha256", True))
        if not verify or sha256_file(destination) == checksum:
            return destination, {
                "source": str(source),
                "staged": str(destination),
                "size": int(destination.stat().st_size),
                "sha256": checksum,
                "reused": True,
                "verified": verify,
                "source_checked": False,
            }
        raise ValueError(
            f"Staged checkpoint {destination} does not match its recorded checksum in {checksum_path}; "
            "refusing to silently reuse a possibly corrupted local copy. Delete the stale staged "
            "files under checkpoint_staging_dir to force re-staging from the archival source."
        )

    if not source.exists():
        raise FileNotFoundError(
            f"No valid staged checkpoint at {destination} (checksum sidecar missing or absent) and "
            f"the archival source is unreachable: {source}"
        )
    source_stat = source.stat()
    if destination.exists() and int(destination.stat().st_size) == int(source_stat.st_size):
        checksum = sha256_file(destination)
        checksum_path.write_text(f"{checksum}  {destination.name}\n", encoding="utf-8")
        return destination, {
            "source": str(source),
            "staged": str(destination),
            "size": int(source_stat.st_size),
            "sha256": checksum,
            "reused": True,
            "verified": True,
            "source_checked": True,
        }

    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with source.open("rb") as source_handle, temporary.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=8 * 1024 * 1024)
    temporary.replace(destination)
    checksum = sha256_file(destination)
    checksum_path.write_text(f"{checksum}  {destination.name}\n", encoding="utf-8")
    return destination, {
        "source": str(source),
        "staged": str(destination),
        "size": int(source_stat.st_size),
        "sha256": checksum,
        "reused": False,
        "verified": True,
        "source_checked": True,
    }


def target_index(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    evaluation = config["evaluation"]
    chunk_range = evaluation["target_chunks"]
    start = int(chunk_range["start"])
    stop = int(chunk_range["stop"])
    target_dir = Path(config["paths"]["target_audio_dir"])
    rows = []
    for song_name in evaluation["target_songs"]:
        path = target_dir / f"{song_name}.wav"
        if not path.exists():
            raise FileNotFoundError(path)
        info = sf.info(path)
        expected_sample_rate = int(evaluation["sample_rate"])
        if int(info.samplerate) != expected_sample_rate:
            raise ValueError(
                f"{path} has sample rate {info.samplerate}, expected {expected_sample_rate}."
            )
        chunk_samples = round(float(evaluation["chunk_sec"]) * int(info.samplerate))
        complete_chunks = int(info.frames) // int(chunk_samples)
        if stop > complete_chunks:
            raise ValueError(f"{path} has {complete_chunks} complete chunks, requested stop={stop}.")
        for chunk_index in range(start, stop):
            rows.append(
                {
                    "target_song_id": str(song_name),
                    "target_chunk_id": int(chunk_index),
                    "target_audio_path": str(path.resolve()),
                    "audio_sample_rate": int(info.samplerate),
                    "chunk_sec": float(evaluation["chunk_sec"]),
                    "generation_seed": stable_generation_seed(
                        base_seed=int(evaluation["generation_seed"]),
                        song_id=str(song_name),
                        chunk_index=chunk_index,
                    ),
                }
            )
    return rows


def write_target_index(config: Mapping[str, Any]) -> Path:
    path = run_dir(config) / "target_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"samples": target_index(config)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def stage_readonly_file(source: Path, destination: Path) -> Path:
    """Stage a large immutable SMB file with size-checked resumable semantics."""

    if not source.exists():
        raise FileNotFoundError(source)
    source_size = int(source.stat().st_size)
    if destination.exists() and int(destination.stat().st_size) == source_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    print(f"stage {source} -> {destination} ({source_size} bytes)", flush=True)
    with source.open("rb") as source_handle, temporary.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=8 * 1024 * 1024)
    if int(temporary.stat().st_size) != source_size:
        raise IOError(f"Incomplete staged file: {temporary}")
    temporary.replace(destination)
    return destination


def job_config(config: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    cfg = load_yaml(REPO_ROOT / str(job["training_config"]))
    cfg = deep_merge(cfg, config.get("runtime", {}).get("config_overrides", {}))
    variant = str(job["preprocessing_variant"])
    processed_key = "processed_interpolated_dir" if variant == "interpolated" else "processed_non_interpolated_dir"
    processed_dir = Path(config["paths"][processed_key])
    audio_dir = Path(config["paths"]["target_audio_dir"])
    for song in cfg["data"].get("songs", []):
        song_name = str(song["name"])
        song["mat_path"] = str(processed_dir / f"{song_name}_passive_Processed.mat")
        song["audio_path"] = str(audio_dir / f"{song_name}.wav")
        for source_name, source in (song.get("sources") or {}).items():
            source["mat_path"] = str(processed_dir / f"{song_name}_{source_name}_Processed.mat")
    cfg["data"]["eeg_chunk_cache_dir"] = None
    cfg["data"]["text_prompt"] = str(config["evaluation"]["prompt"])
    cfg["data"]["chunk_sec"] = float(config["evaluation"]["chunk_sec"])
    cfg["data"]["audio_fs"] = int(config["evaluation"]["sample_rate"])
    cfg.setdefault("audio_encoder", {})["model_id"] = str(config["evaluation"]["model_id"])
    cfg.setdefault("latent_cache", {})["enabled"] = False
    cfg["train"]["device"] = str(config.get("runtime", {}).get("device", "cuda"))
    return cfg


def preflight(config: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]], *, hash_checkpoints: bool) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    inventory: list[dict[str, Any]] = []
    for job in jobs:
        checkpoint = resolve_checkpoint(config, job)
        job_inventory = inventory_job_files(config, job, hash_checkpoint=hash_checkpoints)
        inventory.extend(job_inventory)
        checkpoint_entry = job_inventory[0]
        if not checkpoint_entry["exists"]:
            problems.append(f"Missing checkpoint: {checkpoint}")
        for entry in job_inventory[1:]:
            if not entry["exists"]:
                problems.append(f"Missing {entry['role']} for {job['id']}: {entry['path']}")
        training_config = REPO_ROOT / str(job["training_config"])
        if not training_config.exists():
            problems.append(f"Missing training config for {job['id']}: {training_config}")

    smb_root = Path(config["paths"]["smb_root"])
    for subject in config["raw_subjects"]:
        subject_dir = smb_root / str(subject["directory"])
        for song in config["preprocessing"]["songs"]:
            raw_path = subject_dir / f"{song}.cdt"
            if not raw_path.exists():
                problems.append(f"Missing raw CDT: {raw_path}")
                continue
            try:
                _fallback_curry_paths(raw_path)
            except FileNotFoundError as exc:
                problems.append(str(exc))

    required_variants = {str(job["preprocessing_variant"]) for job in jobs}
    for variant in required_variants:
        key = "processed_interpolated_dir" if variant == "interpolated" else "processed_non_interpolated_dir"
        directory = Path(config["paths"][key])
        for song in config["preprocessing"]["songs"]:
            for condition in config["preprocessing"]["conditions"]:
                path = directory / f"{song}_{condition}_Processed.mat"
                if not path.exists():
                    problems.append(f"Missing processed EEG ({variant}): {path}")

    source_audio_dir = Path(config["paths"]["target_audio_smb_dir"])
    target_audio_dir = Path(config["paths"]["target_audio_dir"])
    for song in config["preprocessing"]["songs"]:
        source = source_audio_dir / f"{song}.wav"
        destination = target_audio_dir / source.name
        if not source.exists():
            problems.append(f"Missing SMB target audio: {source}")
        elif destination.exists() and sha256_file(source) != sha256_file(destination):
            problems.append(f"Staged target audio differs from SMB source: {destination}")

    stems_root = Path(config["paths"]["stems_root"])
    for song in config["evaluation"]["target_songs"]:
        for stem_name in config["stems"].values():
            if isinstance(stem_name, Mapping):
                continue
            stem_path = stems_root / str(song) / f"{stem_name}.wav"
            if not stem_path.exists():
                warnings.append(f"Missing stem (prepare-stems required): {stem_path}")

    if bool(config.get("runtime", {}).get("require_cuda_for_generation", True)) and not torch.cuda.is_available():
        warnings.append("CUDA is not visible; generation stages will refuse to run.")

    try:
        target_path: Path | None = write_target_index(config)
    except (FileNotFoundError, ValueError) as exc:
        problems.append(str(exc))
        target_path = None
    report = {
        "ok": not problems,
        "problems": problems,
        "warnings": warnings,
        "selected_jobs": [job["id"] for job in jobs],
        "file_inventory": inventory,
        "target_index": None if target_path is None else str(target_path),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    metadata_dir = run_dir(config) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (metadata_dir / "file_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    (metadata_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return report


def prepare_data(config: Mapping[str, Any], *, force: bool) -> None:
    smb_root = Path(config["paths"]["smb_root"])
    preprocessing = config["preprocessing"]
    subjects = sorted(config["raw_subjects"], key=lambda row: int(row["index"]))
    source_audio_dir = Path(config["paths"]["target_audio_smb_dir"])
    target_audio_dir = Path(config["paths"]["target_audio_dir"])
    target_audio_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(preprocessing["interpolated_auto_bad_impedance_threshold"])
    disconnected = float(preprocessing["interpolated_disconnected_impedance_value"])
    if threshold > disconnected:
        raise ValueError(
            "interpolated_auto_bad_impedance_threshold must detect the configured disconnected impedance value."
        )
    for song in preprocessing["songs"]:
        source_audio = source_audio_dir / f"{song}.wav"
        destination_audio = target_audio_dir / source_audio.name
        if not source_audio.exists():
            raise FileNotFoundError(source_audio)
        if destination_audio.exists():
            if sha256_file(source_audio) == sha256_file(destination_audio):
                continue
            if not force:
                raise FileExistsError(
                    f"Staged target audio differs from SMB source: {destination_audio}. Use --force after inspection."
                )
        shutil.copy2(source_audio, destination_audio)
    variants = [
        ("non_interpolated", Path(config["paths"]["processed_non_interpolated_dir"]), False),
        ("interpolated", Path(config["paths"]["processed_interpolated_dir"]), True),
    ]
    for _, output_dir, _ in variants:
        output_dir.mkdir(parents=True, exist_ok=True)
    raw_staging_root = Path(config["paths"]["raw_cdt_staging_dir"])
    for song in preprocessing["songs"]:
        raw_files = []
        for subject in subjects:
            source_cdt = smb_root / str(subject["directory"]) / f"{song}.cdt"
            source_data, source_dpa, source_ceo = _fallback_curry_paths(source_cdt)
            subject_staging = raw_staging_root / str(subject["subject"])
            for source in (source_data, source_dpa, source_ceo):
                stage_readonly_file(source, subject_staging / source.name)
            raw_files.append(subject_staging / source_data.name)

        for variant, output_dir, interpolate in variants:
            expected = [output_dir / f"{song}_{condition}_Processed.mat" for condition in preprocessing["conditions"]]
            if all(path.exists() for path in expected) and not force:
                print(f"skip existing {variant} {song}", flush=True)
                continue
            if any(path.exists() for path in expected) and not force:
                raise FileExistsError(f"Partial outputs exist for {variant} {song}; use --force after inspection.")
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/prepare_cdt_eeg.py"),
                "convert-events",
                *[str(path) for path in raw_files],
                "--output-dir",
                str(output_dir),
                "--song-name",
                str(song),
                "--conditions",
                *[str(value) for value in preprocessing["conditions"]],
                "--keep-eeg-channels",
                str(preprocessing["keep_eeg_channels"]),
                "--dst-fs",
                str(preprocessing["dst_fs"]),
                "--center-using-first-samples",
                str(preprocessing["center_using_first_samples"]),
                "--trim-to-shortest",
            ]
            if not bool(preprocessing.get("robust_scale", True)):
                command.append("--no-robust-scale")
            if interpolate:
                command.extend(
                    [
                        "--auto-bad-impedance-threshold",
                        str(preprocessing["interpolated_auto_bad_impedance_threshold"]),
                        "--max-auto-bad-fraction",
                        str(preprocessing["max_auto_bad_fraction"]),
                    ]
                )
            subprocess.run(command, cwd=REPO_ROOT, check=True)


def prepare_stems(config: Mapping[str, Any], *, force: bool) -> None:
    if importlib.util.find_spec("demucs") is None:
        raise RuntimeError("prepare-stems requires the optional 'demucs' package (pip install demucs).")
    stems_root = Path(config["paths"]["stems_root"])
    target_dir = Path(config["paths"]["target_audio_dir"])
    for song in config["evaluation"]["target_songs"]:
        destination = stems_root / str(song)
        expected = [destination / "drums.wav", destination / "vocals.wav", destination / "guitar_proxy_other_bass.wav"]
        if all(path.exists() for path in expected) and not force:
            print(f"skip existing stems for {song}", flush=True)
            continue
        source = target_dir / f"{song}.wav"
        subprocess.run(
            [sys.executable, "-m", "demucs", "-n", "htdemucs", "--out", str(stems_root.parent), str(source)],
            cwd=REPO_ROOT,
            check=True,
        )
        demucs_dir = stems_root / str(song)
        other, sr_other = sf.read(demucs_dir / "other.wav", dtype="float32", always_2d=True)
        bass, sr_bass = sf.read(demucs_dir / "bass.wav", dtype="float32", always_2d=True)
        if sr_other != sr_bass:
            raise ValueError(f"Demucs stem sample-rate mismatch for {song}.")
        length = min(len(other), len(bass))
        sf.write(demucs_dir / "guitar_proxy_other_bass.wav", other[:length] + bass[:length], sr_other)


def require_generation_device(config: Mapping[str, Any]) -> None:
    if bool(config.get("runtime", {}).get("require_cuda_for_generation", True)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by runtime.require_cuda_for_generation but is not visible.")


def legacy_regression_stage(
    config: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    *,
    force: bool,
    max_samples: int | None = None,
) -> None:
    """Generate the archived global-RNG protocol using rebuilt EEG data."""

    require_generation_device(config)
    evaluation = config["evaluation"]
    legacy = config["legacy_regression"]
    references = config.get("regression_references", {})
    for job in jobs:
        job_id = str(job["id"])
        if job_id not in references:
            continue
        generation_root = run_dir(config) / "smoke" if max_samples is not None else run_dir(config)
        output_dir = generation_root / "legacy_correct" / job_id
        manifest = output_dir / "manifest.json"
        if manifest.exists() and not force:
            print(f"skip existing {manifest}", flush=True)
            continue
        cfg = deep_merge(
            job_config(config, job),
            {
                "seed": int(legacy["generation_seed"]),
                "data": {
                    "batch_size": int(legacy["batch_size"]),
                    "num_workers": 0,
                },
                "model": {
                    "projector": {
                        "lat_grid": [int(value) for value in legacy["latent_grid"]],
                    }
                },
            },
        )
        checkpoint_source = resolve_checkpoint(config, job)
        if bool(config.get("runtime", {}).get("stage_checkpoints", True)):
            checkpoint, checkpoint_meta = stage_checkpoint(config, job)
        else:
            checkpoint = checkpoint_source
            checkpoint_meta = {"source": str(checkpoint_source), "staged": None}
        metadata = {
            "id": job_id,
            "model_type": str(job["model_type"]),
            "regime": str(job["regime"]),
            "subject": job.get("subject"),
            "held_out_subject": job.get("held_out_subject"),
            "label": job.get("label"),
            "checkpoint_source": str(checkpoint_source),
            "checkpoint_inventory": checkpoint_meta,
            "generation_seed": int(legacy["generation_seed"]),
        }
        resources = prepare_conditioned_evaluation_resources(
            cfg=cfg,
            checkpoint=checkpoint,
            split=str(evaluation["split"]),
            subjects=[int(value) for value in job["evaluation_subjects"]],
        )
        generate_legacy_correct_regression(
            cfg=cfg,
            checkpoint=checkpoint,
            split=str(evaluation["split"]),
            output_dir=output_dir,
            job_metadata=metadata,
            num_inference_steps=int(evaluation["num_inference_steps"]),
            batch_size=int(legacy["batch_size"]),
            resources=resources,
            max_samples=max_samples,
        )
        del resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_correct_regressions(
    config: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Gate rebuilt EEG via the archived RNG protocol before shuffled generation."""

    references = config.get("regression_references", {})
    tolerance = float(config["preprocessing"]["regression_tolerance"])
    gates: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["id"])
        reference = references.get(job_id)
        if reference is None:
            continue
        paired_correct_manifest = run_dir(config) / "correct" / job_id / "manifest.json"
        if not paired_correct_manifest.exists():
            raise FileNotFoundError(
                "Formal shuffled generation requires its paired correct manifest for "
                f"{job_id}: {paired_correct_manifest}"
            )
        manifest = run_dir(config) / "legacy_correct" / job_id / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                "Run --stage legacy-regression before formal shuffled generation; "
                f"missing {manifest}"
            )
        output_dir = run_dir(config) / "metrics" / "legacy_correct_regression" / job_id
        per_sample_path = output_dir / "per_sample_scores.json"
        if not per_sample_path.exists():
            evaluate_rows(load_rows([str(manifest)]), output_dir=output_dir)
        rows = json.loads(per_sample_path.read_text(encoding="utf-8"))
        values = [float(row["overall_clap"]) for row in rows]
        actual_mean = float(sum(values) / len(values)) if values else float("nan")
        expected_rows = int(reference["num_rows"])
        delta = actual_mean - float(reference["mean_clap"])
        passed = len(rows) == expected_rows and abs(delta) <= tolerance
        gates.append(
            {
                "job_id": job_id,
                "rng_protocol": "legacy_global_batch_stream",
                "expected_rows": expected_rows,
                "actual_rows": len(rows),
                "reference_mean": float(reference["mean_clap"]),
                "actual_mean": actual_mean,
                "delta": delta,
                "tolerance": tolerance,
                "passed": passed,
            }
        )
    gate_path = run_dir(config) / "metrics" / "correct_regression_gate.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gates, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    failures = [gate for gate in gates if not gate["passed"]]
    if failures:
        failed_ids = ", ".join(str(gate["job_id"]) for gate in failures)
        raise RuntimeError(
            f"Correct regression gate failed for {failed_ids}; refusing formal shuffled generation. "
            f"See {gate_path}."
        )
    return gates


def dry_run_stage(config: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Load metadata, build every requested mapping, and validate constraints without generating audio."""

    evaluation = config["evaluation"]
    cross_song_cfg = config.get("cross_song", {})
    temporal_cfg = config.get("temporal_shift", {})
    target_song_names = {str(name) for name in evaluation["target_songs"]}
    chunk_sec = float(evaluation["chunk_sec"])

    report: dict[str, Any] = {"run_id": config["run_id"], "jobs": []}
    for job in jobs:
        cfg = job_config(config, job)
        subjects = [int(value) for value in job["evaluation_subjects"]]
        dataset, _ = build_dataloader(cfg, subjects=subjects, shuffle=False, chunk_split_name=str(evaluation["split"]))
        keys = keys_from_index_map(dataset.index_map)
        job_report: dict[str, Any] = {
            "job_id": job["id"],
            "num_evaluation_targets": len(keys),
            "participant_distribution": dict(Counter(dataset.song_records[key.song_index].name for key in keys)),
        }

        sample_song_name = dataset.song_records[keys[0].song_index].name
        seed_repeat_a = stable_generation_seed(
            base_seed=int(evaluation["generation_seed"]), song_id=sample_song_name, chunk_index=keys[0].chunk_index
        )
        seed_repeat_b = stable_generation_seed(
            base_seed=int(evaluation["generation_seed"]), song_id=sample_song_name, chunk_index=keys[0].chunk_index
        )
        job_report["shared_seed_is_deterministic_per_target"] = seed_repeat_a == seed_repeat_b
        job_report["shared_seed_depends_only_on_target_song_and_chunk"] = True

        if bool(cross_song_cfg.get("enabled", False)):
            pool_dataset = build_full_song_pool_dataset(cfg, subjects=subjects)
            pool_keys = keys_from_index_map(pool_dataset.index_map)
            target_song_indices = {
                index for index, record in enumerate(pool_dataset.song_records) if record.name in target_song_names
            }
            mappings, excluded = build_cross_song_mappings(
                pool_keys,
                num_permutations=int(cross_song_cfg["num_permutations"]),
                seed=int(cross_song_cfg["mapping_seed"]),
                target_song_indices=target_song_indices,
            )
            different_song_confirmed = all(
                pool_keys[source].song_index != pool_keys[target].song_index
                for mapping in mappings
                for target, source in mapping.items()
            )
            source_song_counts = (
                Counter(pool_dataset.song_records[pool_keys[source].song_index].name for source in mappings[0].values())
                if mappings
                else {}
            )
            job_report["cross_song"] = {
                "num_permutations": len(mappings),
                "valid_targets_per_permutation": len(mappings[0]) if mappings else 0,
                "excluded_targets": len(excluded),
                "excluded_reasons": dict(Counter(row["reason"] for row in excluded)),
                "source_song_distribution_first_permutation": dict(source_song_counts),
                "different_song_confirmed": different_song_confirmed,
            }

        if bool(temporal_cfg.get("enabled", False)):
            offsets = [int(value) for value in temporal_cfg["offsets"]]
            mappings_by_offset: dict[int, dict[int, int]] = {}
            excluded_by_offset: dict[int, list[dict[str, Any]]] = {}
            same_song_confirmed = True
            for offset in offsets:
                mapping, excluded = build_temporal_shift_mapping(keys, offset=offset)
                mappings_by_offset[offset] = mapping
                excluded_by_offset[offset] = excluded
                same_song_confirmed = same_song_confirmed and all(
                    keys[source].song_index == keys[target].song_index for target, source in mapping.items()
                )
            common_support = common_support_target_indices(mappings_by_offset)
            job_report["temporal_shift"] = {
                "offsets_chunks": offsets,
                "offsets_seconds": [offset * chunk_sec for offset in offsets],
                "valid_counts_max_available": {str(offset): len(mapping) for offset, mapping in mappings_by_offset.items()},
                "excluded_counts": {str(offset): len(rows) for offset, rows in excluded_by_offset.items()},
                "common_support_count": len(common_support),
                "same_song_confirmed": same_song_confirmed,
            }

        report["jobs"].append(job_report)

    metadata_dir = run_dir(config) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "dry_run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def generate_stage(
    config: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    max_samples: int | None,
    max_permutations: int | None,
    force: bool,
) -> None:
    require_generation_device(config)
    evaluation = config["evaluation"]
    if max_permutations is not None and max_samples is None:
        raise ValueError("--max-permutations is only valid with --max-samples smoke runs.")
    generation_root = run_dir(config) / "smoke" if max_samples is not None else run_dir(config)
    if mode == "pretrained":
        target_path = write_target_index(config)
        rows = json.loads(target_path.read_text(encoding="utf-8"))["samples"]
        output_dir = generation_root / "pretrained"
        manifest = output_dir / "manifest.json"
        if manifest.exists() and not force:
            print(f"skip existing {manifest}", flush=True)
            return
        generate_pretrained_from_targets(
            target_rows=rows,
            output_dir=output_dir,
            model_id=str(evaluation["model_id"]),
            prompt=str(evaluation["prompt"]),
            chunk_sec=float(evaluation["chunk_sec"]),
            generation_seed=int(evaluation["generation_seed"]),
            num_inference_steps=int(evaluation["num_inference_steps"]),
            guidance_scale=float(evaluation["guidance_scale"]),
            max_samples=max_samples,
        )
        return

    if mode == "shuffled" and max_samples is None:
        verify_correct_regressions(config, jobs)
    elif mode == "shuffled" and max_samples is not None:
        print("smoke shuffled run: full correct regression gate is intentionally deferred", flush=True)

    for job in jobs:
        cfg = job_config(config, job)
        subjects = [int(value) for value in job["evaluation_subjects"]]
        if mode == "correct":
            expected_manifests = [generation_root / "correct" / str(job["id"]) / "manifest.json"]
        elif mode == "shuffled":
            permutation_count = int(config["shuffle"]["num_permutations"])
            if max_permutations is not None:
                permutation_count = min(permutation_count, int(max_permutations))
            expected_manifests = [
                generation_root
                / "shuffled"
                / str(job["id"])
                / f"permutation_{permutation_id:03d}"
                / "manifest.json"
                for permutation_id in range(permutation_count)
            ]
        elif mode == "cross_song":
            permutation_count = int(config["cross_song"]["num_permutations"])
            if max_permutations is not None:
                permutation_count = min(permutation_count, int(max_permutations))
            expected_manifests = [
                generation_root
                / "cross_song"
                / str(job["id"])
                / f"permutation_{permutation_id:03d}"
                / "manifest.json"
                for permutation_id in range(permutation_count)
            ]
        else:  # temporal_shift
            offsets = [int(value) for value in config["temporal_shift"]["offsets"] if int(value) != 0]
            if max_permutations is not None:
                offsets = offsets[: max(0, int(max_permutations))]
            expected_manifests = [
                generation_root / "temporal_shift" / str(job["id"]) / f"offset_{offset:+03d}" / "manifest.json"
                for offset in offsets
            ]
        if expected_manifests and all(path.exists() for path in expected_manifests) and not force:
            print(f"skip completed {mode} job {job['id']}", flush=True)
            continue
        checkpoint_source = resolve_checkpoint(config, job)
        if bool(config.get("runtime", {}).get("stage_checkpoints", True)):
            checkpoint, checkpoint_meta = stage_checkpoint(config, job)
        else:
            checkpoint = checkpoint_source
            checkpoint_meta = {"source": str(checkpoint_source), "staged": None}
        metadata = {
            "id": str(job["id"]),
            "model_type": str(job["model_type"]),
            "regime": str(job["regime"]),
            "subject": job.get("subject"),
            "held_out_subject": job.get("held_out_subject"),
            "label": job.get("label"),
            "checkpoint_source": str(checkpoint_source),
            "checkpoint_inventory": checkpoint_meta,
        }
        if mode == "cross_song":
            resources = prepare_cross_song_evaluation_resources(
                cfg=cfg,
                checkpoint=checkpoint,
                subjects=subjects,
            )
        else:
            resources = prepare_conditioned_evaluation_resources(
                cfg=cfg,
                checkpoint=checkpoint,
                split=str(evaluation["split"]),
                subjects=subjects,
            )
        keys = keys_from_index_map(resources.dataset.index_map)
        if mode == "correct":
            output_dir = generation_root / "correct" / str(job["id"])
            manifest = output_dir / "manifest.json"
            if manifest.exists() and not force:
                print(f"skip existing {manifest}", flush=True)
                continue
            generate_conditioned_evaluation(
                cfg=cfg,
                checkpoint=checkpoint,
                split=str(evaluation["split"]),
                subjects=[int(value) for value in job["evaluation_subjects"]],
                output_dir=output_dir,
                evaluation_mode="correct",
                mapping=identity_mapping(keys),
                job_metadata=metadata,
                num_inference_steps=int(evaluation["num_inference_steps"]),
                guidance_scale=float(evaluation["guidance_scale"]),
                eta=float(evaluation.get("eta", 0.0)),
                generation_seed=int(evaluation["generation_seed"]),
                max_samples=max_samples,
                resources=resources,
            )
        elif mode == "shuffled":
            shuffle_cfg = config["shuffle"]
            permutation_count = int(shuffle_cfg["num_permutations"])
            if max_permutations is not None:
                permutation_count = min(permutation_count, int(max_permutations))
            mappings = build_cyclic_derangements(
                keys,
                num_permutations=permutation_count,
                seed=int(shuffle_cfg["seed"]),
                min_chunk_distance=int(shuffle_cfg["min_chunk_distance"]),
            )
            for permutation_id, mapping in enumerate(mappings):
                permutation_seed = int(shuffle_cfg["seed"]) + permutation_id
                mapping_dir = generation_root / "mappings" / str(job["id"])
                mapping_dir.mkdir(parents=True, exist_ok=True)
                mapping_path = mapping_dir / f"permutation_{permutation_id:03d}.json"
                mapping_path.write_text(
                    json.dumps(
                        {
                            "job_id": job["id"],
                            "model_type": job["model_type"],
                            "regime": job["regime"],
                            "permutation_id": permutation_id,
                            "shuffle_seed": permutation_seed,
                            "minimum_chunk_distance": int(shuffle_cfg["min_chunk_distance"]),
                            "mapping": mapping_records(keys, mapping),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                output_dir = (
                    generation_root
                    / "shuffled"
                    / str(job["id"])
                    / f"permutation_{permutation_id:03d}"
                )
                manifest = output_dir / "manifest.json"
                if manifest.exists() and not force:
                    print(f"skip existing {manifest}", flush=True)
                    continue
                generate_conditioned_evaluation(
                    cfg=cfg,
                    checkpoint=checkpoint,
                    split=str(evaluation["split"]),
                    subjects=[int(value) for value in job["evaluation_subjects"]],
                    output_dir=output_dir,
                    evaluation_mode="shuffled",
                    mapping=mapping,
                    job_metadata=metadata,
                    num_inference_steps=int(evaluation["num_inference_steps"]),
                    guidance_scale=float(evaluation["guidance_scale"]),
                    eta=float(evaluation.get("eta", 0.0)),
                    generation_seed=int(evaluation["generation_seed"]),
                    permutation_id=permutation_id,
                    permutation_seed=permutation_seed,
                    max_samples=max_samples,
                    resources=resources,
                    shuffle_min_chunk_distance=int(shuffle_cfg["min_chunk_distance"]),
                )
        elif mode == "cross_song":
            cross_song_cfg = config["cross_song"]
            permutation_count = int(cross_song_cfg["num_permutations"])
            if max_permutations is not None:
                permutation_count = min(permutation_count, int(max_permutations))
            target_song_names = {str(name) for name in evaluation["target_songs"]}
            target_song_indices = {
                index for index, record in enumerate(resources.dataset.song_records) if record.name in target_song_names
            }
            mappings, excluded = build_cross_song_mappings(
                keys,
                num_permutations=permutation_count,
                seed=int(cross_song_cfg["mapping_seed"]),
                target_song_indices=target_song_indices,
            )
            on_failure = str(cross_song_cfg.get("on_failure", "error"))
            if excluded and on_failure == "error":
                raise RuntimeError(
                    f"cross_song job {job['id']}: {len(excluded)} target(s) have no eligible "
                    f"cross-song source (subject present in only one song). Set cross_song.on_failure=skip "
                    "to proceed while excluding them. Excluded sample: "
                    f"{excluded[0]}"
                )
            mapping_dir = generation_root / "cross_song_mappings" / str(job["id"])
            mapping_dir.mkdir(parents=True, exist_ok=True)
            (mapping_dir / "excluded_targets.json").write_text(
                json.dumps(
                    {
                        "job_id": job["id"],
                        "num_excluded": len(excluded),
                        "excluded": excluded,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            for permutation_id, mapping in enumerate(mappings):
                permutation_seed = int(cross_song_cfg["mapping_seed"]) + permutation_id
                source_song_counts = Counter(
                    resources.dataset.song_records[keys[source_index].song_index].name
                    for source_index in mapping.values()
                )
                mapping_path = mapping_dir / f"permutation_{permutation_id:03d}.json"
                mapping_path.write_text(
                    json.dumps(
                        {
                            "job_id": job["id"],
                            "model_type": job["model_type"],
                            "regime": job["regime"],
                            "permutation_id": permutation_id,
                            "mapping_seed": permutation_seed,
                            "target_songs": sorted(target_song_names),
                            "num_valid_targets": len(mapping),
                            "num_excluded_targets": len(excluded),
                            "source_song_distribution": dict(source_song_counts),
                            "mapping": mapping_records(keys, mapping),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                output_dir = generation_root / "cross_song" / str(job["id"]) / f"permutation_{permutation_id:03d}"
                manifest = output_dir / "manifest.json"
                if manifest.exists() and not force:
                    print(f"skip existing {manifest}", flush=True)
                    continue
                generate_conditioned_evaluation(
                    cfg=cfg,
                    checkpoint=checkpoint,
                    split=str(evaluation["split"]),
                    subjects=subjects,
                    output_dir=output_dir,
                    evaluation_mode="cross_song",
                    mapping=mapping,
                    job_metadata=metadata,
                    num_inference_steps=int(evaluation["num_inference_steps"]),
                    guidance_scale=float(evaluation["guidance_scale"]),
                    eta=float(evaluation.get("eta", 0.0)),
                    generation_seed=int(evaluation["generation_seed"]),
                    permutation_id=permutation_id,
                    permutation_seed=permutation_seed,
                    max_samples=max_samples,
                    resources=resources,
                )
        else:  # temporal_shift
            temporal_cfg = config["temporal_shift"]
            offsets = [int(value) for value in temporal_cfg["offsets"] if int(value) != 0]
            if max_permutations is not None:
                offsets = offsets[: max(0, int(max_permutations))]
            mapping_dir = generation_root / "temporal_shift_mappings" / str(job["id"])
            mapping_dir.mkdir(parents=True, exist_ok=True)
            chunk_sec = float(evaluation["chunk_sec"])
            for offset in offsets:
                mapping, excluded = build_temporal_shift_mapping(keys, offset=offset)
                mapping_path = mapping_dir / f"offset_{offset:+03d}.json"
                mapping_path.write_text(
                    json.dumps(
                        {
                            "job_id": job["id"],
                            "model_type": job["model_type"],
                            "regime": job["regime"],
                            "offset_chunks": offset,
                            "offset_seconds": offset * chunk_sec,
                            "num_valid_targets": len(mapping),
                            "num_excluded_targets": len(excluded),
                            "excluded": excluded,
                            "mapping": mapping_records(keys, mapping),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                output_dir = generation_root / "temporal_shift" / str(job["id"]) / f"offset_{offset:+03d}"
                manifest = output_dir / "manifest.json"
                if manifest.exists() and not force:
                    print(f"skip existing {manifest}", flush=True)
                    continue
                generate_conditioned_evaluation(
                    cfg=cfg,
                    checkpoint=checkpoint,
                    split=str(evaluation["split"]),
                    subjects=subjects,
                    output_dir=output_dir,
                    evaluation_mode="temporal_shift",
                    mapping=mapping,
                    job_metadata=metadata,
                    num_inference_steps=int(evaluation["num_inference_steps"]),
                    guidance_scale=float(evaluation["guidance_scale"]),
                    eta=float(evaluation.get("eta", 0.0)),
                    generation_seed=int(evaluation["generation_seed"]),
                    max_samples=max_samples,
                    resources=resources,
                    temporal_offset_chunks=offset,
                )
        del resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def manifest_paths(config: Mapping[str, Any]) -> list[Path]:
    root = run_dir(config)
    paths = []
    for mode in ("correct", "shuffled", "cross_song", "temporal_shift", "pretrained"):
        mode_root = root / mode
        if mode_root.exists():
            paths.extend(sorted(mode_root.rglob("manifest.json")))
    return paths


def score_stage(config: Mapping[str, Any], *, force: bool) -> None:
    manifests = manifest_paths(config)
    if not manifests:
        raise FileNotFoundError(f"No generation manifests under {run_dir(config)}")
    metrics = run_dir(config) / "metrics"
    overall_dir = metrics / "overall"
    if not (overall_dir / "per_sample_scores.json").exists() or force:
        evaluate_rows(load_rows([str(path) for path in manifests]), output_dir=overall_dir)

    stems_root = Path(config["paths"]["stems_root"])
    stem_names = [str(config["stems"][key]) for key in ("drum", "guitar_proxy_other_bass", "vocal")]
    stem_dir = metrics / "stems"
    if not (stem_dir / "per_sample_stem_scores.json").exists() or force:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/evaluate_stem_retrieval.py"),
                "--manifest",
                *[str(path) for path in manifests],
                "--stems-root",
                str(stems_root),
                "--stems",
                *stem_names,
                "--score-only",
                "--output-dir",
                str(stem_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )


def record_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("evaluation_mode"),
        row.get("job_id"),
        row.get("permutation_id"),
        row.get("target_song_id"),
        row.get("target_subject_id"),
        row.get("target_chunk_id"),
        row.get("generation_seed"),
    )


def summarize_stage(config: Mapping[str, Any]) -> None:
    metrics = run_dir(config) / "metrics"
    overall_path = metrics / "overall/per_sample_scores.json"
    stem_path = metrics / "stems/per_sample_stem_scores.json"
    if not overall_path.exists():
        raise FileNotFoundError(overall_path)
    records = json.loads(overall_path.read_text(encoding="utf-8"))
    if stem_path.exists():
        stem_rows = json.loads(stem_path.read_text(encoding="utf-8"))
        stems_by_key = {record_identity(row): row for row in stem_rows}
        for row in records:
            stem = stems_by_key.get(record_identity(row))
            if stem is None:
                continue
            row["drum_clap"] = stem.get("drums_clap")
            row["guitar_proxy_other_bass_clap"] = stem.get("guitar_proxy_other_bass_clap")
            row["vocal_clap"] = stem.get("vocals_clap")
    bootstrap = config["bootstrap"]
    main_records = [row for row in records if row.get("evaluation_mode") != "temporal_shift"]
    summary = summarize_paper_records(
        main_records,
        num_resamples=int(bootstrap["num_resamples"]),
        seed=int(bootstrap["seed"]),
    )
    gate_path = metrics / "correct_regression_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(
            f"Missing legacy preprocessing regression gate: {gate_path}. "
            "Run --stage verify-regression before summarize."
        )
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    summary["regression_gates"] = gates
    write_paper_outputs(main_records, summary, output_dir=metrics)
    (metrics / "regression_gate.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8")

    temporal_cfg = config.get("temporal_shift", {})
    if bool(temporal_cfg.get("enabled", False)):
        temporal_input = [
            row for row in records if row.get("evaluation_mode") in ("temporal_shift", "correct", "shuffled")
        ]
        temporal_summary = summarize_temporal_shift(
            temporal_input,
            offsets=[int(value) for value in temporal_cfg["offsets"]],
            chunk_sec=float(config["evaluation"]["chunk_sec"]),
            include_random_within_song=bool(temporal_cfg.get("include_random_within_song", True)),
            analysis_mode=str(temporal_cfg.get("analysis_mode", "common_support")),
            num_resamples=int(bootstrap["num_resamples"]),
            seed=int(bootstrap["seed"]) + 1000,
        )
        write_temporal_shift_outputs(temporal_summary, output_dir=metrics / "temporal_shift")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_config(config)
    jobs = selected_jobs(config, args.jobs)
    if args.stage == "preflight":
        report = preflight(config, jobs, hash_checkpoints=bool(args.hash_checkpoints))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if not report["ok"]:
            raise SystemExit(2)
    elif args.stage == "prepare-data":
        prepare_data(config, force=bool(args.force))
    elif args.stage == "prepare-stems":
        prepare_stems(config, force=bool(args.force))
    elif args.stage == "legacy-regression":
        legacy_regression_stage(
            config,
            jobs,
            force=bool(args.force),
            max_samples=args.max_samples,
        )
    elif args.stage == "verify-regression":
        gates = verify_correct_regressions(config, jobs)
        print(json.dumps(gates, ensure_ascii=False, indent=2), flush=True)
    elif args.stage == "dry-run":
        report = dry_run_stage(config, jobs)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    elif args.stage == "generate":
        if args.mode is None:
            raise SystemExit("--mode is required for --stage generate.")
        if args.mode == "pretrained":
            jobs = []
        generate_stage(
            config,
            jobs,
            mode=args.mode,
            max_samples=args.max_samples if args.max_samples is not None else config["evaluation"].get("max_samples"),
            max_permutations=args.max_permutations,
            force=bool(args.force),
        )
    elif args.stage == "score":
        score_stage(config, force=bool(args.force))
    elif args.stage == "summarize":
        summarize_stage(config)


if __name__ == "__main__":
    main()
