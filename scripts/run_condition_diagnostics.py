from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile as sf
import torch
import yaml

from scripts.generate import prepare_conditioned_evaluation_resources
from scripts.run_paper_evaluation import (
    job_config,
    load_yaml,
    resolve_checkpoint,
    selected_jobs,
    stage_checkpoint,
)
from scripts.train import build_dataloader
from utils.evaluation_analysis import paired_bootstrap
from utils.evaluation_pairing import build_cyclic_derangements, keys_from_index_map, stable_generation_seed
from utils.generation import generate_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run condition-usage diagnostics before full Table 1 expansion.")
    parser.add_argument("--config", default="configs/diagnose_condition_usage.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one target per diagnostic with two denoising steps.")
    return parser.parse_args()


def rms(tensor: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(tensor.detach().float().square())).cpu().item())


class InstrumentationCollector:
    def __init__(self, control_scale: float) -> None:
        self.control_scale = float(control_scale)
        self.projected_rms: list[float] = []
        self.eps_rms: list[float] = []
        self.layer_rms: dict[str, list[float]] = {}

    def __call__(self, step_index: int, timestep: torch.Tensor, pred: Mapping[str, Any]) -> None:
        del step_index, timestep
        projected = pred.get("projected_latent")
        eps_pred = pred.get("eps_pred")
        if torch.is_tensor(projected):
            self.projected_rms.append(rms(projected))
        if torch.is_tensor(eps_pred):
            self.eps_rms.append(rms(eps_pred))
        residuals = pred.get("control_residuals")
        if not isinstance(residuals, Mapping):
            return
        for index, value in enumerate(residuals.get("down_block_residuals", ())):
            self.layer_rms.setdefault(f"down_{index:02d}", []).append(rms(value))
        mid = residuals.get("mid_block_residual")
        if torch.is_tensor(mid):
            self.layer_rms.setdefault("mid", []).append(rms(mid))

    def summary(self) -> dict[str, Any]:
        def stats(values: Sequence[float]) -> dict[str, float] | None:
            if not values:
                return None
            return {"mean_rms": float(np.mean(values)), "max_rms": float(np.max(values))}

        layers = {name: stats(values) for name, values in sorted(self.layer_rms.items())}
        all_residuals = [value for values in self.layer_rms.values() for value in values]
        return {
            "projected_latent": stats(self.projected_rms),
            "eps_prediction": stats(self.eps_rms),
            "control_residual": stats(all_residuals),
            "scaled_control_residual_mean_rms": None
            if not all_residuals
            else float(np.mean(all_residuals) * self.control_scale),
            "injection_layers": layers,
            "num_diffusion_steps_observed": len(self.projected_rms),
        }


def dataset_lookup(dataset: Any) -> dict[tuple[str, int, int], int]:
    lookup: dict[tuple[str, int, int], int] = {}
    for dataset_index, (song_index, subject_index, chunk_index) in enumerate(dataset.index_map):
        song_name = str(dataset.song_records[int(song_index)].name)
        lookup[(song_name, int(subject_index), int(chunk_index))] = int(dataset_index)
    return lookup


def absolute_chunk(dataset: Any, dataset_index: int) -> int:
    song_index, _subject_index, chunk_index = dataset.index_map[int(dataset_index)]
    return int(dataset.song_records[int(song_index)].chunk_offset) + int(chunk_index)


def mean_eeg_prototypes(dataset: Any, subjects: Sequence[int]) -> dict[int, torch.Tensor]:
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {int(subject): 0 for subject in subjects}
    for dataset_index, (_song, subject, _chunk) in enumerate(dataset.index_map):
        subject = int(subject)
        if subject not in counts:
            continue
        eeg = dataset[dataset_index]["eeg"].float()
        sums[subject] = eeg.clone() if subject not in sums else sums[subject].add_(eeg)
        counts[subject] += 1
    missing = [subject for subject, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"No prototype rows for subjects {missing}")
    return {subject: sums[subject].div(float(counts[subject])) for subject in counts}


def tensor_distance(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.detach().float().cpu()
    reference = reference.detach().float().cpu()
    difference = candidate - reference
    denominator = float(torch.linalg.vector_norm(reference).item()) + 1e-12
    return {
        "mse": float(torch.mean(difference.square()).item()),
        "relative_l2": float(torch.linalg.vector_norm(difference).item() / denominator),
    }


def clap_cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    left = torch.nn.functional.normalize(left.detach().float().cpu(), dim=-1)
    right = torch.nn.functional.normalize(right.detach().float().cpu(), dim=-1)
    similarity = float((left * right).sum(dim=-1)[0].item())
    return float(max(0.0, 1.0 - min(1.0, similarity)))


def generate_variant(
    *,
    resources: Any,
    eeg: torch.Tensor,
    subject_index: int,
    target_audio: torch.Tensor,
    target_song: str,
    target_chunk: int,
    mode: str,
    output_dir: Path,
    generation: Mapping[str, Any],
    use_control: bool,
    use_subject_adapter: bool | None,
) -> dict[str, Any]:
    device = resources.device
    sample_seed = stable_generation_seed(
        base_seed=int(generation["generation_seed"]),
        song_id=target_song,
        chunk_index=int(target_chunk),
    )
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    collector = InstrumentationCollector(control_scale=1.0)
    eeg_batch = eeg.unsqueeze(0).to(device)
    subject = torch.tensor([int(subject_index)], device=device, dtype=torch.long)
    latents = generate_latents(
        resources.model,
        eeg=eeg_batch,
        subject_idx=subject,
        num_inference_steps=int(generation["num_inference_steps"]),
        eta=float(generation.get("eta", 0.0)),
        generator=generator,
        use_control=bool(use_control),
        control_scale=1.0,
        guidance_scale=float(generation["guidance_scale"]),
        use_subject_adapter=use_subject_adapter,
        diagnostic_callback=collector if use_control else None,
    )
    waveform = resources.decoder.decode_latents_to_waveform(latents).detach().float().cpu()
    mel = resources.decoder.decode_latents_to_mel(latents).detach().float().cpu()
    clap = resources.decoder.get_audio_features(
        waveform,
        sample_rate=int(resources.decoder.vocoder_sample_rate),
        normalize=True,
    )
    target_clap = resources.decoder.get_audio_features(
        target_audio.unsqueeze(0),
        sample_rate=16000,
        normalize=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{mode}_{target_song}_subj{int(subject_index):02d}_"
        f"chunk{int(target_chunk):04d}_seed{sample_seed}.wav"
    )
    path = output_dir / filename
    sf.write(path, waveform[0].numpy(), int(resources.decoder.vocoder_sample_rate))
    return {
        "mode": mode,
        "generation_seed": sample_seed,
        "generated_wav": str(path),
        "target_clap": float((clap * target_clap).sum(dim=-1)[0].item()),
        "latent": latents.detach().float().cpu(),
        "mel": mel,
        "clap": clap,
        "instrumentation": collector.summary(),
        "use_control": bool(use_control),
        "use_subject_adapter": resources.model.use_subject_adapter
        if use_subject_adapter is None
        else bool(use_subject_adapter),
    }


def finalize_target_rows(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    base: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = artifacts["correct"]
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for mode, artifact in artifacts.items():
        latent_distance = tensor_distance(artifact["latent"], reference["latent"])
        mel_distance = tensor_distance(artifact["mel"], reference["mel"])
        clap_distance = clap_cosine_distance(artifact["clap"], reference["clap"])
        rows.append(
            {
                **base,
                "mode": mode,
                "generation_seed": artifact["generation_seed"],
                "generated_wav": artifact["generated_wav"],
                "overall_clap": artifact["target_clap"],
                "latent_mse_vs_correct": latent_distance["mse"],
                "latent_relative_l2_vs_correct": latent_distance["relative_l2"],
                "mel_mse_vs_correct": mel_distance["mse"],
                "mel_relative_l2_vs_correct": mel_distance["relative_l2"],
                "generated_clap_cosine_distance_vs_correct": clap_distance,
                "instrumentation": artifact["instrumentation"],
                "use_control": artifact["use_control"],
                "use_subject_adapter": artifact["use_subject_adapter"],
            }
        )
    modes = sorted(artifacts)
    for left_index, left in enumerate(modes):
        for right in modes[left_index + 1 :]:
            pairs.append(
                {
                    **base,
                    "left_mode": left,
                    "right_mode": right,
                    "clap_cosine_distance": clap_cosine_distance(
                        artifacts[left]["clap"], artifacts[right]["clap"]
                    ),
                    "latent": tensor_distance(artifacts[left]["latent"], artifacts[right]["latent"]),
                    "mel": tensor_distance(artifacts[left]["mel"], artifacts[right]["mel"]),
                }
            )
    return rows, pairs


def aggregate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = config["bootstrap"]
    thresholds = config["decision_gate"]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["job_id"]), str(row["experiment"])), []).append(row)
    summaries = []
    evidence = {"cross_song_drop": False, "condition_sensitivity": False, "adapter_drop": False}
    for (job_id, experiment), group in sorted(grouped.items()):
        by_target: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
        for row in group:
            key = (row["target_song_id"], row["target_subject_id"], row["target_chunk_id"])
            by_target.setdefault(key, {})[str(row["mode"])] = row
        modes = sorted({str(row["mode"]) for row in group})
        mode_summary = {}
        for mode in modes:
            mode_rows = [row for row in group if row["mode"] == mode]
            mode_summary[mode] = {
                "n": len(mode_rows),
                "mean_overall_clap": float(np.mean([float(row["overall_clap"]) for row in mode_rows])),
                "mean_generated_clap_distance_vs_correct": float(
                    np.mean([float(row["generated_clap_cosine_distance_vs_correct"]) for row in mode_rows])
                ),
                "mean_latent_relative_l2_vs_correct": float(
                    np.mean([float(row["latent_relative_l2_vs_correct"]) for row in mode_rows])
                ),
                "mean_mel_relative_l2_vs_correct": float(
                    np.mean([float(row["mel_relative_l2_vs_correct"]) for row in mode_rows])
                ),
            }
        comparisons = {}
        for mode in modes:
            if mode == "correct":
                continue
            differences = [
                float(values["correct"]["overall_clap"]) - float(values[mode]["overall_clap"])
                for values in by_target.values()
                if "correct" in values and mode in values
            ]
            if differences:
                comparisons[f"correct_minus_{mode}"] = paired_bootstrap(
                    differences,
                    num_resamples=int(bootstrap["num_resamples"]),
                    seed=int(bootstrap["seed"]) + len(comparisons),
                )
        summaries.append(
            {"job_id": job_id, "experiment": experiment, "modes": mode_summary, "comparisons": comparisons}
        )
        if experiment == "cross_song":
            comparison = comparisons.get("correct_minus_cross_song_same_subject")
            if comparison and float(comparison["ci95_low"]) > 0 and float(comparison["mean_difference"]) >= float(
                thresholds["minimum_mean_clap_drop"]
            ):
                evidence["cross_song_drop"] = True
        for sensitive_mode in ("within_song_shuffled", "fixed_same_subject", "cross_song_same_subject"):
            value = mode_summary.get(sensitive_mode, {}).get("mean_generated_clap_distance_vs_correct")
            if value is not None and float(value) >= float(thresholds["minimum_generated_clap_distance"]):
                evidence["condition_sensitivity"] = True
        adapter = comparisons.get("correct_minus_adapter_off")
        if adapter and float(adapter["ci95_low"]) > 0 and float(adapter["mean_difference"]) >= float(
            thresholds["minimum_mean_clap_drop"]
        ):
            evidence["adapter_drop"] = True
    if any(evidence.values()):
        recommendation = "continue_representative_jobs_first"
    else:
        recommendation = "pause_full_expansion_and_audit_condition_injection"
    return {"experiments": summaries, "decision_evidence": evidence, "recommendation": recommendation}


def run_job(
    diagnostic: Mapping[str, Any],
    paper: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = job_config(paper, job)
    staged_checkpoint = Path(paper["paths"]["checkpoint_staging_dir"]) / (
        f"{job['id']}{Path(str(job['checkpoint'])).suffix}"
    )
    staged_checksum = staged_checkpoint.with_suffix(staged_checkpoint.suffix + ".sha256")
    if staged_checkpoint.exists() and staged_checksum.exists():
        checkpoint = staged_checkpoint
    else:
        checkpoint, _checkpoint_meta = stage_checkpoint(paper, job)
    generation = diagnostic["generation"]
    all_rows: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []

    ood = diagnostic["ood_ablation"]
    resources = prepare_conditioned_evaluation_resources(
        cfg=cfg,
        checkpoint=checkpoint,
        split=str(ood["split"]),
        subjects=[int(value) for value in ood["subjects"]],
    )
    target_dataset = resources.dataset
    lookup = dataset_lookup(target_dataset)
    keys = keys_from_index_map(target_dataset.index_map)
    shuffle_mapping = build_cyclic_derangements(keys, num_permutations=1, seed=0, min_chunk_distance=5)[0]
    prototype_dataset, _ = build_dataloader(
        cfg,
        subjects=[int(value) for value in ood["subjects"]],
        shuffle=False,
        chunk_split_name=str(ood["prototype_split"]),
    )
    prototypes = mean_eeg_prototypes(prototype_dataset, [int(value) for value in ood["subjects"]])
    for subject in [int(value) for value in ood["subjects"]]:
        fixed_index = lookup[(str(ood["song"]), subject, int(ood["fixed_source_chunk"]))]
        fixed_eeg = target_dataset[fixed_index]["eeg"]
        for chunk in [int(value) for value in ood["chunks"]]:
            target_index = lookup[(str(ood["song"]), subject, chunk)]
            target = target_dataset[target_index]
            shuffled = target_dataset[shuffle_mapping[target_index]]
            conditions = {
                "correct": (target["eeg"], True, None, target_index),
                "within_song_shuffled": (shuffled["eeg"], True, None, shuffle_mapping[target_index]),
                "fixed_same_subject": (fixed_eeg, True, None, fixed_index),
                "dataset_prototype": (prototypes[subject], True, None, None),
                "subject_adapter_off": (target["eeg"], True, False, target_index),
                "adapter_off": (target["eeg"], False, None, target_index),
                "zero_eeg": (torch.zeros_like(target["eeg"]), True, None, None),
            }
            artifacts = {}
            source_meta = {}
            absolute = absolute_chunk(target_dataset, target_index)
            for mode in ood["modes"]:
                eeg, use_control, use_subject_adapter, source_index = conditions[str(mode)]
                artifacts[str(mode)] = generate_variant(
                    resources=resources,
                    eeg=eeg,
                    subject_index=subject,
                    target_audio=target["audio"],
                    target_song=str(ood["song"]),
                    target_chunk=absolute,
                    mode=str(mode),
                    output_dir=output_root / str(job["id"]) / "ood_ablation" / "generated",
                    generation=generation,
                    use_control=use_control,
                    use_subject_adapter=use_subject_adapter,
                )
                source_meta[str(mode)] = None if source_index is None else {
                    "dataset_index": int(source_index),
                    "song": str(target_dataset.song_records[target_dataset.index_map[source_index][0]].name),
                    "subject": int(target_dataset.index_map[source_index][1]),
                    "local_chunk": int(target_dataset.index_map[source_index][2]),
                    "absolute_chunk": absolute_chunk(target_dataset, source_index),
                }
            base = {
                "job_id": str(job["id"]),
                "model_type": str(job["model_type"]),
                "experiment": "ood_ablation",
                "split": str(ood["split"]),
                "target_song_id": str(ood["song"]),
                "target_subject_id": subject,
                "target_chunk_id": absolute,
                "target_local_chunk_id": chunk,
                "condition_sources": cfg["data"]["condition_sources"],
                "source_metadata": source_meta,
            }
            rows, pairs = finalize_target_rows(artifacts, base=base)
            all_rows.extend(rows)
            all_pairs.extend(pairs)
    del prototype_dataset, prototypes

    cross = diagnostic["cross_song"]
    cross_dataset, _ = build_dataloader(
        cfg,
        subjects=[int(value) for value in cross["subjects"]],
        shuffle=False,
        chunk_split_name=str(cross["split"]),
    )
    resources.dataset = cross_dataset
    dataset = cross_dataset
    lookup = dataset_lookup(dataset)
    available_chunks = len(
        [key for key in lookup if key[0] == str(cross["target_song"]) and key[1] == int(cross["subjects"][0])]
    )
    for subject in [int(value) for value in cross["subjects"]]:
        for chunk in [int(value) for value in cross["local_chunks"]]:
            target_index = lookup[(str(cross["target_song"]), subject, chunk)]
            within_chunk = (chunk + int(cross["within_song_shift"])) % available_chunks
            within_index = lookup[(str(cross["target_song"]), subject, within_chunk)]
            cross_index = lookup[(str(cross["source_song"]), subject, chunk)]
            target = dataset[target_index]
            conditions = {
                "correct": (target["eeg"], target_index),
                "within_song_shuffled": (dataset[within_index]["eeg"], within_index),
                "cross_song_same_subject": (dataset[cross_index]["eeg"], cross_index),
            }
            artifacts = {}
            source_meta = {}
            absolute = absolute_chunk(dataset, target_index)
            for mode in cross["modes"]:
                eeg, source_index = conditions[str(mode)]
                artifacts[str(mode)] = generate_variant(
                    resources=resources,
                    eeg=eeg,
                    subject_index=subject,
                    target_audio=target["audio"],
                    target_song=str(cross["target_song"]),
                    target_chunk=absolute,
                    mode=str(mode),
                    output_dir=output_root / str(job["id"]) / "cross_song" / "generated",
                    generation=generation,
                    use_control=True,
                    use_subject_adapter=None,
                )
                source_meta[str(mode)] = {
                    "dataset_index": source_index,
                    "song": str(dataset.song_records[dataset.index_map[source_index][0]].name),
                    "subject": int(dataset.index_map[source_index][1]),
                    "local_chunk": int(dataset.index_map[source_index][2]),
                    "absolute_chunk": absolute_chunk(dataset, source_index),
                }
            base = {
                "job_id": str(job["id"]),
                "model_type": str(job["model_type"]),
                "experiment": "cross_song",
                "split": str(cross["split"]),
                "target_song_id": str(cross["target_song"]),
                "target_subject_id": subject,
                "target_chunk_id": absolute,
                "target_local_chunk_id": chunk,
                "condition_sources": cfg["data"]["condition_sources"],
                "source_metadata": source_meta,
            }
            rows, pairs = finalize_target_rows(artifacts, base=base)
            all_rows.extend(rows)
            all_pairs.extend(pairs)
    del resources
    gc.collect()
    torch.cuda.empty_cache()
    return all_rows, all_pairs


def write_outputs(root: Path, rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "per_condition_results.json").write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "pairwise_condition_distances.json").write_text(
        json.dumps(list(pairs), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "summary.json").write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    flat_rows = []
    for row in rows:
        flat_rows.append({key: value for key, value in row.items() if not isinstance(value, (dict, list))})
    if flat_rows:
        fields = sorted(set().union(*(row.keys() for row in flat_rows)))
        with (root / "per_condition_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)


def main() -> None:
    args = parse_args()
    diagnostic = copy.deepcopy(load_yaml(args.config))
    if args.smoke:
        diagnostic["run_id"] = f"{diagnostic['run_id']}_smoke"
        diagnostic["generation"]["num_inference_steps"] = 2
        diagnostic["ood_ablation"]["subjects"] = diagnostic["ood_ablation"]["subjects"][:1]
        diagnostic["ood_ablation"]["chunks"] = diagnostic["ood_ablation"]["chunks"][:1]
        diagnostic["cross_song"]["subjects"] = diagnostic["cross_song"]["subjects"][:1]
        diagnostic["cross_song"]["local_chunks"] = diagnostic["cross_song"]["local_chunks"][:1]
        diagnostic["bootstrap"]["num_resamples"] = 200
    paper = load_yaml(REPO_ROOT / str(diagnostic["paper_config"]))
    if not torch.cuda.is_available():
        raise RuntimeError("Condition diagnostics require CUDA.")
    jobs = selected_jobs(paper, [str(value) for value in diagnostic["jobs"]])
    root = Path(diagnostic["output_root"]) / str(diagnostic["run_id"])
    final_summary = root / "summary.json"
    if final_summary.exists() and not args.force:
        existing = json.loads(final_summary.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"skip completed diagnostics: {final_summary}", flush=True)
            return
    (root / "config_snapshot.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "config_snapshot.yaml").write_text(
        yaml.safe_dump(dict(diagnostic), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for job in jobs:
        job_rows, job_pairs = run_job(diagnostic, paper, job, output_root=root)
        rows.extend(job_rows)
        pairs.extend(job_pairs)
        write_outputs(root, rows, pairs, {"status": "in_progress", "completed_jobs": [r["job_id"] for r in rows]})
    summary = aggregate(rows, diagnostic)
    summary.update({"status": "complete", "num_rows": len(rows), "num_pairwise_rows": len(pairs)})
    write_outputs(root, rows, pairs, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
