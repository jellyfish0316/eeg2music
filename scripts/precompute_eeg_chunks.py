from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import librosa
import numpy as np
import soundfile as sf
import yaml

from datasets.condition_nmedt_dataset import _load_eeg_source
from datasets.eeg_preprocessing import apply_chunk_level_eeg_preprocessing
from utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_song_specs(
    *,
    songs: list[dict] | None,
    mat_path: str | None,
    audio_path: str | None,
    data_key: str,
    condition_sources: dict | None,
) -> list[dict]:
    if songs:
        normalized = []
        for idx, song in enumerate(songs):
            if "audio_path" not in song:
                raise KeyError(f"data.songs[{idx}] is missing 'audio_path'")
            per_song_condition_sources = song.get("condition_sources")
            if per_song_condition_sources is None and "mat_path" not in song:
                per_song_condition_sources = condition_sources
            normalized.append(
                {
                    "name": str(song.get("name", f"song_{idx:02d}")),
                    "mat_path": song.get("mat_path", mat_path),
                    "audio_path": song["audio_path"],
                    "data_key": song.get("data_key", data_key),
                    "condition_sources": per_song_condition_sources,
                }
            )
        return normalized

    if mat_path is None or audio_path is None:
        raise ValueError("Single-song dataset requires both mat_path and audio_path.")
    return [
        {
            "name": Path(audio_path).stem,
            "mat_path": mat_path,
            "audio_path": audio_path,
            "data_key": data_key,
            "condition_sources": condition_sources,
        }
    ]


def resolve_required_sources(
    *,
    conditions: list[str],
    active_instruments: list[str],
    target_instrument: str | None = None,
) -> list[str]:
    required: list[str] = []

    def add(name: str) -> None:
        if name not in required:
            required.append(name)

    for condition in conditions:
        if condition == "passive_x3":
            add("passive")
        elif condition == "multi_attention":
            if len(active_instruments) < 3:
                raise ValueError("multi_attention requires at least 3 active instruments.")
            for inst in active_instruments[:3]:
                add(inst)
        elif condition == "single_repeated":
            if target_instrument is None:
                raise ValueError("single_repeated requires experiment.target_instrument.")
            add(target_instrument)
        else:
            raise ValueError(f"Unsupported condition type for EEG chunk cache: {condition}")

    return required


def main() -> None:
    cfg = load_config("configs/train.yaml")
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    experiment_cfg = cfg.get("experiment", {})
    out_dir = Path(data_cfg.get("eeg_chunk_cache_dir", "data/precomputed/eeg_chunks"))
    out_dir.mkdir(parents=True, exist_ok=True)

    active_instruments = list(experiment_cfg.get("active_instruments", ["drum", "guitar", "vocal"]))
    conditions = list(experiment_cfg.get("conditions", ["passive_x3"]))
    required_sources = resolve_required_sources(
        conditions=conditions,
        active_instruments=active_instruments,
        target_instrument=experiment_cfg.get("target_instrument"),
    )
    chunk_sec = float(data_cfg["chunk_sec"])
    eeg_fs = int(data_cfg["eeg_fs"])
    audio_fs = int(data_cfg["audio_fs"])
    eeg_chunk_len = int(chunk_sec * eeg_fs)
    audio_chunk_len = int(chunk_sec * audio_fs)
    eeg_preprocessing = {
        "drop_face_channels": True,
        "keep_first_n_channels": 124,
        "robust_scaler": True,
        "center_using_first_samples": 1000,
        "std_clamp": 20.0,
        "per_channel_normalization": bool(data_cfg.get("normalize_eeg", True)),
    }

    song_specs = normalize_song_specs(
        songs=data_cfg.get("songs"),
        mat_path=data_cfg.get("mat_path"),
        audio_path=data_cfg.get("audio_path"),
        data_key=data_cfg.get("data_key", "data21"),
        condition_sources=data_cfg.get("condition_sources"),
    )

    manifest: dict[str, object] = {
        "meta": {
            "chunk_sec": chunk_sec,
            "eeg_fs": eeg_fs,
            "eeg_chunk_len": eeg_chunk_len,
            "subject_indices": None,
            "required_sources": required_sources,
        },
        "songs": {},
    }

    for idx, spec in enumerate(song_specs):
        song_name = str(spec.get("name", f"song_{idx:02d}"))
        mat_path = spec.get("mat_path")
        audio_path = spec.get("audio_path")
        data_key = spec.get("data_key", "data21")
        if mat_path is None or audio_path is None:
            raise ValueError(f"Song {song_name!r} missing mat_path/audio_path.")

        source_specs = dict(spec.get("condition_sources") or {})
        if "passive" not in source_specs:
            source_specs["passive"] = {"mat_path": mat_path, "data_key": data_key}
        for inst in active_instruments:
            if inst not in source_specs:
                source_specs[inst] = {"mat_path": mat_path, "data_key": data_key}

        audio, sr = sf.read(str(audio_path))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != audio_fs:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=audio_fs)
        n_chunks_audio = int(len(audio) // audio_chunk_len)

        manifest["songs"][song_name] = {"sources": {}, "n_chunks_audio": n_chunks_audio}
        print(f"precompute EEG chunks: song={song_name} n_chunks_audio={n_chunks_audio}", flush=True)

        for source_name in required_sources:
            source_spec = source_specs[source_name]
            src_mat_path = str(source_spec.get("mat_path", mat_path))
            src_data_key = str(source_spec.get("data_key", data_key))
            eeg = _load_eeg_source(
                name=source_name,
                mat_path=src_mat_path,
                data_key=src_data_key,
                preprocessing=eeg_preprocessing,
            )
            subject_indices = list(range(int(eeg.shape[2])))
            n_chunks_eeg = int(eeg.shape[1] // eeg_chunk_len)
            total_chunks = min(n_chunks_audio, n_chunks_eeg)
            if total_chunks <= 0:
                raise ValueError(f"No usable chunks for {song_name}/{source_name}")

            out_path = out_dir / f"{song_name}_{source_name}.npy"
            chunk_array = np.lib.format.open_memmap(
                out_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(subject_indices), total_chunks, eeg.shape[0], eeg_chunk_len),
            )

            for local_subj_idx, subj_idx in enumerate(subject_indices):
                for chunk_idx in range(total_chunks):
                    st = chunk_idx * eeg_chunk_len
                    ed = st + eeg_chunk_len
                    chunk = eeg[:, st:ed, subj_idx].copy()
                    chunk = apply_chunk_level_eeg_preprocessing(
                        chunk,
                        eeg_preprocessing,
                        fallback_normalize=True,
                    )
                    chunk_array[local_subj_idx, chunk_idx] = chunk.astype(np.float32, copy=False)
                print(
                    f"cached EEG song={song_name} source={source_name} subject={subj_idx} "
                    f"chunks={total_chunks} shape={tuple(chunk_array.shape[2:])}",
                    flush=True,
                )

            del chunk_array
            manifest["meta"]["subject_indices"] = subject_indices
            manifest["songs"][song_name]["sources"][source_name] = {
                "path": os.path.relpath(out_path, out_dir),
                "shape": [len(subject_indices), total_chunks, int(eeg.shape[0]), eeg_chunk_len],
                "subject_indices": subject_indices,
                "mat_path": src_mat_path,
                "data_key": src_data_key,
            }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved EEG chunk cache manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
