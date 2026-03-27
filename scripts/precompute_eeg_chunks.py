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
import scipy.io
import soundfile as sf
import yaml

from utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_song_specs(
    *,
    songs: object,
) -> list[dict]:
    if not isinstance(songs, list) or len(songs) == 0:
        raise ValueError("configs/train.yaml must define a non-empty data.songs list.")

    normalized = []
    for idx, song in enumerate(songs):
        if not isinstance(song, dict):
            raise TypeError(f"data.songs[{idx}] must be a mapping, got {type(song)}")
        if "audio_path" not in song:
            raise KeyError(f"data.songs[{idx}] is missing 'audio_path'")
        if "mat_path" not in song:
            raise KeyError(f"data.songs[{idx}] is missing 'mat_path'")
        if "data_key" not in song:
            raise KeyError(f"data.songs[{idx}] is missing 'data_key'")
        normalized.append(
            {
                "name": str(song.get("name", f"song_{idx:02d}")),
                "mat_path": str(song["mat_path"]),
                "audio_path": str(song["audio_path"]),
                "data_key": str(song["data_key"]),
            }
        )
    return normalized


def load_eeg_array(
    *,
    source_name: str,
    mat_path: str,
    data_key: str,
) -> np.ndarray:
    mat = scipy.io.loadmat(mat_path)
    if data_key not in mat:
        raise KeyError(f"{source_name}: '{data_key}' not found in {mat_path}. keys={list(mat.keys())}")
    eeg = np.asarray(mat[data_key], dtype=np.float32)
    if eeg.ndim != 3:
        raise ValueError(f"{source_name}: expected EEG shape [C,T,S], got {eeg.shape}")
    return eeg


def main() -> None:
    cfg = load_config("configs/train.yaml")
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    out_dir = Path(data_cfg.get("eeg_chunk_cache_dir", "data/precomputed/eeg_chunks"))
    out_dir.mkdir(parents=True, exist_ok=True)

    required_sources = ["passive"]
    chunk_sec = float(data_cfg["chunk_sec"])
    eeg_fs = int(data_cfg["eeg_fs"])
    audio_fs = int(data_cfg["audio_fs"])
    eeg_chunk_len = int(chunk_sec * eeg_fs)
    audio_chunk_len = int(chunk_sec * audio_fs)

    song_specs = load_song_specs(songs=data_cfg.get("songs"))

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
            eeg = load_eeg_array(
                source_name=source_name,
                mat_path=str(mat_path),
                data_key=str(data_key),
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
                    chunk = eeg[:, st:ed, subj_idx]
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
                "mat_path": str(mat_path),
                "data_key": str(data_key),
            }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved EEG chunk cache manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
