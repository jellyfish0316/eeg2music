from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import scipy.io
import soundfile as sf
import torch
from torch.utils.data import Dataset


@dataclass
class EEGSource:
    name: str
    mat_path: str
    data_key: str
    shape: tuple[int, int, int]  # [C, T, S]
    chunk_cache_path: str | None = None
    chunk_cache_shape: tuple[int, int, int, int] | None = None  # [S, N, C, T]
    chunk_cache_subject_indices: tuple[int, ...] | None = None

    @property
    def n_channels(self) -> int:
        if self.chunk_cache_shape is not None:
            return int(self.chunk_cache_shape[2])
        return int(self.shape[0])

    @property
    def total_time(self) -> int:
        if self.chunk_cache_shape is not None:
            return int(self.chunk_cache_shape[1] * self.chunk_cache_shape[3])
        return int(self.shape[1])

    @property
    def total_subjects(self) -> int:
        if self.chunk_cache_subject_indices is not None:
            return int(len(self.chunk_cache_subject_indices))
        return int(self.shape[2])


@dataclass
class SongRecord:
    name: str
    audio: np.ndarray
    n_chunks: int
    sources: dict[str, EEGSource]
    chunk_offset: int = 0


def _load_eeg_source(
    name: str,
    mat_path: str,
    data_key: str,
) -> np.ndarray:
    mat = scipy.io.loadmat(mat_path)
    if data_key not in mat:
        raise KeyError(f"'{data_key}' not found in {mat_path}. keys={list(mat.keys())}")
    arr = mat[data_key].astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected EEG shape [C,T,S], got {arr.shape}")
    return arr


def _get_mat_shape(
    mat_path: str,
    data_key: str,
) -> tuple[int, int, int]:
    for key, shape, _dtype in scipy.io.whosmat(mat_path):
        if key != data_key:
            continue
        if len(shape) != 3:
            raise ValueError(f"{data_key}: expected EEG shape [C,T,S], got {shape}")
        return tuple(int(v) for v in shape)
    raise KeyError(f"'{data_key}' not found in {mat_path}.")


class ConditionNMEDTDataset(Dataset):
    """
    Passive-only EEG dataset.
    """

    CONDITION_TO_ID = {
        "passive": 0,
    }

    def __init__(
        self,
        *,
        condition_type: str,
        mat_path: str | None = None,
        audio_path: str | None = None,
        data_key: str = "data21",
        songs: list[dict[str, Any]] | None = None,
        chunk_sec: float = 3.5,
        eeg_fs: int = 125,
        audio_fs: int = 16000,
        subjects: list[int] | None = None,
        normalize_audio: bool = True,
        text_prompt: str = "Pop music",
        precomputed_latents_path: str | None = None,
        chunk_range: tuple[float, float] | None = None,
        eeg_chunk_cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        if condition_type != "passive":
            raise ValueError(f"Unsupported condition_type: {condition_type}. Only 'passive' is supported.")

        self.condition_type = condition_type
        self.chunk_sec = float(chunk_sec)
        self.eeg_fs = int(eeg_fs)
        self.audio_fs = int(audio_fs)
        self.eeg_chunk_len = int(self.chunk_sec * self.eeg_fs)
        self.audio_chunk_len = int(self.chunk_sec * self.audio_fs)
        self.normalize_audio = normalize_audio
        self.text_prompt = text_prompt
        self.precomputed_latents_path = precomputed_latents_path
        self.chunk_range = self._normalize_chunk_range(chunk_range)
        self._source_cache: dict[tuple[str, str], np.ndarray] = {}
        self._source_cache_order: list[tuple[str, str]] = []
        self._max_cached_sources = 1
        self._chunk_cache_arrays: dict[str, np.ndarray] = {}
        self._chunk_cache_manifest = self._load_eeg_chunk_cache_manifest(eeg_chunk_cache_dir)

        song_specs = self._normalize_song_specs(
            songs=songs,
            mat_path=mat_path,
            audio_path=audio_path,
            data_key=data_key,
        )
        required_sources = ["passive"]
        self.song_records = self._build_song_records(song_specs, required_sources=required_sources)
        first = self.song_records[0].sources[required_sources[0]]
        self.base_eeg_channels = first.n_channels
        self.total_subjects = first.total_subjects
        if subjects is None:
            self.subjects = list(range(self.total_subjects))
        else:
            for s in subjects:
                if s < 0 or s >= self.total_subjects:
                    raise ValueError(f"Invalid subject index {s}; total_subjects={self.total_subjects}")
            self.subjects = list(subjects)

        self.eeg_out_channels = self.base_eeg_channels

        self.index_map: list[tuple[int, int, int]] = []
        for song_idx, record in enumerate(self.song_records):
            for subj_idx in self.subjects:
                for chunk_idx in range(record.n_chunks):
                    self.index_map.append((song_idx, subj_idx, chunk_idx))

        self.n_chunks = sum(record.n_chunks for record in self.song_records)
        self.z0_by_chunk: torch.Tensor | None = None
        self.z0_by_song: list[torch.Tensor] | None = None
        self.latents_by_song: list[torch.Tensor] | None = None
        self.latent_shape: tuple[int, int, int] | None = None
        if precomputed_latents_path is not None:
            self._load_latent_cache(Path(precomputed_latents_path))

    @staticmethod
    def _normalize_chunk_range(chunk_range: tuple[float, float] | None) -> tuple[float, float]:
        if chunk_range is None:
            return (0.0, 1.0)
        start, end = float(chunk_range[0]), float(chunk_range[1])
        if not (0.0 <= start < end <= 1.0):
            raise ValueError(f"chunk_range must satisfy 0 <= start < end <= 1, got {chunk_range}")
        return (start, end)

    @staticmethod
    def _normalize_song_specs(
        *,
        songs: list[dict[str, Any]] | None,
        mat_path: str | None,
        audio_path: str | None,
        data_key: str,
    ) -> list[dict[str, Any]]:
        if songs:
            normalized = []
            for idx, song in enumerate(songs):
                if "audio_path" not in song:
                    raise KeyError(f"data.songs[{idx}] is missing 'audio_path'")
                normalized.append(
                    {
                        "name": str(song.get("name", f"song_{idx:02d}")),
                        "mat_path": song.get("mat_path", mat_path),
                        "audio_path": song["audio_path"],
                        "data_key": song.get("data_key", data_key),
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
            }
        ]

    @staticmethod
    def _load_eeg_chunk_cache_manifest(eeg_chunk_cache_dir: str | None) -> dict[str, Any] | None:
        if eeg_chunk_cache_dir is None:
            return None
        cache_dir = Path(eeg_chunk_cache_dir)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"eeg_chunk_cache_dir set but manifest not found: {manifest_path}")
        import json

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["_cache_dir"] = str(cache_dir.resolve())
        return manifest

    def _get_chunk_cache_entry(self, song_name: str, source_name: str) -> dict[str, Any] | None:
        if self._chunk_cache_manifest is None:
            return None
        songs = self._chunk_cache_manifest.get("songs", {})
        song_meta = songs.get(song_name)
        if not isinstance(song_meta, dict):
            return None
        sources = song_meta.get("sources", {})
        entry = sources.get(source_name)
        if not isinstance(entry, dict):
            return None
        return entry

    def _build_song_records(
        self,
        song_specs: list[dict[str, Any]],
        *,
        required_sources: list[str],
    ) -> list[SongRecord]:
        records: list[SongRecord] = []
        global_shape_cache: dict[tuple[str, str], tuple[int, int, int]] = {}

        for idx, spec in enumerate(song_specs):
            mat_path = spec.get("mat_path")
            audio_path = spec.get("audio_path")
            data_key = spec.get("data_key", "data21")
            if mat_path is None:
                raise ValueError(f"Song spec {spec.get('name', idx)!r} is missing mat_path.")

            sources: dict[str, EEGSource] = {}
            key = (str(mat_path), str(data_key))
            chunk_cache_entry = self._get_chunk_cache_entry(str(spec.get("name", f"song_{idx:02d}")), "passive")
            if key not in global_shape_cache:
                global_shape_cache[key] = _get_mat_shape(
                    mat_path=str(mat_path),
                    data_key=str(data_key),
                )
            sources["passive"] = EEGSource(
                name="passive",
                mat_path=str(mat_path),
                data_key=str(data_key),
                shape=global_shape_cache[key],
                chunk_cache_path=None
                if chunk_cache_entry is None
                else str(Path(self._chunk_cache_manifest["_cache_dir"]) / chunk_cache_entry["path"]),
                chunk_cache_shape=None
                if chunk_cache_entry is None
                else tuple(int(v) for v in chunk_cache_entry["shape"]),
                chunk_cache_subject_indices=None
                if chunk_cache_entry is None
                else tuple(int(v) for v in chunk_cache_entry["subject_indices"]),
            )

            first = sources[required_sources[0]]
            for name in required_sources[1:]:
                src = sources[name]
                if src.n_channels != first.n_channels:
                    raise ValueError(f"Channel mismatch in {spec.get('name', idx)!r}: {name}={src.n_channels} vs {first.n_channels}")
                if src.total_subjects != first.total_subjects:
                    raise ValueError(
                        f"Subject mismatch in {spec.get('name', idx)!r}: {name}={src.total_subjects} vs {first.total_subjects}"
                    )

            audio, sr = sf.read(str(audio_path))
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != self.audio_fs:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_fs)
            audio = audio.astype(np.float32)

            n_chunks_audio = len(audio) // self.audio_chunk_len
            n_chunks_eeg = min(sources[name].total_time // self.eeg_chunk_len for name in required_sources)
            total_chunks = int(min(n_chunks_audio, n_chunks_eeg))
            if total_chunks == 0:
                raise ValueError(
                    f"No usable chunks for {spec.get('name', idx)!r}. audio={n_chunks_audio}, eeg={n_chunks_eeg}"
                )
            start_frac, end_frac = self.chunk_range
            chunk_offset = int(np.floor(total_chunks * start_frac))
            chunk_stop = int(np.floor(total_chunks * end_frac))
            chunk_stop = max(chunk_offset + 1, min(total_chunks, chunk_stop))
            n_chunks = int(chunk_stop - chunk_offset)

            if records:
                ref = records[0].sources[required_sources[0]]
                if first.n_channels != ref.n_channels:
                    raise ValueError(f"Channel mismatch across songs: {spec.get('name', idx)!r}")
                if first.total_subjects != ref.total_subjects:
                    raise ValueError(f"Subject mismatch across songs: {spec.get('name', idx)!r}")

            records.append(
                SongRecord(
                    name=str(spec.get("name", f"song_{idx:02d}")),
                    audio=audio,
                    n_chunks=n_chunks,
                    sources=sources,
                    chunk_offset=chunk_offset,
                )
            )
        return records

    def _load_latent_cache(self, latent_path: Path) -> None:
        if not latent_path.exists():
            raise FileNotFoundError(f"precomputed_latents_path not found: {latent_path}")
        payload = torch.load(latent_path, map_location="cpu")
        expected_song_names = [record.name for record in self.song_records]

        if torch.is_tensor(payload):
            self._set_single_song_latents(payload)
            return
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported latent cache format: {type(payload)}")

        if "z0_by_song" in payload:
            per_song = payload["z0_by_song"]
        elif "latents_by_song" in payload:
            per_song = payload["latents_by_song"]
        elif "z0_by_chunk" in payload:
            self._set_single_song_latents(payload["z0_by_chunk"])
            return
        elif "latents" in payload:
            self._set_single_song_latents(payload["latents"])
            return
        else:
            raise KeyError("Expected latent cache to contain z0_by_chunk, latents, z0_by_song, or latents_by_song.")

        cache_song_names: list[str] = []
        meta = payload.get("meta")
        if isinstance(meta, dict):
            meta_songs = meta.get("songs")
            if isinstance(meta_songs, list):
                for song_meta in meta_songs:
                    if isinstance(song_meta, dict) and "name" in song_meta:
                        cache_song_names.append(str(song_meta["name"]))

        if len(self.song_records) != len(per_song):
            if cache_song_names:
                if len(cache_song_names) != len(per_song):
                    raise ValueError(
                        f"Latent cache meta songs {len(cache_song_names)} != latent tensors {len(per_song)} "
                        f"for cache={latent_path}."
                    )
                per_song_map = {name: latents for name, latents in zip(cache_song_names, per_song)}
                missing = [name for name in expected_song_names if name not in per_song_map]
                if missing:
                    raise ValueError(
                        f"Latent cache missing dataset songs {missing}. cache={latent_path} "
                        f"dataset_song_names={expected_song_names} cache_song_names={cache_song_names}. "
                        "Re-run scripts/precompute_audio_latents.py with the current multi-song config, "
                        "or set latent_cache.enabled=false to encode audio on the fly."
                    )
                per_song = [per_song_map[name] for name in expected_song_names]
            else:
                details = [
                    f"Latent cache songs {len(per_song)} != dataset songs {len(self.song_records)}.",
                    f"cache={latent_path}",
                    f"dataset_song_names={expected_song_names}",
                ]
                details.append(
                    "Re-run scripts/precompute_audio_latents.py with the current multi-song config, "
                    "or set latent_cache.enabled=false to encode audio on the fly."
                )
                raise ValueError(" ".join(details))

        self.z0_by_song = []
        for song_idx, (record, latents) in enumerate(zip(self.song_records, per_song)):
            if not torch.is_tensor(latents) or latents.dim() != 4:
                raise ValueError(f"Expected cached latents [N,C,H,W] for song {song_idx}, got {type(latents)}")
            required_chunks = record.chunk_offset + record.n_chunks
            if latents.shape[0] < required_chunks:
                raise ValueError(
                    f"Latent cache chunks {latents.shape[0]} < required chunks {required_chunks} "
                    f"for song {record.name} (offset={record.chunk_offset}, n_chunks={record.n_chunks})"
                )
            self.z0_by_song.append(latents.float().contiguous())
        self.latents_by_song = self.z0_by_song
        self.latent_shape = tuple(int(v) for v in self.z0_by_song[0].shape[1:])
        if len(self.song_records) == 1:
            self.z0_by_chunk = self.z0_by_song[0]

    def _set_single_song_latents(self, z0_by_chunk: torch.Tensor) -> None:
        if z0_by_chunk.dim() != 4:
            raise ValueError(f"Expected latent cache [N,C,H,W], got {tuple(z0_by_chunk.shape)}")
        if len(self.song_records) != 1:
            raise ValueError("Single-song latent cache provided for a multi-song dataset.")
        record = self.song_records[0]
        required_chunks = record.chunk_offset + record.n_chunks
        if z0_by_chunk.shape[0] < required_chunks:
            raise ValueError(
                f"Latent cache chunks {z0_by_chunk.shape[0]} < required chunks {required_chunks} "
                f"(offset={record.chunk_offset}, n_chunks={record.n_chunks})"
            )
        self.z0_by_chunk = z0_by_chunk.float().contiguous()
        self.z0_by_song = [self.z0_by_chunk]
        self.latents_by_song = self.z0_by_song
        self.latent_shape = tuple(int(v) for v in self.z0_by_chunk.shape[1:])

    def __len__(self) -> int:
        return len(self.index_map)

    def _get_source_array(self, src: EEGSource) -> np.ndarray:
        key = (src.mat_path, src.data_key)
        cached = self._source_cache.get(key)
        if cached is not None:
            if key in self._source_cache_order:
                self._source_cache_order.remove(key)
            self._source_cache_order.append(key)
            return cached

        arr = _load_eeg_source(
            name=src.name,
            mat_path=src.mat_path,
            data_key=src.data_key,
        )
        self._source_cache[key] = arr
        self._source_cache_order.append(key)
        while len(self._source_cache_order) > self._max_cached_sources:
            old_key = self._source_cache_order.pop(0)
            self._source_cache.pop(old_key, None)
        return arr

    def _get_source_chunk_array(self, src: EEGSource) -> np.ndarray | None:
        if src.chunk_cache_path is None:
            return None
        cached = self._chunk_cache_arrays.get(src.chunk_cache_path)
        if cached is not None:
            return cached
        arr = np.load(src.chunk_cache_path, mmap_mode="r")
        self._chunk_cache_arrays[src.chunk_cache_path] = arr
        return arr

    def _slice_eeg(self, song_idx: int, src_name: str, subj_idx: int, chunk_idx: int) -> np.ndarray:
        src = self.song_records[song_idx].sources[src_name]
        absolute_chunk_idx = self.song_records[song_idx].chunk_offset + chunk_idx
        chunk_arr = self._get_source_chunk_array(src)
        if chunk_arr is not None:
            if src.chunk_cache_subject_indices is None:
                raise RuntimeError(f"Chunk cache subject index metadata missing for source {src.name}.")
            try:
                local_subj_idx = src.chunk_cache_subject_indices.index(int(subj_idx))
            except ValueError as exc:
                raise KeyError(
                    f"Subject {subj_idx} not present in chunk cache for source {src.name}. "
                    f"cached_subjects={list(src.chunk_cache_subject_indices)}"
                ) from exc
            return np.asarray(chunk_arr[local_subj_idx, absolute_chunk_idx], dtype=np.float32).copy()

        st = absolute_chunk_idx * self.eeg_chunk_len
        ed = st + self.eeg_chunk_len
        src_arr = self._get_source_array(src)
        return src_arr[:, st:ed, subj_idx].copy()

    def _build_eeg(self, song_idx: int, subj_idx: int, chunk_idx: int) -> tuple[np.ndarray, bool]:
        eeg_one = self._slice_eeg(song_idx, "passive", subj_idx, chunk_idx)
        return eeg_one, True

    def __getitem__(self, idx: int) -> dict[str, Any]:
        song_idx, subj_idx, chunk_idx = self.index_map[idx]
        song = self.song_records[song_idx]
        eeg, is_passive = self._build_eeg(song_idx, subj_idx, chunk_idx)

        absolute_chunk_idx = song.chunk_offset + chunk_idx
        a_st = absolute_chunk_idx * self.audio_chunk_len
        a_ed = a_st + self.audio_chunk_len
        audio = song.audio[a_st:a_ed].copy()

        if self.normalize_audio:
            max_abs = np.max(np.abs(audio)) + 1e-8
            audio = audio / max_abs

        sample = {
            "eeg": torch.tensor(eeg, dtype=torch.float32),                 # [C, T]
            "audio": torch.tensor(audio, dtype=torch.float32),             # [L]
            "subject_idx": torch.tensor(subj_idx, dtype=torch.long),
            "chunk_idx": torch.tensor(chunk_idx, dtype=torch.long),
            "song_idx": torch.tensor(song_idx, dtype=torch.long),
            "song_name": song.name,
            "condition_type": self.condition_type,
            "condition_id": torch.tensor(self.CONDITION_TO_ID[self.condition_type], dtype=torch.long),
            "is_passive": torch.tensor(bool(is_passive)),
            "text": self.text_prompt,
        }
        if self.z0_by_song is not None:
            sample["z0"] = self.z0_by_song[song_idx][absolute_chunk_idx].clone()
        return sample
