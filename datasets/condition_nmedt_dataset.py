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
    data: np.ndarray  # [C, T, S]

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def total_time(self) -> int:
        return int(self.data.shape[1])

    @property
    def total_subjects(self) -> int:
        return int(self.data.shape[2])


@dataclass
class SongRecord:
    name: str
    audio: np.ndarray
    n_chunks: int
    sources: dict[str, EEGSource]


def _load_eeg_source(
    name: str,
    mat_path: str,
    data_key: str,
) -> EEGSource:
    mat = scipy.io.loadmat(mat_path)
    if data_key not in mat:
        raise KeyError(f"'{data_key}' not found in {mat_path}. keys={list(mat.keys())}")
    arr = mat[data_key].astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name}: expected EEG shape [C,T,S], got {arr.shape}")
    return EEGSource(name=name, data=arr)


class ConditionNMEDTDataset(Dataset):
    """
    Condition-aware EEG dataset supporting:
      - multi_attention
      - single_repeated
      - passive_x3
    Output EEG channels are always 3x base channels by design.
    """

    CONDITION_TO_ID = {
        "multi_attention": 0,
        "single_repeated": 1,
        "passive_x3": 2,
    }

    def __init__(
        self,
        *,
        condition_type: str,
        active_instruments: list[str],
        target_instrument: str | None,
        mat_path: str | None = None,
        audio_path: str | None = None,
        data_key: str = "data21",
        condition_sources: dict[str, dict[str, str]] | None = None,
        songs: list[dict[str, Any]] | None = None,
        chunk_sec: float = 3.5,
        eeg_fs: int = 125,
        audio_fs: int = 16000,
        subjects: list[int] | None = None,
        normalize_eeg: bool = True,
        normalize_audio: bool = True,
        text_prompt: str = "Pop music",
        eeg_preprocessing: dict[str, Any] | None = None,
        precomputed_latents_path: str | None = None,
    ) -> None:
        super().__init__()
        if condition_type not in self.CONDITION_TO_ID:
            raise ValueError(f"Unsupported condition_type: {condition_type}")
        if len(active_instruments) == 0:
            raise ValueError("active_instruments cannot be empty.")
        if condition_type == "single_repeated" and target_instrument is None:
            raise ValueError("target_instrument is required for single_repeated.")
        if condition_type == "single_repeated" and target_instrument not in active_instruments:
            raise ValueError(f"target_instrument={target_instrument} not in active_instruments={active_instruments}")

        self.condition_type = condition_type
        self.active_instruments = list(active_instruments)
        self.target_instrument = target_instrument
        self.chunk_sec = float(chunk_sec)
        self.eeg_fs = int(eeg_fs)
        self.audio_fs = int(audio_fs)
        self.eeg_chunk_len = int(self.chunk_sec * self.eeg_fs)
        self.audio_chunk_len = int(self.chunk_sec * self.audio_fs)
        self.normalize_eeg = normalize_eeg
        self.normalize_audio = normalize_audio
        self.text_prompt = text_prompt
        self.eeg_preprocessing = dict(eeg_preprocessing or {"per_channel_normalization": normalize_eeg})
        self.precomputed_latents_path = precomputed_latents_path

        self.instrument_to_id = {inst: i for i, inst in enumerate(self.active_instruments)}

        song_specs = self._normalize_song_specs(
            songs=songs,
            mat_path=mat_path,
            audio_path=audio_path,
            data_key=data_key,
            condition_sources=condition_sources,
        )
        required_sources = self._required_source_names()
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

        self.eeg_out_channels = self.base_eeg_channels * 3

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
    def _normalize_song_specs(
        *,
        songs: list[dict[str, Any]] | None,
        mat_path: str | None,
        audio_path: str | None,
        data_key: str,
        condition_sources: dict[str, dict[str, str]] | None,
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
                        "condition_sources": song.get("condition_sources", condition_sources),
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

    def _build_song_records(
        self,
        song_specs: list[dict[str, Any]],
        *,
        required_sources: list[str],
    ) -> list[SongRecord]:
        records: list[SongRecord] = []
        global_cache: dict[tuple[str, str], EEGSource] = {}

        for idx, spec in enumerate(song_specs):
            mat_path = spec.get("mat_path")
            audio_path = spec.get("audio_path")
            data_key = spec.get("data_key", "data21")
            if mat_path is None:
                raise ValueError(f"Song spec {spec.get('name', idx)!r} is missing mat_path.")

            source_specs = dict(spec.get("condition_sources") or {})
            if "passive" not in source_specs:
                source_specs["passive"] = {"mat_path": mat_path, "data_key": data_key}
            for inst in self.active_instruments:
                if inst not in source_specs:
                    source_specs[inst] = {"mat_path": mat_path, "data_key": data_key}

            sources: dict[str, EEGSource] = {}
            for name, source_spec in source_specs.items():
                p = source_spec.get("mat_path", mat_path)
                k = source_spec.get("data_key", data_key)
                key = (str(p), str(k))
                if key not in global_cache:
                    global_cache[key] = _load_eeg_source(name=name, mat_path=str(p), data_key=str(k))
                sources[name] = global_cache[key]

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
            n_chunks = int(min(n_chunks_audio, n_chunks_eeg))
            if n_chunks == 0:
                raise ValueError(
                    f"No usable chunks for {spec.get('name', idx)!r}. audio={n_chunks_audio}, eeg={n_chunks_eeg}"
                )

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

        if len(self.song_records) != len(per_song):
            cache_song_names: list[str] = []
            meta = payload.get("meta")
            if isinstance(meta, dict):
                meta_songs = meta.get("songs")
                if isinstance(meta_songs, list):
                    for song_meta in meta_songs:
                        if isinstance(song_meta, dict) and "name" in song_meta:
                            cache_song_names.append(str(song_meta["name"]))
            details = [
                f"Latent cache songs {len(per_song)} != dataset songs {len(self.song_records)}.",
                f"cache={latent_path}",
                f"dataset_song_names={expected_song_names}",
            ]
            if cache_song_names:
                details.append(f"cache_song_names={cache_song_names}")
            details.append(
                "Re-run scripts/precompute_latents.py with the current multi-song config, "
                "or set latent_cache.enabled=false to encode audio on the fly."
            )
            raise ValueError(" ".join(details))

        self.z0_by_song = []
        for song_idx, (record, latents) in enumerate(zip(self.song_records, per_song)):
            if not torch.is_tensor(latents) or latents.dim() != 4:
                raise ValueError(f"Expected cached latents [N,C,H,W] for song {song_idx}, got {type(latents)}")
            if latents.shape[0] < record.n_chunks:
                raise ValueError(f"Latent cache chunks {latents.shape[0]} < dataset chunks {record.n_chunks} for song {record.name}")
            self.z0_by_song.append(latents[: record.n_chunks].float().contiguous())
        self.latents_by_song = self.z0_by_song
        self.latent_shape = tuple(int(v) for v in self.z0_by_song[0].shape[1:])
        if len(self.song_records) == 1:
            self.z0_by_chunk = self.z0_by_song[0]

    def _set_single_song_latents(self, z0_by_chunk: torch.Tensor) -> None:
        if z0_by_chunk.dim() != 4:
            raise ValueError(f"Expected latent cache [N,C,H,W], got {tuple(z0_by_chunk.shape)}")
        if len(self.song_records) != 1:
            raise ValueError("Single-song latent cache provided for a multi-song dataset.")
        if z0_by_chunk.shape[0] < self.song_records[0].n_chunks:
            raise ValueError(f"Latent cache chunks {z0_by_chunk.shape[0]} < dataset chunks {self.song_records[0].n_chunks}")
        self.z0_by_chunk = z0_by_chunk[: self.song_records[0].n_chunks].float().contiguous()
        self.z0_by_song = [self.z0_by_chunk]
        self.latents_by_song = self.z0_by_song
        self.latent_shape = tuple(int(v) for v in self.z0_by_chunk.shape[1:])

    def _required_source_names(self) -> list[str]:
        if self.condition_type == "multi_attention":
            if len(self.active_instruments) < 3:
                raise ValueError("multi_attention requires at least 3 active instruments.")
            return self.active_instruments[:3]
        if self.condition_type == "single_repeated":
            return [self.target_instrument]  # repeated x3
        if self.condition_type == "passive_x3":
            return ["passive"]
        raise RuntimeError("unreachable")

    def __len__(self) -> int:
        return len(self.index_map)

    def _slice_eeg(self, song_idx: int, src_name: str, subj_idx: int, chunk_idx: int) -> np.ndarray:
        src = self.song_records[song_idx].sources[src_name]
        st = chunk_idx * self.eeg_chunk_len
        ed = st + self.eeg_chunk_len
        return src.data[:, st:ed, subj_idx].copy()

    def _build_eeg(self, song_idx: int, subj_idx: int, chunk_idx: int) -> tuple[np.ndarray, int, int, bool]:
        if self.condition_type == "multi_attention":
            parts = [self._slice_eeg(song_idx, inst, subj_idx, chunk_idx) for inst in self.active_instruments[:3]]
            eeg = np.concatenate(parts, axis=0)
            return eeg, -1, -1, False

        if self.condition_type == "single_repeated":
            eeg_one = self._slice_eeg(song_idx, self.target_instrument, subj_idx, chunk_idx)
            eeg = np.concatenate([eeg_one, eeg_one, eeg_one], axis=0)
            return eeg, self.instrument_to_id[self.target_instrument], 0, False

        if self.condition_type == "passive_x3":
            eeg_one = self._slice_eeg(song_idx, "passive", subj_idx, chunk_idx)
            eeg = np.concatenate([eeg_one, eeg_one, eeg_one], axis=0)
            return eeg, -1, -1, True

        raise RuntimeError("unreachable")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        song_idx, subj_idx, chunk_idx = self.index_map[idx]
        song = self.song_records[song_idx]
        eeg, instrument_id, trial_id, is_passive = self._build_eeg(song_idx, subj_idx, chunk_idx)

        a_st = chunk_idx * self.audio_chunk_len
        a_ed = a_st + self.audio_chunk_len
        audio = song.audio[a_st:a_ed].copy()

        if self.normalize_eeg:
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True) + 1e-8
            eeg = (eeg - mean) / std
        if self.normalize_audio:
            max_abs = np.max(np.abs(audio)) + 1e-8
            audio = audio / max_abs

        sample = {
            "eeg": torch.tensor(eeg, dtype=torch.float32),                 # [3C, T]
            "audio": torch.tensor(audio, dtype=torch.float32),             # [L]
            "subject_idx": torch.tensor(subj_idx, dtype=torch.long),
            "chunk_idx": torch.tensor(chunk_idx, dtype=torch.long),
            "song_idx": torch.tensor(song_idx, dtype=torch.long),
            "song_name": song.name,
            "condition_type": self.condition_type,
            "condition_id": torch.tensor(self.CONDITION_TO_ID[self.condition_type], dtype=torch.long),
            "instrument_id": torch.tensor(instrument_id, dtype=torch.long),
            "trial_id": torch.tensor(trial_id, dtype=torch.long),
            "is_passive": torch.tensor(bool(is_passive)),
            "text": self.text_prompt,
            "eeg_preprocessing": self.eeg_preprocessing,
        }
        if self.z0_by_song is not None:
            sample["z0"] = self.z0_by_song[song_idx][chunk_idx].clone()
        return sample
