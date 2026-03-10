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
        mat_path: str,
        audio_path: str,
        data_key: str = "data21",
        condition_sources: dict[str, dict[str, str]] | None = None,
        chunk_sec: float = 3.5,
        eeg_fs: int = 125,
        audio_fs: int = 16000,
        subjects: list[int] | None = None,
        normalize_eeg: bool = True,
        normalize_audio: bool = True,
        text_prompt: str = "Pop music",
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
        self.precomputed_latents_path = precomputed_latents_path

        self.instrument_to_id = {inst: i for i, inst in enumerate(self.active_instruments)}

        condition_sources = dict(condition_sources or {})
        # Ensure passive source exists.
        if "passive" not in condition_sources:
            condition_sources["passive"] = {"mat_path": mat_path, "data_key": data_key}
        for inst in self.active_instruments:
            if inst not in condition_sources:
                condition_sources[inst] = {"mat_path": mat_path, "data_key": data_key}

        self.sources: dict[str, EEGSource] = {}
        cache: dict[tuple[str, str], EEGSource] = {}
        for name, spec in condition_sources.items():
            p = spec.get("mat_path", mat_path)
            k = spec.get("data_key", data_key)
            key = (p, k)
            if key not in cache:
                cache[key] = _load_eeg_source(name=name, mat_path=p, data_key=k)
            self.sources[name] = cache[key]

        required_sources = self._required_source_names()
        first = self.sources[required_sources[0]]
        self.base_eeg_channels = first.n_channels
        for name in required_sources[1:]:
            src = self.sources[name]
            if src.n_channels != self.base_eeg_channels:
                raise ValueError(f"Channel mismatch: {name}={src.n_channels} vs {self.base_eeg_channels}")
            if src.total_subjects != first.total_subjects:
                raise ValueError(f"Subject mismatch: {name}={src.total_subjects} vs {first.total_subjects}")

        self.total_subjects = first.total_subjects
        if subjects is None:
            self.subjects = list(range(self.total_subjects))
        else:
            for s in subjects:
                if s < 0 or s >= self.total_subjects:
                    raise ValueError(f"Invalid subject index {s}; total_subjects={self.total_subjects}")
            self.subjects = list(subjects)

        # Audio is shared across conditions; keep chunk alignment by chunk_idx.
        audio, sr = sf.read(audio_path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != self.audio_fs:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_fs)
        self.audio = audio.astype(np.float32)

        n_chunks_audio = len(self.audio) // self.audio_chunk_len
        n_chunks_eeg = min(self.sources[n].total_time // self.eeg_chunk_len for n in required_sources)
        self.n_chunks = int(min(n_chunks_audio, n_chunks_eeg))
        if self.n_chunks == 0:
            raise ValueError(f"No usable chunks. audio={n_chunks_audio}, eeg={n_chunks_eeg}")

        self.eeg_out_channels = self.base_eeg_channels * 3

        self.index_map: list[tuple[int, int]] = []
        for subj_idx in self.subjects:
            for chunk_idx in range(self.n_chunks):
                self.index_map.append((subj_idx, chunk_idx))

        self.z0_by_chunk: torch.Tensor | None = None
        if precomputed_latents_path is not None:
            latent_path = Path(precomputed_latents_path)
            if not latent_path.exists():
                raise FileNotFoundError(f"precomputed_latents_path not found: {latent_path}")
            payload = torch.load(latent_path, map_location="cpu")
            if isinstance(payload, dict):
                if "z0_by_chunk" in payload:
                    z0_by_chunk = payload["z0_by_chunk"]
                elif "latents" in payload:
                    z0_by_chunk = payload["latents"]
                else:
                    raise KeyError("Expected 'z0_by_chunk' or 'latents' in latent cache.")
            elif torch.is_tensor(payload):
                z0_by_chunk = payload
            else:
                raise TypeError(f"Unsupported latent cache format: {type(payload)}")
            if z0_by_chunk.dim() != 4:
                raise ValueError(f"Expected latent cache [N,C,H,W], got {tuple(z0_by_chunk.shape)}")
            if z0_by_chunk.shape[0] < self.n_chunks:
                raise ValueError(f"Latent cache chunks {z0_by_chunk.shape[0]} < dataset chunks {self.n_chunks}")
            self.z0_by_chunk = z0_by_chunk[: self.n_chunks].float().contiguous()

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

    def _slice_eeg(self, src_name: str, subj_idx: int, chunk_idx: int) -> np.ndarray:
        src = self.sources[src_name]
        st = chunk_idx * self.eeg_chunk_len
        ed = st + self.eeg_chunk_len
        return src.data[:, st:ed, subj_idx].copy()

    def _build_eeg(self, subj_idx: int, chunk_idx: int) -> tuple[np.ndarray, int, int, bool]:
        if self.condition_type == "multi_attention":
            parts = [self._slice_eeg(inst, subj_idx, chunk_idx) for inst in self.active_instruments[:3]]
            eeg = np.concatenate(parts, axis=0)
            return eeg, -1, -1, False

        if self.condition_type == "single_repeated":
            eeg_one = self._slice_eeg(self.target_instrument, subj_idx, chunk_idx)
            eeg = np.concatenate([eeg_one, eeg_one, eeg_one], axis=0)
            return eeg, self.instrument_to_id[self.target_instrument], 0, False

        if self.condition_type == "passive_x3":
            eeg_one = self._slice_eeg("passive", subj_idx, chunk_idx)
            eeg = np.concatenate([eeg_one, eeg_one, eeg_one], axis=0)
            return eeg, -1, -1, True

        raise RuntimeError("unreachable")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        subj_idx, chunk_idx = self.index_map[idx]
        eeg, instrument_id, trial_id, is_passive = self._build_eeg(subj_idx, chunk_idx)

        a_st = chunk_idx * self.audio_chunk_len
        a_ed = a_st + self.audio_chunk_len
        audio = self.audio[a_st:a_ed].copy()

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
            "condition_type": self.condition_type,
            "condition_id": torch.tensor(self.CONDITION_TO_ID[self.condition_type], dtype=torch.long),
            "instrument_id": torch.tensor(instrument_id, dtype=torch.long),
            "trial_id": torch.tensor(trial_id, dtype=torch.long),
            "is_passive": torch.tensor(bool(is_passive)),
            "text": self.text_prompt,
        }
        if self.z0_by_chunk is not None:
            sample["z0"] = self.z0_by_chunk[chunk_idx].clone()
        return sample
