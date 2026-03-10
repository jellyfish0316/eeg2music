from __future__ import annotations

from typing import Any
from pathlib import Path

import scipy.io
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader


class NMEDTDataset(Dataset):
    def __init__(
        self,
        mat_path: str,
        audio_path: str,
        data_key: str = "data21",
        chunk_sec: float = 3.5,
        eeg_fs: int = 125,
        audio_fs: int = 16000,
        subjects: list[int] | None = None,
        normalize_eeg: bool = True,
        normalize_audio: bool = True,
        text_prompt: str = "Pop music",
        precomputed_latents_path: str | None = None,
    ):
        self.chunk_sec = chunk_sec
        self.eeg_fs = eeg_fs
        self.audio_fs = audio_fs
        self.eeg_chunk_len = int(chunk_sec * eeg_fs)      # 437
        self.audio_chunk_len = int(chunk_sec * audio_fs)  # 56000
        self.normalize_eeg = normalize_eeg
        self.normalize_audio = normalize_audio
        self.text_prompt = text_prompt
        self.precomputed_latents_path = precomputed_latents_path

        # Load EEG from .mat
        mat = scipy.io.loadmat(mat_path)
        if data_key not in mat:
            raise KeyError(f"'{data_key}' not found in mat file. Available keys: {list(mat.keys())}")

        data = mat[data_key]  # expected shape: (channels, time, subjects)
        self.data = data.astype(np.float32)

        if self.data.ndim != 3:
            raise ValueError(
                f"Expected EEG data shape (channels, time, subjects), got {self.data.shape}"
            )

        self.n_channels, self.total_time, self.total_subjects = self.data.shape

        if subjects is None:
            self.subjects = list(range(self.total_subjects))
        else:
            for s in subjects:
                if s < 0 or s >= self.total_subjects:
                    raise ValueError(f"Invalid subject index {s}, total_subjects={self.total_subjects}")
            self.subjects = subjects

        # Load audio from .wav
        audio, sr = sf.read(audio_path)

        # Convert stereo -> mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        # Resample if needed
        if sr != audio_fs:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=audio_fs)

        self.audio = audio.astype(np.float32)

        # Compute usable chunk count
        self.n_chunks_eeg = self.total_time // self.eeg_chunk_len
        self.n_chunks_audio = len(self.audio) // self.audio_chunk_len
        self.n_chunks = min(self.n_chunks_eeg, self.n_chunks_audio)

        if self.n_chunks == 0:
            raise ValueError(
                f"No usable chunks found. EEG chunks={self.n_chunks_eeg}, audio chunks={self.n_chunks_audio}"
            )

        # Build index mapping: one sample = (subject, chunk)
        self.index_map: list[tuple[int, int]] = []
        for subj_idx in self.subjects:
            for chunk_idx in range(self.n_chunks):
                self.index_map.append((subj_idx, chunk_idx))

        self.z0_by_chunk: torch.Tensor | None = None
        if self.precomputed_latents_path is not None:
            latent_path = Path(self.precomputed_latents_path)
            if not latent_path.exists():
                raise FileNotFoundError(f"precomputed_latents_path not found: {latent_path}")

            payload = torch.load(latent_path, map_location="cpu")
            if isinstance(payload, dict):
                if "z0_by_chunk" in payload:
                    z0_by_chunk = payload["z0_by_chunk"]
                elif "latents" in payload:
                    z0_by_chunk = payload["latents"]
                else:
                    raise KeyError(
                        "Expected key 'z0_by_chunk' (or 'latents') in latent cache file."
                    )
            elif torch.is_tensor(payload):
                z0_by_chunk = payload
            else:
                raise TypeError(
                    f"Unsupported latent cache format: {type(payload)}"
                )

            if not torch.is_tensor(z0_by_chunk) or z0_by_chunk.dim() != 4:
                raise ValueError(
                    f"Expected cached latents shape [N,C,H,W], got {type(z0_by_chunk)} "
                    f"with shape {tuple(getattr(z0_by_chunk, 'shape', []))}"
                )
            if z0_by_chunk.shape[0] < self.n_chunks:
                raise ValueError(
                    f"Latent cache has fewer chunks than dataset: {z0_by_chunk.shape[0]} < {self.n_chunks}"
                )
            self.z0_by_chunk = z0_by_chunk[: self.n_chunks].float().contiguous()

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        subj_idx, chunk_idx = self.index_map[idx]

        eeg_start = chunk_idx * self.eeg_chunk_len
        eeg_end = eeg_start + self.eeg_chunk_len

        audio_start = chunk_idx * self.audio_chunk_len
        audio_end = audio_start + self.audio_chunk_len

        # EEG: [C, T]
        eeg = self.data[:, eeg_start:eeg_end, subj_idx].copy()

        # Audio: [L]
        audio = self.audio[audio_start:audio_end].copy()

        if self.normalize_eeg:
            mean = eeg.mean(axis=1, keepdims=True)
            std = eeg.std(axis=1, keepdims=True) + 1e-8
            eeg = (eeg - mean) / std

        if self.normalize_audio:
            max_abs = np.max(np.abs(audio)) + 1e-8
            audio = audio / max_abs

        sample = {
            "eeg": torch.tensor(eeg, dtype=torch.float32),             # [C, T]
            "audio": torch.tensor(audio, dtype=torch.float32),         # [L]
            "subject_idx": torch.tensor(subj_idx, dtype=torch.long),
            "chunk_idx": torch.tensor(chunk_idx, dtype=torch.long),
            "text": self.text_prompt,
        }
        if self.z0_by_chunk is not None:
            # z0 is chunk-level latent shared across subjects.
            sample["z0"] = self.z0_by_chunk[chunk_idx].clone()
        return sample


if __name__ == "__main__":
    dataset = NMEDTDataset(
        mat_path="data/EEG/song21_Imputed.mat",
        audio_path="data/songs/song21_16k.wav",
        data_key="data21",
        text_prompt="Pop music",
    )

    print("dataset len:", len(dataset))

    sample = dataset[0]
    print("eeg shape:", sample["eeg"].shape)
    print("audio shape:", sample["audio"].shape)
    print("subject_idx:", sample["subject_idx"])
    print("chunk_idx:", sample["chunk_idx"])
    print("text:", sample["text"])

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print("batch eeg:", batch["eeg"].shape)         # [4, 125, 437]
    print("batch audio:", batch["audio"].shape)     # [4, 56000]
    print("batch subject_idx:", batch["subject_idx"].shape)
    print("batch chunk_idx:", batch["chunk_idx"].shape)
    print("batch text:", batch["text"])
