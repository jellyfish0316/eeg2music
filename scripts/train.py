from __future__ import annotations

import yaml
import torch
from torch.utils.data import DataLoader

from datasets.nmedt_dataset import NMEDTDataset
from models.eeg_controlnet import EEGControlNetModel
from utils.seed import set_seed


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():

    # -----------------------
    # Load config
    # -----------------------
    cfg = load_config("configs/train.yaml")

    set_seed(cfg["seed"])

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # -----------------------
    # Dataset
    # -----------------------
    dataset = NMEDTDataset(
        mat_path="data/EEG/song21_Imputed.mat",
        audio_path="data/songs/song21_16k.wav",
        data_key="data21",
        chunk_sec=cfg["data"]["chunk_sec"],
        eeg_fs=cfg["data"]["eeg_fs"],
        audio_fs=cfg["data"]["audio_fs"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )

    print("dataset size:", len(dataset))

    # -----------------------
    # Model
    # -----------------------
    model = EEGControlNetModel(
        eeg_channels=cfg["data"]["eeg_channels"],
        num_subjects=dataset.total_subjects,
        use_subject_adapter=cfg["model"]["use_subject_adapter"],
        subject_emb_dim=cfg["model"]["subject_emb_dim"],
    ).to(device)

    print("model parameters:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
    )

    # -----------------------
    # Training loop
    # -----------------------
    model.train()

    for epoch in range(cfg["train"]["epochs"]):

        for step, batch in enumerate(loader):

            eeg = batch["eeg"].to(device)               # [B,125,437]
            audio = batch["audio"].to(device)           # [B,56000]
            subject_idx = batch["subject_idx"].to(device)

            out = model(
                eeg=eeg,
                audio=audio,
                subject_idx=subject_idx,
            )

            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["train"]["grad_clip"],
            )

            optimizer.step()

            if step % 10 == 0:

                print(
                    f"[epoch {epoch:02d} step {step:04d}] "
                    f"loss={loss.item():.6f} "
                    f"eeg={tuple(eeg.shape)} "
                    f"audio={tuple(audio.shape)} "
                    f"z0={tuple(out['z0'].shape)} "
                    f"cond={tuple(out['eeg_cond'].shape)}"
                )

    # -----------------------
    # Save checkpoint
    # -----------------------
    torch.save(model.state_dict(), "eeg_controlnet_smoke.pt")

    print("saved -> eeg_controlnet_smoke.pt")


if __name__ == "__main__":
    main()