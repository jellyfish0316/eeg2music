# EEG-to-Music With AudioLDM2

EEG-conditioned music generation built on pretrained `cvssp/audioldm2-music`.

Reference paper:

- *Naturalistic Music Decoding from EEG Data via Latent Diffusion Models*

Current repo focus:

- passive EEG only
- single-condition training
- pretrained diffusers AudioLDM2 U-Net backbone
- ControlNet-style EEG conditioning
- NMED-T style multi-song splits with optional OOD song evaluation

## Current Pipeline

```text
EEG -> subject adapter -> 1D EEG projector -> EEG ControlNet branch -> frozen AudioLDM2 U-Net
                                                                    \
audio -> mel -> AudioLDM2 VAE latent ------------------------------- diffusion loss
```

Audio path conventions in the current code:

- mel layout: `[B, 1, T, F]`
- latent layout: `[B, C, T/4, F/4]`

These are aligned to the current diffusers `AudioLDM2Pipeline` path used by the repo.

## What The Repo Supports Now

- `passive` condition only
- fixed prompt from `data.text_prompt`
- checkpoint selection by `val_clap`
- train / val / test chunk splits on in-distribution songs
- separate `ood_test` song split
- precomputed AudioLDM2 latent cache
- precomputed EEG chunk cache

Removed from the main runtime path:

- multi-attention conditions
- per-instrument condition wiring
- LOSO / fold training
- dataset-side EEG preprocessing

## Repo Layout

- [configs/train.yaml](/home/bryan/eeg/configs/train.yaml): main config
- [scripts/train.py](/home/bryan/eeg/scripts/train.py): training entrypoint
- [scripts/generate.py](/home/bryan/eeg/scripts/generate.py): generate `.wav` from a checkpoint
- [scripts/evaluate_generation.py](/home/bryan/eeg/scripts/evaluate_generation.py): CLAP audio-audio evaluation against targets
- [scripts/evaluate_audio_sets.py](/home/bryan/eeg/scripts/evaluate_audio_sets.py): set-level CLAP/text/Frechet-style evaluation
- [scripts/precompute_audio_latents.py](/home/bryan/eeg/scripts/precompute_audio_latents.py): precompute AudioLDM2 latents
- [scripts/precompute_eeg_chunks.py](/home/bryan/eeg/scripts/precompute_eeg_chunks.py): cut EEG chunk cache
- [scripts/prepare_nmedt_raw_eeg.py](/home/bryan/eeg/scripts/prepare_nmedt_raw_eeg.py): inspect and convert raw NMED-T recordings
- [scripts/check_mel_vocoder_compat.py](/home/bryan/eeg/scripts/check_mel_vocoder_compat.py): diagnose mel/vocoder mismatch and roundtrip noise
- [datasets/condition_nmedt_dataset.py](/home/bryan/eeg/datasets/condition_nmedt_dataset.py): passive EEG dataset
- [models/eeg_conditioned_audioldm2.py](/home/bryan/eeg/models/eeg_conditioned_audioldm2.py): main model
- [models/eeg_projector.py](/home/bryan/eeg/models/eeg_projector.py): 1D EEG projector
- [models/eeg_controlnet.py](/home/bryan/eeg/models/eeg_controlnet.py): EEG ControlNet branch
- [models/audioldm2_unet_wrapper.py](/home/bryan/eeg/models/audioldm2_unet_wrapper.py): pretrained diffusers U-Net wrapper
- [models/audioldm2_vae_wrapper.py](/home/bryan/eeg/models/audioldm2_vae_wrapper.py): AudioLDM2 VAE / decode / CLAP helper

## Data Assumptions

The default config expects:

- song-level EEG `.mat` files shaped as `[channels, time, subjects]`
- one EEG file per song
- one aligned audio file per song
- `chunk_sec = 3.5`
- `eeg_fs = 1000`
- `audio_fs = 16000`

The current config uses:

- in-distribution songs: `song22` to `song30`
- OOD song: `song21`
- subject subset: `[1]` which means subject 2 if mats are zero-based indexed

## Raw NMED-T Conversion

If you start from raw MATLAB v7.3 participant recordings, first inspect them:

```bash
python scripts/prepare_nmedt_raw_eeg.py inspect \
  --file data/EEG/02_1_raw.mat
```

Then convert them to song-level processed mats:

```bash
python scripts/prepare_nmedt_raw_eeg.py convert \
  --raw-dir data/EEG \
  --output-dir data/NMEDT_EEG_processed \
  --eeg-key eeg \
  --src-fs 1000 \
  --dst-fs 1000
```

Notes:

- the current converter is trigger-based
- fallback fixed-length song cutting was removed
- EEG preprocessing is expected to happen here, not later in the dataset

## CDT / Curry EEG Conversion

The training dataset does not require MATLAB specifically. It requires one song-level EEG array per song with shape `[channels, time, subjects]`, saved in a `.mat` under a key like `data21`.

For Curry / Neuroscan `.cdt` files, install MNE in the active environment:

```bash
pip install mne
```

Inspect a `.cdt` file:

```bash
python scripts/prepare_cdt_eeg.py inspect data/cdt/song21_subject02.cdt
```

Convert one or more already aligned song-level `.cdt` files into the repo-compatible `.mat` format:

```bash
python scripts/prepare_cdt_eeg.py convert \
  data/cdt/song21_subject02.cdt data/cdt/song21_subject03.cdt \
  --output data/SelfRecorded_EEG_Processed/song21_Processed.mat \
  --song-name song21 \
  --dst-fs 1000 \
  --trim-to-shortest
```

If your `.cdt` is a continuous recording, first use `inspect` to check annotations/triggers, then pass `--tmin` and `--duration` for the song segment you want to export. After conversion, point `configs/train.yaml` at the new `.mat` and keep `data_key` aligned with the song, for example `data21`.

If your PsychoPy task writes condition triggers like this:

- cue: `11` / `12` / `13` / `14`
- music start: `21` / `22` / `23` / `24`
- music end: `31` / `32` / `33` / `34`

use trigger-based conversion instead. By default it cuts:

- `drum`: `21 -> 31`
- `vocal`: `22 -> 32`
- `guitar`: `23 -> 33`
- `passive`: `24 -> 34`

```bash
python scripts/prepare_cdt_eeg.py convert-events \
  data/cdt/sub02_song7.cdt data/cdt/sub03_song7.cdt \
  --output-dir data/SelfRecorded_EEG_Processed \
  --song-name song7 \
  --data-key data7 \
  --dst-fs 1000 \
  --trim-to-shortest
```

Put the expected channel count in the training config so loading fails early if the processed `.mat` has the wrong shape:

```yaml
data:
  expected_eeg_channels: 128
```

The current training path uses only `passive`, so point a self-recorded config at `data/SelfRecorded_EEG_Processed/song7_passive_Processed.mat` unless you re-enable multi-condition training.

For multi-condition EEG conditioning, set `data.condition_sources`. The dataset concatenates these sources on the EEG channel axis before passing them to the model. For example, 128-channel self-recorded EEG with three sources becomes 384 input channels.

Self-recorded example using `guitar + vocal + drum`:

```yaml
data:
  condition_sources: [guitar, vocal, drum]
  songs:
    - name: song7
      mat_path: data/SelfRecorded_EEG_Processed/song7_passive_Processed.mat
      audio_path: data/SelfRecorded_songs/wav_16k/song7.wav
      data_key: data7
      sources:
        guitar:
          mat_path: data/SelfRecorded_EEG_Processed/song7_guitar_Processed.mat
          data_key: data7
        vocal:
          mat_path: data/SelfRecorded_EEG_Processed/song7_vocal_Processed.mat
          data_key: data7
        drum:
          mat_path: data/SelfRecorded_EEG_Processed/song7_drum_Processed.mat
          data_key: data7
```

Fallback using `passive * 3`:

```yaml
data:
  condition_sources: [passive, passive, passive]
```

If you use `scripts/precompute_eeg_chunks.py`, re-run it after changing `condition_sources` or source paths so the EEG chunk manifest matches the config.

## Config Notes

Important fields in [train.yaml](/home/bryan/eeg/configs/train.yaml):

- `data.songs`: full song list
- `data.eeg_chunk_cache_dir`: EEG chunk cache directory
- `latent_cache.path`: precomputed AudioLDM2 latent cache
- `split.subject_indices`: selected subjects
- `split.song_splits`: in-distribution train/val/test songs
- `split.ood_song_splits.ood_test`: OOD songs
- `split.chunk_splits`: chunk ranges for each split
- `train.validation_metric`: `clap` by default
- `train.output_root`: output directory for checkpoints and results

## Environment

Typical environment:

```bash
conda activate eeg
```

Core dependencies:

- `torch`
- `torchaudio`
- `diffusers`
- `transformers`
- `pyyaml`
- `scipy`
- `soundfile`
- `librosa`
- `pytest`

## Precompute

### 1. Audio latent cache

Run this whenever the audio latent layout or mel preprocessing changes:

```bash
python scripts/precompute_audio_latents.py --config configs/train_self_recorded_multicond.yaml
```

### 2. EEG chunk cache

Run this when song-level EEG mats or chunk rules change:

```bash
python scripts/precompute_eeg_chunks.py --config configs/train_self_recorded_multicond.yaml
```

## Training

Minimal smoke:

```bash
python scripts/train.py --config configs/train.yaml --max-steps 20
```

Full run:

```bash
python scripts/train.py --config configs/train.yaml
```

Outputs are written under `train.output_root`, for example:

```text
outputs/checked_byme_subject2_v1_ep100/
```

Typical artifacts:

- `result.json`
- `model.pt`
- `best_model.pt`
- `checkpoint_path.txt`
- `all_results.json`

## Generation

Generate in-distribution test audio:

```bash
python scripts/generate.py \
  --config configs/train.yaml \
  --checkpoint outputs/checked_byme_subject2_v1_ep100/best_model.pt \
  --split test \
  --num-inference-steps 50 \
  --output-dir outputs/generated/test_real
```

Generate OOD audio:

```bash
python scripts/generate.py \
  --config configs/train.yaml \
  --checkpoint outputs/checked_byme_subject2_v1_ep100/best_model.pt \
  --split ood_test \
  --num-inference-steps 50 \
  --output-dir outputs/generated/ood_real
```

Generate no-control baseline:

```bash
python scripts/generate.py \
  --config configs/train.yaml \
  --checkpoint outputs/checked_byme_subject2_v1_ep100/best_model.pt \
  --split ood_test \
  --num-inference-steps 50 \
  --disable-control \
  --output-dir outputs/generated/ood_no_control
```

Useful generation flags:

- `--max-batches`
- `--eeg-mode real|zero|random`
- `--disable-control`

## Evaluation

Evaluate generated vs target audio with CLAP audio similarity:

```bash
python scripts/evaluate_generation.py \
  --manifest outputs/generated/ood_real/manifest.json \
  --output-dir outputs/generated/ood_real_eval
```

The key number is:

- `mean_clap_audio_cosine`

## Audio Path Diagnostics

If generation sounds semantically right but noisy, check the audio path itself first:

```bash
python scripts/check_mel_vocoder_compat.py \
  --audio data/songs/song22_16k.wav \
  --output-dir outputs/mel_vocoder_check_song22
```

This script compares:

- `mel -> vocoder`
- `mel -> VAE latent -> decode -> vocoder`

and writes:

- `summary.json`
- per-variant `.wav` files

This is the fastest way to tell whether noise is already present before EEG conditioning.

## Tests

Run the core tests:

```bash
python -m pytest tests/test_generation_pipeline.py tests/test_paper_alignment_smoke.py tests/test_train_validation_metric.py -q
```

## Important Practical Notes

- Old checkpoints from the pre-refactor latent layout are not compatible with the current code.
- If latent layout or mel preprocessing changes, re-run [precompute_audio_latents.py](/home/bryan/eeg/scripts/precompute_audio_latents.py) before training.
- `best_model.pt` is the main checkpoint because selection follows `val_clap`.
- The current audio path is aligned to diffusers layout, but mel/vocoder compatibility should still be checked empirically.

## Citation

If you use this repo, cite the original paper:

```bibtex
@inproceedings{postolache2025naturalistic,
  title={Naturalistic Music Decoding from EEG Data via Latent Diffusion Models},
  author={Postolache, Emilian and others},
  booktitle={ICASSP 2025},
  year={2025}
}
```
