# EEG-to-Music With AudioLDM2

EEG-conditioned music generation built on pretrained `cvssp/audioldm2-music`.

Reference paper:

- *Naturalistic Music Decoding from EEG Data via Latent Diffusion Models*

Current repo focus:

- passive EEG (NMED-T) and multi-condition self-recorded EEG (guitar / vocal / drum)
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

- `passive` condition (NMED-T) and multi-condition (`guitar`, `vocal`, `drum`) for self-recorded EEG
- fixed prompt from `data.text_prompt`
- checkpoint selection by `val_clap`
- train / val / test chunk splits on in-distribution songs
- separate `ood_test` song split
- precomputed AudioLDM2 latent cache
- precomputed EEG chunk cache
- EEG channel interpolation variant (see `_interp` configs)

Removed from the main runtime path:

- LOSO / fold training
- dataset-side EEG preprocessing

## Repo Layout

- [configs/train_NMEDT.yaml](configs/train_NMEDT.yaml): main config for NMED-T passive EEG
- [configs/train_self_recorded_multicond.yaml](configs/train_self_recorded_multicond.yaml): self-recorded guitar + vocal + drum
- [configs/train_self_recorded_multicond_interp.yaml](configs/train_self_recorded_multicond_interp.yaml): same but with interpolated EEG mats
- [configs/train_self_recorded_passive.yaml](configs/train_self_recorded_passive.yaml): self-recorded passive only
- [configs/train_self_recorded_passive3.yaml](configs/train_self_recorded_passive3.yaml): self-recorded passive × 3
- [configs/train_self_recorded_passive3_interp.yaml](configs/train_self_recorded_passive3_interp.yaml): passive × 3 with interpolated EEG mats
- [configs/train_self_recorded_guitar.yaml](configs/train_self_recorded_guitar.yaml): self-recorded guitar only
- [configs/train_self_recorded_vocal.yaml](configs/train_self_recorded_vocal.yaml): self-recorded vocal only
- [configs/train_self_recorded_drum.yaml](configs/train_self_recorded_drum.yaml): self-recorded drum only
- [scripts/train.py](scripts/train.py): training entrypoint
- [scripts/generate.py](scripts/generate.py): generate `.wav` from a checkpoint
- [scripts/evaluate_generation.py](scripts/evaluate_generation.py): CLAP audio-audio evaluation against targets
- [scripts/evaluate_audio_sets.py](scripts/evaluate_audio_sets.py): set-level CLAP/text/Frechet-style evaluation
- [scripts/evaluate_stem_retrieval.py](scripts/evaluate_stem_retrieval.py): CLAP-based stem retrieval evaluation for multi-condition runs
- [scripts/compare_unet_to_official.py](scripts/compare_unet_to_official.py): compare wrapper U-Net output to official AudioLDM2 U-Net
- [scripts/precompute_audio_latents.py](scripts/precompute_audio_latents.py): precompute AudioLDM2 latents
- [scripts/precompute_eeg_chunks.py](scripts/precompute_eeg_chunks.py): cut EEG chunk cache
- [scripts/prepare_nmedt_raw_eeg.py](scripts/prepare_nmedt_raw_eeg.py): inspect and convert raw NMED-T recordings
- [scripts/prepare_cdt_eeg.py](scripts/prepare_cdt_eeg.py): inspect and convert Curry / Neuroscan `.cdt` files
- [scripts/check_mel_vocoder_compat.py](scripts/check_mel_vocoder_compat.py): diagnose mel/vocoder mismatch and roundtrip noise
- [datasets/condition_nmedt_dataset.py](datasets/condition_nmedt_dataset.py): EEG dataset supporting passive and multi-condition (`condition_sources`) channel concatenation
- [models/eeg_conditioned_audioldm2.py](models/eeg_conditioned_audioldm2.py): main model
- [models/eeg_projector.py](models/eeg_projector.py): 1D EEG projector
- [models/eeg_controlnet.py](models/eeg_controlnet.py): EEG ControlNet branch
- [models/subject_adapter.py](models/subject_adapter.py): per-subject embedding adapter
- [models/audioldm2_unet_wrapper.py](models/audioldm2_unet_wrapper.py): pretrained diffusers U-Net wrapper
- [models/audioldm2_vae_wrapper.py](models/audioldm2_vae_wrapper.py): AudioLDM2 VAE / decode / CLAP helper

## Data Assumptions

All configs share these data format requirements:

- song-level EEG `.mat` files shaped as `[channels, time, subjects]`
- one EEG file per song
- one aligned audio file per song
- `chunk_sec = 3.5`
- `eeg_fs = 1000`
- `audio_fs = 16000`

The NMED-T config (`train_NMEDT.yaml`) uses:

- in-distribution songs: `song22` to `song30`
- OOD song: `song21`
- subject subset: `[1]` — index 1 selects the second subject (zero-based)

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

If your `.cdt` is a continuous recording, first use `inspect` to check annotations/triggers, then pass `--tmin` and `--duration` for the song segment you want to export. After conversion, point the config at the new `.mat` and keep `data_key` aligned with the song, for example `data21`.

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

For single-condition self-recorded runs use `train_self_recorded_passive.yaml` and point `mat_path` at the passive mat. For multi-condition EEG conditioning, set `data.condition_sources`. The dataset concatenates these sources on the EEG channel axis before passing them to the model. For example, 128-channel self-recorded EEG with three sources becomes 384 input channels.

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

## Interpolated EEG Variants

The `_interp` configs (`train_self_recorded_multicond_interp.yaml`, `train_self_recorded_passive3_interp.yaml`) point to `data/SelfRecorded_EEG_Processed_Interpolated/` instead of `data/SelfRecorded_EEG_Processed/`. These mats have had missing or noisy channels interpolated before saving. Use them when the raw `.cdt` conversion produces mats with bad channels. Everything else (training, generation, evaluation) works identically.

## Config Notes

Important fields shared across all configs:

- `data.songs`: full song list
- `data.eeg_chunk_cache_dir`: EEG chunk cache directory
- `latent_cache.path`: precomputed AudioLDM2 latent cache
- `split.subject_indices`: selected subjects
- `split.song_splits`: in-distribution train/val/test songs
- `split.ood_song_splits.ood_test`: OOD songs
- `split.chunk_splits`: chunk ranges for each split
- `train.validation_metric`: `clap` by default
- `train.output_root`: output directory for checkpoints and results

Pick the config that matches your data:

| Config | Data | Condition |
|--------|------|-----------|
| `train_NMEDT.yaml` | NMED-T | passive |
| `train_self_recorded_multicond.yaml` | self-recorded | guitar + vocal + drum |
| `train_self_recorded_multicond_interp.yaml` | self-recorded (interpolated) | guitar + vocal + drum |
| `train_self_recorded_passive.yaml` | self-recorded | passive |
| `train_self_recorded_passive3.yaml` | self-recorded | passive × 3 |
| `train_self_recorded_passive3_interp.yaml` | self-recorded (interpolated) | passive × 3 |
| `train_self_recorded_guitar.yaml` | self-recorded | guitar |
| `train_self_recorded_vocal.yaml` | self-recorded | vocal |
| `train_self_recorded_drum.yaml` | self-recorded | drum |

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

Replace the config with whichever one you are training with. The `_interp` configs need their own chunk cache because they point to different mat files.

## Training

Minimal smoke test (NMED-T):

```bash
python scripts/train.py --config configs/train_NMEDT.yaml --max-steps 20
```

Full run (NMED-T):

```bash
python scripts/train.py --config configs/train_NMEDT.yaml
```

Self-recorded multi-condition:

```bash
python scripts/train.py --config configs/train_self_recorded_multicond.yaml
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
  --config configs/train_NMEDT.yaml \
  --checkpoint outputs/checked_byme_subject2_v1_ep100/best_model.pt \
  --split test \
  --num-inference-steps 50 \
  --output-dir outputs/generated/test_real
```

Generate OOD audio:

```bash
python scripts/generate.py \
  --config configs/train_NMEDT.yaml \
  --checkpoint outputs/checked_byme_subject2_v1_ep100/best_model.pt \
  --split ood_test \
  --num-inference-steps 50 \
  --output-dir outputs/generated/ood_real
```

Generate no-control baseline:

```bash
python scripts/generate.py \
  --config configs/train_NMEDT.yaml \
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

### CLAP audio similarity (generated vs target)

```bash
python scripts/evaluate_generation.py \
  --manifest outputs/generated/ood_real/manifest.json \
  --output-dir outputs/generated/ood_real_eval
```

The key number is `mean_clap_audio_cosine`.

### Stem retrieval (multi-condition runs)

Measures whether the generated audio retrieves the correct instrument stem over distractors using CLAP similarity. Stems are expected to be pre-separated with [htdemucs](https://github.com/facebookresearch/demucs) and placed under `--stems-root`.

```bash
python scripts/evaluate_stem_retrieval.py \
  --manifest outputs/generated/test_real/manifest.json \
  --stems-root data/SelfRecorded_songs/separated/htdemucs \
  --output-dir outputs/generated/test_real_stem_eval
```

Multiple manifests can be passed to `--manifest` to compare runs side by side. The default stems are `drums`, `other_bass`, `vocals`. The condition names used during training map to htdemucs buckets as follows: `drum` → `drums`, `vocal` → `vocals`, `guitar` → `other_bass` (htdemucs groups guitar/bass/piano under `other` and `bass` tracks).

### Set-level evaluation

Compares a generated set against a reference set using CLAP text similarity and Fréchet-style CLAP audio distance:

```bash
python scripts/evaluate_audio_sets.py \
  --candidate generated=outputs/generated/test_real \
  --reference outputs/target_audio \
  --output outputs/audio_set_eval/summary.json
```

`--candidate` accepts `label=path` and can be repeated to compare multiple runs at once. `--reference` and `--candidate` paths can be a wav directory, `manifest.json`, or a manifest root directory.

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

### U-Net output comparison

To verify the wrapper U-Net matches the official AudioLDM2 U-Net numerically:

```bash
python scripts/compare_unet_to_official.py \
  --config configs/train_NMEDT.yaml \
  --output outputs/unet_compare.json
```

## Tests

Run the core tests:

```bash
python -m pytest tests/test_generation_pipeline.py tests/test_paper_alignment_smoke.py tests/test_train_validation_metric.py -q
```

## Important Practical Notes

- Old checkpoints from the pre-refactor latent layout are not compatible with the current code.
- If latent layout or mel preprocessing changes, re-run [precompute_audio_latents.py](scripts/precompute_audio_latents.py) before training.
- `best_model.pt` is the main checkpoint because selection follows `val_clap`.
- The current audio path is aligned to diffusers layout, but mel/vocoder compatibility should still be checked empirically.
- After switching between a regular and `_interp` config, re-run [precompute_eeg_chunks.py](scripts/precompute_eeg_chunks.py) since the mat paths differ.

## Limitations

- **Multi-condition EEG quantity imbalance**: the self-recorded dataset has more passive trials than attention-directed (guitar / vocal / drum) trials. This can affect model convergence when all three conditions are concatenated.
- **Single subject per run**: `split.subject_indices` currently selects one subject at a time. Cross-subject generalization requires re-running with different index settings or extending the training loop.
- **No online EEG preprocessing**: artifact rejection, bandpass filtering, and rereferencing are expected to be done during the `.mat` conversion step (`prepare_nmedt_raw_eeg.py` / `prepare_cdt_eeg.py`). The dataset loader does not apply any EEG preprocessing.
- **Fixed text prompt**: the model uses a single global text prompt (`data.text_prompt`) for all samples. Per-song or per-chunk text conditioning is not wired up.
- **Checkpoint compatibility**: checkpoints are not compatible across latent layout changes. Re-precompute latents and retrain when the mel or VAE path changes.

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
