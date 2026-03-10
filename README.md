# EEG AudioLDM2 Reproduction

Paper-aligned EEG-to-audio generation prototype based on `AudioLDM2-music`.

Target paper:

- *Naturalistic Music Decoding from EEG Data via Latent Diffusion Models*

Current model path:

`EEG -> subject adapter -> 1D Conv projector -> ControlNet encoder copy -> frozen AudioLDM2 U-Net -> latent diffusion`

This repo is set up to follow the paper design more closely than the earlier `global encoder + hint encoder` version. The active training path now uses:

- subject-aware EEG affine modulation
- 1D temporal EEG projector
- ControlNet-style encoder copy with zero-initialized residual injection
- frozen AudioLDM2 backbone
- latent diffusion loss on AudioLDM2 VAE latents

## Overview

This repo aims to reproduce the paper's EEG-conditioned latent diffusion setup without an official author release. The implementation is therefore intentionally conservative: it prefers paper-described behavior over extra engineering tricks, and it makes explicit where the code is an inferred reproduction rather than a line-by-line match to an unavailable original repository.

At a high level, the system:

1. reads raw EEG time-series
2. optionally applies subject-specific affine modulation
3. projects EEG into an AudioLDM2-compatible latent grid
4. injects ControlNet-style residuals into a frozen AudioLDM2 U-Net encoder
5. trains against diffusion noise prediction in latent space

## Repo Layout

- [configs/train.yaml](/home/bryan/eeg/configs/train.yaml): main training config
- [scripts/train.py](/home/bryan/eeg/scripts/train.py): LOSO training entrypoint
- [scripts/precompute_latents.py](/home/bryan/eeg/scripts/precompute_latents.py): precompute AudioLDM2 latents
- [models/eeg_controlnet.py](/home/bryan/eeg/models/eeg_controlnet.py): main model
- [models/eeg_projector.py](/home/bryan/eeg/models/eeg_projector.py): paper-style EEG projector
- [models/audioldm_control_branch.py](/home/bryan/eeg/models/audioldm_control_branch.py): ControlNet adapter branch
- [models/audioldm2_wrapper.py](/home/bryan/eeg/models/audioldm2_wrapper.py): AudioLDM2 VAE wrapper and latent-shape introspection
- [datasets/condition_nmedt_dataset.py](/home/bryan/eeg/datasets/condition_nmedt_dataset.py): condition-aware EEG dataset
- [tests/test_paper_alignment_smoke.py](/home/bryan/eeg/tests/test_paper_alignment_smoke.py): smoke and regression tests

## Data Assumptions

The default config expects:

- EEG `.mat` data shaped as `[channels, time, subjects]`
- audio waveform aligned to EEG chunks
- chunk size `3.5s`
- EEG sample rate `125 Hz`
- audio sample rate `16 kHz`

Default text conditioning is fixed to:

```text
Pop music
```

Default EEG preprocessing is intentionally minimal:

- per-channel normalization
- no band-pass
- no ICA
- raw time-series input

## Config Notes

The active config is [configs/train.yaml](/home/bryan/eeg/configs/train.yaml).

Important fields:

- `data.text_prompt`: constant text prompt, default `"Pop music"`
- `data.eeg_preprocessing`: explicit preprocessing policy
- `model.projector.channels`: default `[256, 512, 1024, 2048]`
- `model.projector.strides`: default `[5, 2, 2, 2]`
- `model.projector.lat_grid`: set to `null` by default, so latent shape is derived from latent cache or checkpoint
- `model.projector.use_linear_fallback`: fallback only when temporal length does not match latent grid exactly
- `controlnet.copy_encoder_weights`: must remain `true` for paper-aligned behavior
- `controlnet.inject_middle_block`: whether to inject ControlNet residuals at the U-Net middle block

## Environment

The examples below assume the conda env is:

```bash
conda activate eeg
```

If your Python executable is not the active env default, use:

```bash
/home/bryan/miniconda3/envs/eeg/bin/python
```

Core runtime dependencies include:

- `torch`
- `torchaudio`
- `diffusers`
- `pyyaml`
- `scipy`
- `soundfile`
- `librosa`
- `pytest` for tests

## Tests

Run the paper-alignment smoke and regression tests:

```bash
python -m pytest tests/test_paper_alignment_smoke.py -q
```

Current test coverage includes:

- forward/backward smoke test
- freeze policy regression
- latent-grid derivation precedence
- projector fallback behavior

If you want the shortest reproducibility check, run:

```bash
python -m pytest tests/test_paper_alignment_smoke.py -q
python scripts/train.py --config configs/train.yaml --fold 0 --max-steps 1
```

## Training

Run a minimal end-to-end sanity check:

```bash
python scripts/train.py --config configs/train.yaml --fold 0 --max-steps 1
```

Run a normal single-fold training job:

```bash
python scripts/train.py --config configs/train.yaml --fold 0
```

Run all folds with a short smoke budget:

```bash
python scripts/train.py --config configs/train.yaml --all-folds --max-steps 1
```

Outputs are written under:

```text
outputs/loso_runs/
```

Typical artifacts:

- `all_results.json`
- `pairwise_report.json`
- per-condition `result.json`
- per-condition `model.pt`

## Example Result

A minimal sanity run should produce logs like:

```text
device=cuda cuda_available=True cuda_device_count=1
total_subjects: 20
condition_jobs: ['multi_attention', 'passive_x3']
[fold 0][multi_attention] ... loss=...
[fold 0][passive_x3] ... loss=...
saved: outputs/loso_runs/all_results.json
saved: outputs/loso_runs/pairwise_report.json
```

## Latent Cache

If `latent_cache.enabled: true`, training uses precomputed AudioLDM2 latents instead of encoding waveform on the fly.

To generate latents:

```bash
python scripts/precompute_latents.py --config configs/train.yaml
```

The config default points to:

```text
data/precomputed/song21_audioldm2_latents.pt
```

## Implementation Status

What is paper-aligned now:

- frozen AudioLDM2 latent diffusion backbone
- checkpoint-derived latent grid
- 1D EEG projector
- ControlNet-style encoder-copy branch with zero-conv residuals
- subject adapter before projector

What is still approximate:

- `L(y,s)` is implemented as affine modulation, because the paper does not provide a more exact public implementation
- exact paper code is unavailable, so some internal ControlNet block mapping is inferred from the vendored AudioLDM U-Net structure

## Decisions Made Without an Official Repo

Because the paper does not ship an official implementation, the following choices are explicit reproduction decisions rather than verified author code:

- `L(y,s)` is implemented as feature-wise affine modulation in [subject_adapter.py](/home/bryan/eeg/models/subject_adapter.py).
Reason: the paper indicates subject-aware modulation, but does not provide a public exact layer definition.

- The EEG projector is implemented as a 4-layer 1D ConvNet in [eeg_projector.py](/home/bryan/eeg/models/eeg_projector.py).
Reason: the paper describes a 1D temporal projector with channels `(256, 512, 1024, 2048)` and strides `(5, 2, 2, 2)`, so the repo uses that as the default shape-defining architecture.

- `model.projector.lat_grid` defaults to `null` in [train.yaml](/home/bryan/eeg/configs/train.yaml), and the real latent grid is derived from latent cache or checkpoint at runtime.
Reason: `[8, 16, 87]` is a common observed result for `AudioLDM2-music`, but it should not be treated as a universal constant.

- A linear remap exists in the projector as a fallback, not as the primary architecture.
Reason: the paper's intended design is `1D conv -> reshape`; the fallback only prevents training from breaking when temporal length does not land exactly on the checkpoint-derived latent grid.

- The ControlNet branch in [audioldm_control_branch.py](/home/bryan/eeg/models/audioldm_control_branch.py) deep-copies the vendored AudioLDM U-Net encoder and middle block, then adds zero-initialized 1x1 convolutions.
Reason: this is the closest faithful interpretation of the paper's `copy encoder weights + zero-conv residual injection` description, given no official code.

- Middle-block injection is configurable through `controlnet.inject_middle_block`, but defaults to enabled.
Reason: the paper description includes encoder-path residual control and is compatible with ControlNet-style middle-block residuals; enabling it is the more conservative reproduction choice.

- The U-Net no longer uses the earlier repo's extra global EEG FiLM path.
Reason: the paper-aligned path should be `EEG projector -> ControlNet residual injection`, not `global EEG embedding -> extra U-Net conditioning`.

- The text prompt is fixed to `"Pop music"` by default, and EEG preprocessing is kept minimal.
Reason: those two details are closer to the paper's stated setup than adding extra handcrafted EEG transforms.

- Internal block/channel matching for ControlNet residual injection is inferred from the vendored `AudioLDM` U-Net structure in [audioldm_unet_wrapper.py](/home/bryan/eeg/models/audioldm_unet_wrapper.py).
Reason: with no official repo, the vendored model structure is the only concrete source of truth for stage alignment.

These choices are meant to keep the repo paper-aligned in spirit and interface, while making each non-verifiable implementation choice explicit.

## Citation

If you use this repo, cite the original paper first. A basic BibTeX stub:

```bibtex
@inproceedings{postolache2025naturalistic,
  title={Naturalistic Music Decoding from EEG Data via Latent Diffusion Models},
  author={Postolache, Emilian and others},
  booktitle={ICASSP 2025},
  year={2025}
}
```

## Known Practical Notes

- The vendored AudioLDM code is loaded from [vendor/audioldm_min/README.md](/home/bryan/eeg/vendor/audioldm_min/README.md), not from a tracked upstream repo.
- On some systems you may see CUDA or joblib warnings during import or tests; the current smoke tests still pass under those warnings.
- `model.projector.lat_grid` should usually stay `null` unless you intentionally want to override checkpoint-derived shape.
