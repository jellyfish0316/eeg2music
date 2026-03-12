

# EEG Attention-Guided Music Experiment Framework

Proposal-driven EEG-to-music experiment framework built on pretrained `AudioLDM2-music`.

Reference paper:

- *Naturalistic Music Decoding from EEG Data via Latent Diffusion Models*

Reference model path:

`EEG -> subject adapter -> 1D Conv projector -> ControlNet encoder copy -> frozen AudioLDM2 U-Net -> latent diffusion`

The current implementation uses:

- subject-aware EEG affine modulation
- 1D temporal EEG projector
- ControlNet-style encoder copy with zero-initialized residual injection
- pretrained AudioLDM2 U-Net loaded from `AudioLDM2Pipeline`
- latent diffusion loss on AudioLDM2 VAE latents

## Model Architecture

<p align="center">
  <img width="467" height="591" alt="EEG2Music drawio" src="https://github.com/user-attachments/assets/6e06a40f-3675-451c-aae2-5e546a6e0ca6" />
</p>

## Overview

This repo is intended to support attention-guided EEG-to-music experiments, using *Naturalistic Music Decoding from EEG Data via Latent Diffusion Models* as the backbone reference architecture.

At a high level, the system:

1. reads raw EEG time-series
2. optionally applies subject-specific affine modulation
3. projects EEG into an AudioLDM2-compatible latent grid
4. injects ControlNet-style residuals into a frozen AudioLDM2 U-Net encoder
5. trains against diffusion noise prediction in latent space

## Current Status

Paper-aligned components already present on `main`:

- the denoiser backbone is the pretrained diffusers `AudioLDM2Pipeline.unet`
- the ControlNet branch is copied from the pretrained U-Net encoder and middle block
- the fixed prompt from `data.text_prompt` is encoded through `AudioLDM2Pipeline.encode_prompt()`
- the repo supports the current condition types `multi_attention`, `single_repeated`, and `passive_x3`
- training uses LOSO subject splits and can select checkpoints with CLAP-based validation on the current config

Proposal-specific work not yet validated without real data:

- the current config does not prove that attention conditions come from distinct real recordings
- preprocessing assumptions are still placeholders until the actual dataset is inspected
- current smoke-oriented configs and outputs should not be treated as proposal-level experiment results

## Paper Vs Proposal

The original paper is the model reference, but the proposal extends it in several important ways.

- The paper provides the latent diffusion backbone: pretrained `AudioLDM2-music`, frozen U-Net, ControlNet conditioning, fixed prompt usage, and CLAP-oriented validation.
- The proposal adds attention-guided condition design, especially multi-attention integration across `drum`, `vocal`, and `guitar` style recordings.
- The proposal also shifts the experimental emphasis toward LOSO pilot evaluation and explicit comparisons between `multi_attention`, `single_repeated`, and `passive_x3`.

Because of that, proposal assumptions may differ from the original paper in subject count, preprocessing policy, condition construction, and evaluation setup. This README treats the paper as the architectural baseline and the proposal as the active experimental target.

## Repo Layout

- [configs/train.yaml](/home/bryan/eeg/configs/train.yaml): main training config
- [scripts/train.py](/home/bryan/eeg/scripts/train.py): LOSO training entrypoint
- [scripts/precompute_latents.py](/home/bryan/eeg/scripts/precompute_latents.py): precompute AudioLDM2 latents
- [scripts/generate.py](/home/bryan/eeg/scripts/generate.py): decode EEG-conditioned latents into `.wav` files
- [scripts/evaluate_generation.py](/home/bryan/eeg/scripts/evaluate_generation.py): CLAP audio-similarity evaluation over generated vs target audio
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

`data.text_prompt` defaults to `"Pop music"` and is encoded once at model initialization through `AudioLDM2Pipeline.encode_prompt()`. The cached prompt embeddings are then reused for every batch.

Default EEG preprocessing is intentionally minimal:

- per-channel normalization
- no band-pass
- no ICA
- raw time-series input

## Config Notes

The active config is [configs/train.yaml](/home/bryan/eeg/configs/train.yaml).

Important fields:

- `data.text_prompt`: dataset-level/default prompt string, default `"Pop music"`
- `data.eeg_preprocessing`: explicit preprocessing policy
- `model.unet.cache_pipeline`: keep the diffusers pipeline object alive after extracting the pretrained U-Net
- `model.unet.text_cache_path`: optional on-disk cache for fixed-prompt AudioLDM2 text embeddings
- `model.projector.channels`: default `[256, 512, 1024, 2048]`
- `model.projector.strides`: default `[5, 2, 2, 2]`
- `model.projector.lat_grid`: set to `null` by default, so latent shape is derived from latent cache or checkpoint
- `model.projector.use_linear_fallback`: fallback only when temporal length does not match latent grid exactly
- `controlnet.copy_encoder_weights`: must remain `true` for paper-aligned behavior
- `controlnet.inject_middle_block`: whether to inject ControlNet residuals at the U-Net middle block
- `train.validation_metric`: `clap` by default, so checkpoint selection follows generation-based validation
- `train.validation_num_inference_steps`: denoising steps used for optional validation-by-generation

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
python -m pytest tests/test_paper_alignment_smoke.py tests/test_generation_pipeline.py tests/test_train_validation_metric.py -q
```

Current test coverage includes:

- forward/backward smoke test
- freeze policy regression
- latent-grid derivation precedence
- projector fallback behavior

If you want the shortest reproducibility check, run:

```bash
python -m pytest tests/test_paper_alignment_smoke.py tests/test_generation_pipeline.py tests/test_train_validation_metric.py -q
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
- per-condition `best_model.pt`

With the current default config, `best_model.pt` is the primary checkpoint because model selection is driven by `val_clap`. `model.pt` is still saved as the last-epoch checkpoint.

`model.pt` and `best_model.pt` both include the cached fixed-prompt text embeddings via persistent buffers, so loading the checkpoint does not require regenerating them.

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

These logs indicate that the current training path is wired correctly. They do not, by themselves, establish that the proposal experiment has been completed.

## Generation And Evaluation

Generate decoded audio from a trained checkpoint:

```bash
python scripts/generate.py \
  --config configs/train.yaml \
  --checkpoint outputs/loso_runs/fold_00/multi_attention/best_model.pt \
  --fold 0 \
  --condition multi_attention \
  --split test \
  --num-inference-steps 50 \
  --output-dir outputs/generated_audio/multi_attention
```

Evaluate generated audio against target chunks with CLAP audio cosine:

```bash
python scripts/evaluate_generation.py \
  --manifest outputs/generated_audio/multi_attention/manifest.json \
  --output-dir outputs/generated_audio/eval
```

## Latent Cache

If `latent_cache.enabled: true`, training uses precomputed AudioLDM2 latents instead of encoding waveform on the fly.

## What Must Be Done After Data Arrives

The following milestones should be completed before claiming that proposal-level experiments are running:

1. Data audit

Verify subject count, condition definitions, recording alignment, EEG sampling rate, usable channel count, and whether each attention condition is truly a separate recording rather than a placeholder reuse.

2. Preprocessing alignment

Implement or confirm the final preprocessing pipeline required by the proposal and the paper reference, including channel selection, scaling policy, and any clamping or normalization rules.

3. Dataset wiring

Map the real condition-specific files into `multi_attention`, `single_repeated`, and `passive_x3` so each branch uses the intended source recordings.

4. Config finalization

Update chunk length, EEG sample rate, train/validation/test policy, LOSO settings, and latent-cache assumptions so they match the real dataset rather than the current smoke defaults.

5. Baselines and controls

Run the proposal's comparison conditions and confirm that each control is matched by actual data design, not only by tensor shape or repeated placeholders.

6. Evaluation

Generate audio and compute the agreed metrics, keeping smoke outputs clearly separated from real experiment outputs.

7. Result validation

Compare observed results against both the original paper and the proposal hypotheses, then document where the implementation matches, extends, or diverges from each.

To generate latents with the current script:

```bash
python scripts/precompute_latents.py
```

Note: `scripts/precompute_latents.py` currently reads `configs/train.yaml` directly and does not yet expose a `--config` CLI flag.

The config default points to:

```text
data/precomputed/song21_audioldm2_latents.pt
```

The fixed prompt text cache defaults to:

```text
data/precomputed/audioldm2_text_pop_music.pt
```

## Implementation Status

What is aligned now:

- pretrained AudioLDM2 latent diffusion backbone loaded from diffusers weights
- checkpoint-derived latent grid
- 1D EEG projector
- ControlNet-style encoder-copy branch with zero-conv residuals
- subject adapter before projector

What is still approximate:

- `L(y,s)` is implemented as affine modulation, because the paper does not provide a more exact public implementation
- exact paper code is unavailable, so some internal ControlNet block mapping is inferred from the diffusers AudioLDM2 U-Net block structure
- text conditioning is fixed-prompt only, not yet per-sample or prompt-variable

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

- The ControlNet branch in [audioldm_control_branch.py](/home/bryan/eeg/models/audioldm_control_branch.py) deep-copies the pretrained diffusers AudioLDM2 U-Net encoder path and middle block, then adds zero-initialized 1x1 convolutions.
Reason: this keeps the runtime backbone tied to actual pretrained weights instead of an architecture-only reimplementation.

- Middle-block injection is configurable through `controlnet.inject_middle_block`, but defaults to enabled.
Reason: the paper description includes encoder-path residual control and is compatible with ControlNet-style middle-block residuals; enabling it is the more conservative reproduction choice.

- The U-Net no longer uses the earlier repo's extra global EEG FiLM path.
Reason: the paper-aligned path should be `EEG projector -> ControlNet residual injection`, not `global EEG embedding -> extra U-Net conditioning`.

- The dataset/config default prompt remains `"Pop music"`, and EEG preprocessing is kept minimal.
Reason: those two details are closer to the paper's stated setup than adding extra handcrafted EEG transforms, and the fixed prompt is now encoded through the official AudioLDM2 text path.

- Internal block/channel matching for ControlNet residual injection is inferred from the pretrained diffusers `AudioLDM2Pipeline.unet` structure in [audioldm_unet_wrapper.py](/home/bryan/eeg/models/audioldm_unet_wrapper.py).
Reason: with no official repo, the pretrained diffusers module graph is the most concrete source of truth for stage alignment.

These choices are meant to keep the repo close to the paper in architecture and training interface, while making each non-verifiable implementation choice explicit.

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

- The runtime U-Net is loaded from `diffusers` `AudioLDM2Pipeline`; the vendored AudioLDM subset is no longer the training backbone.
- The current text conditioning is a cached fixed prompt, not yet per-sample prompt conditioning.
- Fixed-prompt text embeddings can be cached on disk through `model.unet.text_cache_path` and are also saved inside `model.pt`.
- The default training config now uses `validation_metric: clap`, so `best_model.pt` is expected after each condition run.
- On some systems you may see CUDA or joblib warnings during import or tests; the current smoke tests still pass under those warnings.
- `model.projector.lat_grid` should usually stay `null` unless you intentionally want to override checkpoint-derived shape.
