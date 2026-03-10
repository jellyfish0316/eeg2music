# audioldm_min

This directory contains the minimal subset of upstream `AudioLDM-training-finetuning`
code retained for the current paper-aligned training path.

Source provenance:

- upstream project: `AudioLDM-training-finetuning`
- copied modules:
  - `audioldm_train/modules/diffusionmodules/openaimodel.py`
  - `audioldm_train/modules/diffusionmodules/attention.py`
- locally re-authored minimal utility support:
  - `diffusion_util.py`

Local modifications:

- imports rewritten to local relative imports
- upstream package side effects removed
- only the current U-Net training chain is preserved
- unused upstream training/data/audio helpers are intentionally omitted

This repo owns maintenance of this subset. It is no longer intended to track
upstream structure or commits.
