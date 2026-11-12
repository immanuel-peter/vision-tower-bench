---
license: other
license_name: minimax-community
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
base_model:
- MiniMaxAI/MiniMax-M3.1-Flash
tags:
- vision
---

# MiniMax-M3.1-Flash Vision

Standalone Tower and Projector extracted from MiniMaxAI/MiniMax-M3.1-Flash,
revision `54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`. No language-model weights
are included. The accompanying MiniMax Community License applies to these weights.

The Tower has 32 layers, width 1280, 16 heads, MLP width 5120, patch size 14,
temporal patch size 2, and 3D RoPE. The nominal image size is 672.
At 448 pixels it produces 1024 Tower tokens of width 1280. The bench merges
2x2 spatial neighbors into 256 tokens of width 5120. The Projector outputs
256 tokens of width 6144.

The bundle contains 515 Tower tensors in `model.safetensors` and eight Projector
tensors in `projector.safetensors`. All tensors preserve their pinned source
values and dtypes. `export_manifest.json` records checkpoint identity and file hashes.
Shard selection comes from the pinned source index, rather than M3 shard numbers.

## Use

Requires Transformers 5.17.0, PyTorch and Pillow. See `examples/inference.py`.
Load the Tower with `MiniMaxM3VLVisionModel.from_pretrained`; use `projector.py`
to load the separate Projector. The native parent processor uses dynamic resizing.
The benchmark uses a bicubic short-side resize and center crop at 448, CLIP
normalization and two repeated frames.

## Validation

`scripts/validate_minimax_m3.py` in vision-tower-bench independently loads the
pinned source tensors into native HF parent classes. Every Tower and Projector
tensor matches exactly. On fixed synthetic images at 56 and 448 pixels, all
eight Tower depths, merged tokens and projected tokens match exactly in BF16.
Changing nominal image size between 672 and 2016 leaves those outputs unchanged.
The report records the Transformers source hash. The pinned HF parent repository
contains no separate modeling Python file; this check uses the native HF implementation.

MiniMax M3 and M3.1-Flash share this architecture and preprocessing, but their
weights differ. Benchmark results and the complete tensor drift report are maintained
in [vision-tower-bench](https://github.com/immanuel-peter/vision-tower-bench).
The native HF vision forward packs supplied images into one attention sequence.
Forward unrelated images independently. The benchmark adapter isolates both the
Tower and Projector per image before restoring the external batch layout.
Its corrected outputs match independently loaded single-image parent forwards
exactly for batches of 1, 2 and 8, reversed order and changed neighbors, at 56 and
448 pixels. The reference uses native HF classes, not a second modeling implementation.
No benchmark score is implied by this release.

## Reproduce

`python scripts/export_minimax_m3_vision.py --checkpoint minimax_m31_flash`

Built with MiniMax M3.1-Flash. MiniMax released the parent weights and native Transformers implementation.
See LICENSE and NOTICE.
