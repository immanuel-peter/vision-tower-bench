---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
pipeline_tag: image-feature-extraction
library_name: pytorch
tags:
- vision
- feature-extraction
- nemotron
base_model:
- nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16
---

# C-RADIOv4-H-3.5-Super

This repository packages the C-RADIOv4-H Tower, final LayerNorm, and Projector from [Nemotron 3.5 Super VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 391 | RADIO Tower from Super `vision_model.*`, including `summary_idxs` |
| `final_norm.safetensors` | 2 | `vision_projector.vision_final_layernorm` |
| `projector.safetensors` | 3 | Super `mlp1` Projector |
| `config.json`, `inventory.json` | | Source and code pins, widths, and the shape, dtype and SHA256 of every tensor |
| `vision.py` | | Loader used by `examples/inference.py` |

## Architecture

| Component | Details |
|---|---|
| Tower | C-RADIOv4-H, 32 layers, 1280 hidden, patch size 16, ten prefix tokens |
| Final LayerNorm | `LayerNorm(1280)`, eps `1e-6`, applied before the pixel shuffle |
| Token compression | InternVL v2 2x2 pixel shuffle, scale 0.5, no learned parameters |
| Projector (`mlp1`) | `RMSNorm(5120)`, `Linear(5120, 20480)` no bias, SquaredReLU, `Linear(20480, 4096)` no bias |

At 448 the bench reports 784 Tower tokens of width 1280, 196 `merged` tokens of
width 5120 (after the final LayerNorm), and 196 `projected` tokens of width 4096.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.

## Validation

The [release parity test](https://github.com/immanuel-peter/vision-tower-bench/blob/main/tests/test_release_parity.py) compares every exported tensor with the pinned parent using `torch.equal`, then checks all stages against the parent's own `RadioModel` and `VisionProjector` code.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_nemotron35_vision.py) reads `vision_model.*`, `mlp1.*`, and `vision_projector.vision_final_layernorm.*` from the pinned parent and writes the original BF16 tensors.

## Credits

NVIDIA released [Nemotron 3.5 Super VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) and [C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H).

## License

[OpenMDW-1.1](LICENSE), the license from the source model card.
