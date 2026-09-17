---
license: mit
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- zai-org/GLM-5.3-Flash
---

# GLM-5.3-Flash Vision

This repository packages the native vision encoder and learned merger from
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 347 | Tower and learned merger, extracted from source shard 62 |
| `config.json` | | Vision-only `Glm5NextVisionConfig` |
| `preprocessor_config.json` | | GLM image preprocessing configuration |

## Architecture

| Component | Details |
|---|---|
| Tower | 24 blocks, 1024 hidden, 16 heads, 4096 intermediate, patch size 14, image size 448, axial 2D-RoPE, silu |
| Patch embed | Conv3d, temporal patch size 2 |
| Token compression | 2x2 spatial grouping of 1024-wide patches to 4096 |
| Learned merger | `Linear(4096, 4096)` no bias, `LayerNorm(4096)`, GELU, SwiGLU with `Linear(4096, 10240)` gate and up, `Linear(10240, 4096)` down, silu, clamp 10 |

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.

## Validation

The [parity script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/tests/test_parity.py)
compares all 347 tensors with the pinned parent checkpoint using `torch.equal`.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_glm5_vision.py)
reads `model.visual.*` from shard 62 of `zai-org/GLM-5.3-Flash`, removes the prefix, and
writes the original BF16 tensors. It copies the image section of the parent processor
configuration.

## Credits

Z.ai released the GLM-5.3-Flash weights and the native Transformers implementation.

## License

[MIT License](LICENSE), the same license as the source model.
