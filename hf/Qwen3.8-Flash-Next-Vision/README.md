---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next Vision

This repository packages the Tower and learned merger from
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 333 | Tower and learned merger, extracted from source shard 1 |
| `config.json` | | Vision-only `Qwen3_5VisionConfig` |
| `preprocessor_config.json` | | Qwen image preprocessing configuration |
| `export_manifest.json` | | Source pin, tensor counts and file hashes |
| `LICENSE` | | Qwen Community License 1.0 |

Qwen implements the learned merger inside `Qwen3_5VisionModel`, so this repository does
not need a separate Projector file.

## Architecture

| Component | Details |
|---|---|
| Tower | 27 layers, 1152 hidden, 16 heads, 4304 intermediate, patch size 16 |
| Token compression | 2x2 spatial grouping |
| Learned merger | `LayerNorm(1152)`, `Linear(4608, 4608)`, GELU, `Linear(4608, 2560)` |

`last_hidden_state` contains the raw Tower tokens. `pooler_output` contains the merged
features at the language-model width. At 448 pixels the example reports Tower
`(1, 784, 1152)`, merged `(1, 196, 4608)` and projected `(1, 196, 2560)`.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.

## Validation

The release tests compare all 333 tensors with the pinned parent checkpoint using
`torch.equal`. Fixed-image Tower and merged outputs also match the parent implementation
bit-for-bit on CPU and in BF16 on an NVIDIA RTX PRO 6000 Blackwell.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_qwen3_8_vision.py)
reads `model.visual.*` from shard 1 of `Qwen/Qwen3.8-Flash-Next`, removes the prefix,
and writes the original BF16 tensors. The script pins the parent revision.

## Credits

Qwen released the Qwen3.8-Flash-Next weights and the native Transformers implementation.

## License

[Qwen Community License 1.0](LICENSE), the same license as the source model.
