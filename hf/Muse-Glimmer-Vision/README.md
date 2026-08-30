---
license: apache-2.0
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- meta-models/Muse-Glimmer-30B
---

# Muse Glimmer Vision

This repository packages the Tower and Projector from
[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 806 | Muse Glimmer Tower |
| `projector.safetensors` | 3 | Two vision-adapter matrices and the language-width projection |
| `projector_config.json`, `projector.py` | | Projector shapes and loader |
| `config.json`, `preprocessor_config.json` | | Vision-only model and image-processing configuration |

## Architecture

| Component | Details |
|---|---|
| Tower | 50 layers, 1536 hidden, 16 heads, 8960 intermediate, patch size 14 |
| Token compression | 2x2 pixel shuffle, no learned parameters |
| Projector | `Linear(6144, 4096)`, GELU, `Linear(4096, 4096)`, GELU, `Linear(4096, 6656)`, scale-free `RMSNorm(6656)` |

All three Projector linear layers omit bias. The native `MuseGlimmerVisionModel` returns
the merged 6144-wide tokens; `projector.py` maps them to the 6656-wide language space.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.

## Validation

The release tests compare all 806 Tower tensors and three Projector tensors with the
pinned parent checkpoint using `torch.equal`. Fixed-image merged and projected outputs
also match the parent implementation bit-for-bit on CPU and in BF16 on an NVIDIA A100.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_muse_glimmer_vision.py)
reads the Tower and Projector ranges from both `meta-models/Muse-Glimmer-30B` shards. It
removes the parent prefixes, preserves the original BF16 weights, and extracts the image
section of the parent processor configuration. The script pins the parent revision.

## Credits

Meta released the Muse Glimmer weights and the native Transformers implementation.

## License

[Apache License 2.0](LICENSE), the same license as the source model. The source
[usage policy](USAGE_POLICY.md) is included.
