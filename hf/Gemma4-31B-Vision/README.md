---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- google/gemma-4-31B-it
---

# Gemma4-31B Vision

This repository packages the Tower and Projector from
[Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 355 | Tower, extracted from `model.vision_tower.*` in `model-00001-of-00002.safetensors` |
| `projector.safetensors` | 1 | `Linear(1152, 5376)` weight from `model.embed_vision.*` |
| `projector_config.json` | | Projector shapes |
| `config.json` | | Vision-only `Gemma4VisionModel` (`model_type`: `gemma4_vision`) |
| `preprocessor_config.json` | | Gemma 4 image preprocessing configuration |
| `projector.py` | | Projector loader used by `examples/inference.py` |

## Architecture

| Component | Details |
|---|---|
| Tower | 27 layers, 1152 hidden, 16 heads, 4304 intermediate, patch size 16, 3x3 pooling kernel |
| Projector | scale-free `RMSNorm(1152)` then `Linear(1152, 5376)` with no bias |

The RMSNorm has no learned scale, so the packaged Projector tensor is that one Linear
weight.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.

## Validation

The [parity script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/tests/test_parity.py)
compares the 355 Tower tensors and the Projector Linear with the pinned parent using
`torch.equal`.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_gemma4_vision.py)
reads `model.vision_tower.*` (355 tensors) and `model.embed_vision.*` (1 tensor) from
`model-00001-of-00002.safetensors` of `google/gemma-4-31B-it`. It strips the prefixes
and writes the original BF16 tensors.

## Credits

Google DeepMind released the Gemma 4 31B weights and the native Transformers
implementation. The Gemma team also wrote a [technical report](https://arxiv.org/abs/2607.02770) 
that describes the vision encoder in more detail.

## License

[Apache License 2.0](LICENSE), the same license as the source model.
