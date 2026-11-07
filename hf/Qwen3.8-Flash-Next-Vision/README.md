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
| `model.safetensors` | 333 | BF16 Tower and in-model learned merger |
| `config.json` | | Vision-only Transformers configuration |
| `preprocessor_config.json` | | Image preprocessing configuration |
| `examples/inference.py` | | Single-image extraction example |
| `export_manifest.json` | | Pinned source, component counts and file SHA256s |
| `LICENSE` | | Qwen Community License 1.0 |
| `README.md`, `.gitattributes` | | Model card and Hub storage metadata |

The merger lives inside `Qwen3_5VisionModel`; no separate Projector file is needed.

## Architecture

| Component | Details at a 448-pixel square input |
|---|---|
| Tower | 27 blocks, width 1152, 16 heads, 16-pixel patches; `(1, 784, 1152)` |
| Merged | 2x2 spatial grouping; `(1, 196, 4608)` |
| Projector | LayerNorm, linear, GELU, linear; `(1, 196, 2560)` |

## Usage

With PyTorch, Pillow, torchvision and Transformers 5.17 or later installed, run
[`examples/inference.py`](examples/inference.py) on one image from this repository's downloaded snapshot:

```bash
python examples/inference.py test-image.jpg
```

The example uses CUDA BF16 when available, crops to 448 pixels, and prints all three shapes.
It loads the snapshot with `AutoModel` and `AutoImageProcessor`.

## Validation

All 333 tensors equal the pinned parent; fixed-image outputs match its vision implementation.
[Validation and provenance](https://github.com/immanuel-peter/vision-tower-bench/tree/t3/qwen-flash-vision-compatibility/results/evidence/qwen38-flash-next) are retained in the bench.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/t3/qwen-flash-vision-compatibility/scripts/export_qwen3_8_vision.py)
uses `--checkpoint qwen38_flash_next` and reads only the indexed vision shard at the pinned revision.

## Credits

Qwen released the weights and the original vision implementation.

## License

[Qwen Community License 1.0](LICENSE), retained from the parent checkpoint.
