---
license: other
license_name: minimax-community
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- MiniMaxAI/MiniMax-M3.1-Flash
---

# MiniMax-M3.1-Flash Vision

This repository packages the Tower and Projector from
[MiniMax M3.1 Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash).

## Contents

| File | Contents |
|---|---|
| `model.safetensors` | Original-dtype Tower |
| `projector.safetensors`, `projector.py`, `projector_config.json` | Projector weights, configuration and loader |
| `config.json`, `preprocessor_config.json` | Vision and image-processing configuration |
| `examples/inference.py` | Single-image inference |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | Licenses and attribution |
| `README.md`, `export_manifest.json` | Card, pinned source and file hashes |
| `.gitattributes` | Hub-managed storage rules |

Hub-only `.gitattributes` controls storage; the manifest identifies it separately from content hashes.

## Architecture

| Component | Details |
|---|---|
| Tower | 32 layers, 1280 hidden, 16 heads, 5120 intermediate, patch size 14, 3D RoPE, image size 672 |
| Token compression | 2x2 spatial merge, temporal patch size 2 |
| Projector | `Linear(1280, 6144)`, GELU, `Linear(6144, 6144)`, flatten four 6144-wide patches, `Linear(24576, 6144)`, GELU, `Linear(6144, 6144)` |

The class is `MiniMaxM3VLVisionModel`. Still images are repeated as two frames for the
temporal patch embedding. At 448 pixels the example reports Tower `(1, 1024, 1280)`,
merged `(1, 256, 5120)` and projected `(1, 256, 6144)`.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.
Forward images independently to avoid attention coupling in packed batches.
Forward one image per call: the native HF attention couples images in a single call.

## Validation

The release tests compare all 515 Tower tensors and eight Projector tensors with the
pinned parent checkpoint using `torch.equal`. Fixed-image Tower, merged and projected
outputs also match the native parent implementation at 56 and 448 pixels.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_minimax_m3_vision.py)
reads the Tower and Projector tensors from the pinned parent and writes the original
tensors, preserving their dtypes. The script pins the parent revision.

## Credits

MiniMax released the MiniMax-M3.1-Flash weights and the native Transformers implementation.

## License

[MiniMax Community License](LICENSE), the same license as the source model. Helper code
is MIT; see [LICENSE-CODE](LICENSE-CODE) and [NOTICE](NOTICE).
