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

This repository packages the Tower and Projector from [MiniMax M3.1 Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 515 | MiniMax-M3.1-Flash vision Tower |
| `projector.safetensors` | 8 | `multi_modal_projector` and remapped `patch_merge_mlp` |
| `projector_config.json`, `projector.py` | | Projector shapes and loader |
| `config.json`, `preprocessor_config.json` | | Vision-only model and image-processing configuration |
| `export_manifest.json` | | Source pin, tensor counts and file hashes |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | | Weight license, code license and attribution |
| `examples/inference.py` | | Single-image extraction example |
| `README.md`, `.gitattributes` | | Model card and Hub storage metadata |

## Architecture

| Component | Details |
|---|---|
| Tower | 32 layers, 1280 hidden, 16 heads, 5120 intermediate, patch size 14, 3D RoPE, image size 672 |
| Token compression | 2x2 spatial merge, temporal patch size 2 |
| Projector | `Linear(1280, 6144)`, GELU, `Linear(6144, 6144)`, flatten four 6144-wide patches, `Linear(24576, 6144)`, GELU, `Linear(6144, 6144)` |

The class is `MiniMaxM3VLVisionModel`. Still images are repeated as two frames for the temporal patch embedding. At 448 pixels the example reports Tower `(1, 1024, 1280)`,
merged `(1, 256, 5120)` and projected `(1, 256, 6144)`.

## Usage

With PyTorch, Pillow, torchvision and Transformers 5.17+, run `python examples/inference.py test-image.jpg`
from the downloaded snapshot; see [the example](examples/inference.py). It uses CUDA BF16 when available.
Forward one image per call: the native HF attention couples images in a single call.

## Validation

The release tests compare all 515 Tower tensors and eight Projector tensors with the
pinned parent checkpoint using `torch.equal`. Fixed-image Tower, merged and projected
outputs also match the native parent implementation at 56 and 448 pixels.

[Validation evidence](https://github.com/immanuel-peter/vision-tower-bench/tree/t3/complete-issue-21/results/diagnostics/minimax-m31-flash).

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_minimax_m3_vision.py)
reads the Tower and Projector tensors from the pinned parent and writes the original
tensors, preserving their dtypes. The script pins the parent revision.

## Credits

MiniMax released the MiniMax-M3.1-Flash weights and the native Transformers implementation.

## License

[MiniMax Community License](LICENSE), the same license as the source model. Helper code is MIT; see [LICENSE-CODE](LICENSE-CODE) and [NOTICE](NOTICE).
