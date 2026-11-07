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

Standalone Tower and Projector from [MiniMax M3.1 Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash).
No language-model weights are included.

## Contents

| File | Contents |
|---|---|
| `model.safetensors` | 515 Tower tensors |
| `projector.safetensors` | Eight Projector tensors |
| `projector.py`, `projector_config.json` | Standalone Projector and loader |
| `config.json`, `preprocessor_config.json` | Vision and image-processing configuration |
| `examples/inference.py` | Three-Stage image example |
| `README.md`, `export_manifest.json`, `.gitattributes` | Documentation, source pin, file hashes and storage rules |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | Weight license, helper-code license and attribution |

## Architecture

| Component | Details and batch-one output at 448 pixels |
|---|---|
| Tower | 32 layers, width 1280, 16 heads, MLP 5120, patch 14, 3D RoPE; `(1, 1024, 1280)` |
| Merged | Lossless 2x2 spatial concatenation; `(1, 256, 5120)` |
| Projector | Four linear layers with GELU and learned 2x2 compression; `(1, 256, 6144)` |

Still images repeat as two frames for the temporal patch embedding.
The nominal image size is 672; the example uses a 448-pixel bicubic center crop.

## Usage

Requires Transformers 5.17.0, PyTorch, torchvision and Pillow.
See [examples/inference.py](examples/inference.py) for local bundle loading and BF16 features.

```bash
python examples/inference.py /path/to/image.jpg
```

Forward images independently: native HF attention couples images in one call.

## Validation

Source tensors are preserved exactly, including their original dtypes.
Native-HF reference checks and inference limits are recorded in the [bench evidence](https://github.com/immanuel-peter/vision-tower-bench/tree/t3/complete-issue-21/results/diagnostics/minimax-m31-flash).

## Reproduction

The pinned, index-driven [exporter](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_minimax_m3_vision.py) accepts `--checkpoint minimax_m31_flash`.
`export_manifest.json` records source shards and every distributed file hash except itself.

## Credits

MiniMax released the parent weights; Transformers provides the native Tower implementation.

## License

Weights retain the [MiniMax Community License](LICENSE). Original helper code is [MIT](LICENSE-CODE); see [NOTICE](NOTICE).
