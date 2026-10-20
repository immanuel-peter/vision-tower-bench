---
license: mit
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
- feature-extraction
- mimo
base_model:
- XiaomiMiMo/MiMo-V2.6-Pro-RL
---

# MiMoViT-2.6-Pro

This repository packages the vision Tower and merger from [MiMo V2.6 Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 364 | Tower blocks, patch embedding and merger, from the parent's `visual.*` |
| `modeling_mimo_vision.py` | | Vision classes extracted from the parent, loaded with `trust_remote_code` |
| `config.json`, `preprocessor_config.json` | | Vision configuration and the parent's `Qwen2VLImageProcessor` settings |
| `export_manifest.json` | | Source pin and the SHA256 of every tensor |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | | Weight and code licenses |

## Architecture

| Component | Details |
|---|---|
| Tower | 28 blocks, 1280 hidden, SwiGLU 4608, 32 query heads, 8 KV heads, head dimension 64, RMSNorm, 2D RoPE |
| Patches | 2 frames x 16 x 16 pixels; a still image is repeated as two frames |
| Attention | Full attention at blocks 0, 9, 18 and 27; the other blocks use a ±64-token window and a per-head sink bias on key column 0 |
| Merger | `LayerNorm(1280)`, 2x2 concatenation to 5120, `Linear(5120, 5120)`, GELU, `Linear(5120, 6144)`, no biases |

At 448 the bench reports 784 Tower tokens of width 1280 (restored to raster
order), 196 `merged` tokens of width 5120, and 196 `projected` tokens of width 6144.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.
`AutoModel.from_pretrained(..., trust_remote_code=True)` loads `modeling_mimo_vision.py`.

## Validation

The [release parity test](https://github.com/immanuel-peter/vision-tower-bench/blob/main/tests/test_release_parity.py) compares every exported tensor with the pinned parent using `torch.equal`, then checks all stages against the parent's own `MiMoVisionTransformer`.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_mimo26_vision.py) reads the `visual.*` tensors from the pinned parent and writes the original BF16 tensors.

## Credits

Xiaomi released the MiMo V2.6 Pro weights and the vision implementation.

## License

Weights are MIT, as declared on the source model card. `modeling_mimo_vision.py` keeps its Apache-2.0 header; see [`LICENSE-CODE`](LICENSE-CODE) and [`NOTICE`](NOTICE).
