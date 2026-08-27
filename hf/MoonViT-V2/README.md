---
license: other
license_name: kimi-k3
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- moonshotai/Kimi-K3
---

# MoonViT-V2

This repository packages the [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)
MoonViT-V2 Tower and Projector. It reproduces K3's image path without downloading the full
96-shard, 2.8T-parameter checkpoint. [AI4Industry/MoonViT-V2](https://huggingface.co/AI4Industry/MoonViT-V2)
publishes the Tower alone.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 165 | MoonViT-V2 Tower, bit-identical to `AI4Industry/MoonViT-V2` |
| `projector.safetensors` | 3 | Kimi K3 `patchmergerv2` |
| `projector_config.json` | | Projector shapes, read from the weights |
| `config.json`, `configuration_moonvit_v2.py`, `modeling_moonvit_v2.py`, `image_processing_moonvit_v2.py`, `preprocessor_config.json` | | Vendored from `AI4Industry/MoonViT-V2` |

## Architecture

| Component | Details |
|---|---|
| Tower | 27 layers, 1024 hidden, 12 heads, 1536 QKV, patch size 14 |
| Token compression | 2x2 pixel shuffle plus temporal pooling, no learned parameters |
| Projector | `Linear(4096, 4096)` no bias, GELU, `Linear(4096, 7168)` no bias, `RMSNorm(7168)` |

## Usage

See [`examples/inference.py`](examples/inference.py) for a complete image-to-projected-features
example.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_moonvit_v2.py)
extracts `mm_projector` from shard 95 and `vision_tower` from shard 96 of
`moonshotai/Kimi-K3`. It preserves the original BF16 weights and verifies all 165 Tower
tensors against `AI4Industry/MoonViT-V2` before writing.

## Credits

Moonshot AI released the Kimi K3 weights. The modeling and image-processing code comes
unchanged from `AI4Industry/MoonViT-V2`.

## License

[Kimi K3 License](LICENSE), the same license as the source model.
