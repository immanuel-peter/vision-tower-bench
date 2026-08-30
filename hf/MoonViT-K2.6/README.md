---
license: other
license_name: kimi-k2.6
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- moonshotai/Kimi-K2.6
---

# MoonViT K2.6

This repository packages the [Kimi K2.6](https://huggingface.co/moonshotai/Kimi-K2.6)
MoonViT Tower and Projector as a self-contained Transformers model. It does not download
the full 64-shard checkpoint or fetch architecture code from another repository at
runtime.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 329 | MoonViT Tower |
| `projector.safetensors` | 6 | Kimi K2.6 Projector |
| `projector_config.json`, `projector.py` | | Projector shapes and loader |
| `config.json`, `configuration_moonvit.py`, `modeling_moonvit.py` | | Standalone Transformers model |
| `preprocessor_config.json`, `kimi_k25_vision_processing.py`, `media_utils.py` | | Image preprocessing |

## Architecture

| Component | Details |
|---|---|
| Tower | 27 layers, 1152 hidden, 16 heads, 4304 intermediate, patch size 14 |
| Token compression | 2x2 spatial regrouping plus temporal pooling, no learned parameters |
| Projector | `LayerNorm(1152)`, flatten four patches, `Linear(4608, 4608)`, GELU, `Linear(4608, 7168)` |

The Tower loads through `AutoModel.from_pretrained(..., trust_remote_code=True)`. The
processor accepts the standard `images=...` interface, and `projector.py` maps the merged
tokens to the Kimi K2.6 language width.

## Usage

See [`examples/inference.py`](examples/inference.py) for image-to-projected-features
inference.

## Validation

The release tests compare all 335 packaged tensors with shards 63 and 64 of the pinned
Moonshot checkpoint using `torch.equal`. Fixed-image merged and projected outputs also
match the parent implementation bit-for-bit on CPU and in BF16 on an NVIDIA A100.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_moonvit_k26.py)
splits the BF16 weights published by
[`exolabs/Kimi-K2.6-vision`](https://huggingface.co/exolabs/Kimi-K2.6-vision) into separate
Tower and Projector files. It derives a vision-only loader and processor from the pinned
Moonshot implementation.

## Credits

Moonshot AI released Kimi K2.6 and its modeling code. Exolabs published the combined
vision-only weight file used by the exporter.

## License

[Kimi K2.6 License](LICENSE), the same license as the source model. The upstream
[third-party notices](THIRD_PARTY_NOTICES.md) are included.
