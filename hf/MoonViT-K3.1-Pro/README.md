---
license: other
license_name: kimi-k3-1-pro
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- moonshotai/Kimi-K3.1-Pro
---

# MoonViT-K3.1-Pro

This repository packages the [Kimi K3.1 Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)
MoonViT Tower and Projector.

## Contents

| File | Contents |
|---|---|
| `model.safetensors` | BF16 Tower |
| `projector.safetensors`, `projector.py`, `projector_config.json` | Projector weights, configuration and loader |
| `config.json`, `preprocessor_config.json` | Vision and image-processing configuration |
| `configuration_moonvit_v2.py`, `modeling_moonvit_v2.py`, `image_processing_moonvit_v2.py` | Vision modeling and processing code |
| `examples/inference.py` | Single-image inference |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | Licenses and attribution |
| `README.md`, `export_manifest.json` | Card, pinned source and file hashes |
| `.gitattributes` | Hub-managed storage rules |

Hub-only `.gitattributes` controls storage; the manifest identifies it separately from content hashes.

## Architecture

| Component | Details |
|---|---|
| Tower | 27 layers, 1024 hidden, 12 heads, 1536 QKV, patch size 14 |
| Token compression | 2x2 pixel shuffle plus temporal pooling, no learned parameters |
| Projector | `Linear(4096, 4096)` no bias, GELU, `Linear(4096, 7168)` no bias, `RMSNorm(7168)` |

The Tower output includes the final RMSNorm. At 448 pixels the example reports
Tower `(1, 1024, 1024)`, merged `(1, 256, 4096)` and projected `(1, 256, 7168)`.

## Usage

See [`examples/inference.py`](examples/inference.py) for a complete image-to-projected-features
example: `python examples/inference.py test-image.jpg`. Use batch size 1.

## Validation

The release tests compare all 165 Tower tensors and three Projector tensors with
the pinned parent checkpoint using `torch.equal`. Fixed-image Tower, merged and
projected outputs also match the parent implementation bit-for-bit in FP32 and
in BF16 with mathematical attention. Production vLLM BF16 differs by about 8.1%
relative L2; see [detailed evidence](https://github.com/immanuel-peter/vision-tower-bench/tree/t3/k31-pro-diagnostics/results/diagnostics/kimi-k31-pro-checkpoint/package-evidence).

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_moonvit_v2.py)
extracts `vision_tower` from shard 96 and `mm_projector` from shard 95 of
`moonshotai/Kimi-K3.1-Pro`. It preserves the original BF16 weights and verifies
every exported tensor against the pinned parent before writing.

## Credits

Moonshot AI released the Kimi K3.1 Pro weights. The modeling and image-processing
code comes unchanged from `AI4Industry/MoonViT-V2`.

## License

[Kimi K3.1-Pro License](LICENSE), the same license as the source model. The vendored
code retains its original copyright notices; see `LICENSE-CODE` and `NOTICE`.
