---
license: other
license_name: tencent-hy-license
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
- feature-extraction
- hy4
base_model:
- tencent/Hy4
---

# Hy4-Vision

This repository packages the vision Tower and Projector from [Tencent Hy4](https://huggingface.co/tencent/Hy4).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 649 | 643 Tower and 6 Projector tensors from `vit.*` |
| `modeling_hy4.py` | | Standalone Tower and Projector (`trust_remote_code=True`) |
| `config.json` | | Vision configuration and `AutoModel` mapping |
| `export_manifest.json` | | Source pin, shards and tensor counts |

## Architecture

| Component | Details |
|---|---|
| Tower | 40 blocks, 1536 hidden, 6144 MLP, 16 heads, patch size 16, GELU, learned 128 x 128 absolute positions |
| Projector | RMSNorm, `Conv2d(1536, 6144, 2, stride 2)`, GELU, `Conv2d(6144, 6144, 1)` |

At 512 px this gives 1024 Tower tokens (1536-d), 256 merged tokens (6144-d), and 256 projected tokens (6144-d).

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction. `AutoModel.from_pretrained(..., trust_remote_code=True)` loads `modeling_hy4.py`.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/hy4.py) verifies all 649 tensors against the pinned parent with `torch.equal` and compares forward outputs against the `hy_v4_vl` reference in [Transformers](https://github.com/huggingface/transformers/commit/1031277cef60200fda98628e9adc8fb15f0e1549).

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/hy4.py) range-reads the 649 `vit.*` tensors from the pinned parent and writes the original BF16 tensors.

## Credits

Tencent released the Hy4 weights. The vision forward follows Hugging Face Transformers.

## License

Weights use the [Tencent Hy License, Version 1.0](LICENSE) (Section 10 adds terms for large Model-as-a-Service businesses). Code is [MIT](LICENSE-CODE); see [NOTICE](NOTICE) for attribution.
