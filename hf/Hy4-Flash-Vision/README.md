---
license: other
license_name: tencent-hy-license
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
- feature-extraction
- siglip2
base_model:
- tencent/Hy4-flash
---

# Hy4-Flash-Vision

This repository packages the vision Tower and Projector from [Hy4 flash](https://huggingface.co/tencent/Hy4-flash), initialized from [SigLIP 2 So400m](https://huggingface.co/google/siglip2-so400m-patch14-384) and extended to native resolutions from 512 to 2048 px.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 441 | 435 Tower and 6 Projector (`perceive.*`) tensors from `vit.*` |
| `modeling_hy4_flash_vision.py` | | Standalone Tower and Projector (`trust_remote_code=True`) |
| `config.json` | | Vision configuration |
| `export_manifest.json` | | Source pin, shards and tensor counts |

## Architecture

| Component | Details |
|---|---|
| Tower | 27 blocks, 1152 hidden, 4304 MLP, 16 heads, patch size 16, GELU, learned 128 x 128 absolute positions |
| Projector | RMSNorm, `Conv2d(1152, 4608, 2, stride 2)`, GELU, `Conv2d(4608, 3072, 1)`, learned `image_newline` per row |

At 512 px this gives 1024 Tower tokens (1152-d), 256 merged tokens (4608-d), and 256 projected tokens (3072-d) plus 16 newline tokens.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction. `AutoModel.from_pretrained(..., trust_remote_code=True)` loads `modeling_hy4_flash_vision.py`.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/hy4_flash.py) verifies all 441 tensors against the pinned parent with `torch.equal` and compares forward outputs against the `hy_v4_flash` reference in Transformers [PR #50903](https://github.com/huggingface/transformers/pull/50903).

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/hy4_flash.py) range-reads the 441 `vit.*` tensors from the pinned parent and writes the original BF16 tensors.

## Credits

Tencent released the Hy4 flash weights; Google released SigLIP 2. The vision forward follows Hugging Face Transformers.

## License

Weights use the [Tencent Hy License, Version 1.0](LICENSE) (Section 10 adds terms for large Model-as-a-Service businesses). Code is [MIT](LICENSE-CODE); see [NOTICE](NOTICE) for attribution.
