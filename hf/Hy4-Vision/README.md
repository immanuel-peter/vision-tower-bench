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
| `model.safetensors` | 649 | The parent's BF16 `vit.*` tensors with the `vit.` prefix removed: 643 Tower and 6 Projector |
| `modeling_hy4.py` | | Standalone Tower and Projector for Transformers |
| `config.json` | | Vision configuration and `AutoModel` mapping |
| `export_manifest.json` | | Source pin, source shards, tensor counts and the vision reference |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | | Weight license, code license and attribution |

## Architecture

| Component | Details |
|---|---|
| Tower | 40 pre-norm blocks, width 1536, MLP width 6144, 16 heads of width 96, GELU and LayerNorm with eps 1e-5 |
| Patches | 16 × 16 pixels, with bias in the patch embedding |
| Positions | Learned absolute positions on a 128 × 128 grid. The table has shape `[16385, 1536]` with row 0 unused. Bilinear interpolation uses `scale_factor = (size + 0.1) / 128` and `align_corners=False`. |
| Tower output | The last block feeds the Projector directly. Features contain patch tokens in raster order. |
| Projector | RMSNorm with eps 1e-5, a 2 × 2 stride-2 convolution from 1536 to 6144, GELU, and a 1 × 1 convolution at width 6144 |

Pixels are normalized with mean and std 0.5. At 512 x 512 the Tower gives 1024 tokens of width 1536, the merge 256 tokens of width 6144, and the Projector 256 tokens of width 6144.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction. `AutoModel.from_pretrained(..., trust_remote_code=True)` loads `modeling_hy4.py`; its forward takes `pixel_values` and `depths` and returns `(hidden_by_depth, merged, projected)`.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/hy4.py) compares every released tensor with the pinned parent using `torch.equal`. It then compares the Tower depths, the merge and the Projector with the `hy_v4_vl` implementation in [Transformers commit `1031277`](https://github.com/huggingface/transformers/commit/1031277cef60200fda98628e9adc8fb15f0e1549), run on the parent's tensors.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/hy4.py) range-reads the 649 `vit.*` tensors from eight of the parent's 245 shards (revision `c10c0de`) and writes the original BF16 tensors.

## Credits

Tencent released the Hy4 weights. The vision forward follows the Hugging Face Transformers implementation.

## License

The weights use the [Tencent Hy License, Version 1.0](LICENSE). Section 10 contains terms for large Model-as-a-Service businesses. The standalone modeling code and inference example use the [MIT License](LICENSE-CODE). See [NOTICE](NOTICE) for attribution and extraction details.
