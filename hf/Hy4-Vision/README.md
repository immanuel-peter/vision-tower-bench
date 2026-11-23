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

This repository packages the 1.16B-parameter vision Tower and 75.5M-parameter Projector from [Tencent Hy4](https://huggingface.co/tencent/Hy4). The weights come from the [pinned parent revision](https://huggingface.co/tencent/Hy4/tree/c10c0de5b6f850703fb0a90e65d1c32b558b9167).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 649 | The parent's BF16 `vit.*` tensors with the `vit.` prefix removed. There are 643 Tower tensors and 6 Projector tensors. |
| `modeling_hy4.py` | | Standalone Tower and Projector for Transformers |
| `config.json` | | Vision configuration and `AutoModel` mapping |
| `export_manifest.json` | | Source repo, revision, tensor count and weight bytes |
| [`examples/inference.py`](examples/inference.py) | | Image preprocessing and feature extraction |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | | Weight license, code license and attribution |

## Architecture

| Component | Details |
|---|---|
| Tower | 40 pre-norm blocks, width 1536, MLP width 6144, 16 heads of width 96, GELU and LayerNorm with eps 1e-5 |
| Patches | 16 × 16 pixels, with bias in the patch embedding |
| Positions | Learned absolute positions on a 128 × 128 grid. The table has shape `[16385, 1536]` with row 0 unused. Bilinear interpolation uses `scale_factor = (size + 0.1) / 128` and `align_corners=False`. |
| Tower output | The last block feeds the Projector directly. Features contain patch tokens in raster order. |
| Projector | RMSNorm with eps 1e-5, a 2 × 2 stride-2 convolution from 1536 to 6144, GELU, and a 1 × 1 convolution at width 6144 |

The benchmark resizes the shorter side of the RGB input to 512 px with bicubic interpolation, then center-crops to 512 × 512. Each channel uses mean and standard deviation 0.5.

| Stage | Shape for one image | Features |
|---|---|---|
| `tower` | `(1, 1024, 1536)` | 32 × 32 patch grid at a requested block |
| `merged` | `(1, 256, 6144)` | RMS-normalized tokens grouped in 2 × 2 neighborhoods |
| `projected` | `(1, 256, 6144)` | Projector output with the learned row delimiters omitted |

## Usage

With PyTorch, Torchvision, Pillow and Transformers 5.17 or later installed, download the bundle and run [the inference example](examples/inference.py).

```bash
hf download immanuelpeter/Hy4-Vision --local-dir Hy4-Vision
python Hy4-Vision/examples/inference.py image.jpg
```

The example prints the final Tower, merged and projected shapes. It uses BF16 on CUDA and FP32 on CPU. Pass `--device cpu` to select CPU explicitly.

`AutoModel.from_pretrained(..., trust_remote_code=True)` loads the standalone model. Its forward accepts normalized `pixel_values` of shape `(batch, 3, 512, 512)` and a tuple of one-indexed `depths`. It returns `(hidden_by_depth, merged, projected)`.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/hy4.py) compares every exported tensor and all three Stages with the pinned parent, using the `hy_v4_vl` implementation in [Transformers commit `1031277`](https://github.com/huggingface/transformers/commit/1031277cef60200fda98628e9adc8fb15f0e1549). All 649 weights and all 10 forward comparisons per run were bit-exact on CUDA in FP32 and BF16 and on CPU in FP32.

## Reproduction

From a [vision-tower-bench](https://github.com/immanuel-peter/vision-tower-bench) checkout, run `python -m vtb.export hy4`. The [exporter](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/hy4.py) range-reads 649 `vit.*` tensors from eight of the parent's 245 shards and preserves the original BF16 values.

## Credits

Tencent released the Hy4 weights. The vision forward follows the Hugging Face Transformers implementation.

## License

The weights use the [Tencent Hy License, Version 1.0](LICENSE). Section 10 contains terms for large Model-as-a-Service businesses. The standalone modeling code and inference example use the [MIT License](LICENSE-CODE). See [NOTICE](NOTICE) for attribution and extraction details.
