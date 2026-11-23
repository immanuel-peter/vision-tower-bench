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

This repository packages the vision Tower and Projector from [Hy4 flash](https://huggingface.co/tencent/Hy4-flash). The Tower was initialized from Google's SigLIP2-So400m-Patch16-512 and trained further with the language model. Tencent extended its learned position table to 128 x 128 so it takes native-resolution input from 512 to 2048 px.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 441 | The parent's `vit.*` tensors with the prefix `vit.` removed: 435 Tower and 6 Projector (`perceive.*`) |
| `modeling_hy4_flash_vision.py` | | Standalone Tower and Projector, loaded with `trust_remote_code` |
| `config.json` | | Vision configuration |
| `export_manifest.json` | | Source pin, source shards, tensor counts and the vision reference |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | | Weight license, code license and attribution |

## Architecture

| Component | Details |
|---|---|
| Tower | 27 pre-norm blocks, 1152 hidden, MLP 4304, 16 heads, exact GELU, LayerNorm eps 1e-5, biases on every Linear |
| Patches | 16 x 16 pixels, patch embedding with bias |
| Positions | Learned absolute only. The 128 x 128 table (`[16385, 1152]`, row 0 unused) is interpolated bilinearly with `scale_factor = (size + 0.1) / 128` and `align_corners=False` |
| Output | No final norm and no class or register tokens: the last block feeds the Projector |
| Projector | RMSNorm (eps 1e-5), `Conv2d(1152, 4608, 2, stride 2)`, GELU, `Conv2d(4608, 3072, 1)`, then a learned `image_newline` token after each row |

Pixels are normalized with mean and std 0.5. At 512 x 512 the Tower gives 1024 tokens of width 1152, the 2x2 merge 256 tokens of width 4608, and the Projector 256 tokens of width 3072, the language model's hidden size, plus 16 newline tokens.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction. `AutoModel.from_pretrained(..., trust_remote_code=True)` loads `modeling_hy4_flash_vision.py`; its `pooler_output` is the Projector output with the newline tokens.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/hy4_flash.py) compares every released tensor with the pinned parent using `torch.equal`. It then compares the Tower depths, the merge and the Projector with the `hy_v4_flash` vision classes in transformers [PR #50903](https://github.com/huggingface/transformers/pull/50903) at `c2ec850`, run on the parent's tensors. FP32 on CPU is bit-exact; BF16 on CUDA is within the parity tolerance.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/hy4_flash.py) range-reads the 441 `vit.*` tensors from 20 of the parent's 148 shards and writes the original BF16 tensors.

## Credits

Tencent released the Hy4 flash weights. Google released SigLIP2. The vision forward follows the Hugging Face transformers implementation.

## License

Weights are under the [Tencent Hy License, Version 1.0](LICENSE), the license of the source model. Section 10 adds terms for large Model-as-a-Service businesses and applies to every recipient. `modeling_hy4_flash_vision.py` and `examples/inference.py` are MIT; see [`LICENSE-CODE`](LICENSE-CODE) and [`NOTICE`](NOTICE).
