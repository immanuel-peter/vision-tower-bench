---
license: apache-2.0
pipeline_tag: image-feature-extraction
library_name: timm
tags:
- vision
- feature-extraction
- perception-encoder
base_model:
- thinkingmachines/Inkling-2-Small
---

# Inkling-2-Small-Vision

This repository packages the vision Tower and Projector from [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small). The Tower was initialized from Meta's [PE-Lang L](https://huggingface.co/facebook/vit_pe_lang_large_patch14_448_timm) and trained further with the language model.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 420 | Tower, the parent's `model.visual.backbone.*` with the prefix `model.visual.` removed |
| `projector.safetensors` | 2 | Projector, the parent's `model.visual.proj_in` and `proj_out` |
| `inkling2_vision.py` | | Loads the Tower onto timm's `vit_pe_lang_large_patch14_448` at patch size 20, plus the 2x2 merge, Projector and pixel normalization |
| `config.json` | | The parent's `vision_config`, the language model width and the native resolution |
| `export_manifest.json` | | Source pin, tensor counts and the vision reference |
| `LICENSE` | | Apache-2.0 |

All tensors are the parent's BF16 values, unchanged.

## Architecture

| Component | Details |
|---|---|
| Tower | 23 blocks, width 1024, 16 heads, MLP ratio 4, exact GELU, LayerScale, class token, `ln_pre`, no `ln_post`, no pooling head |
| Patches | 20 x 20 pixels, flattened (row, column, channel) by a bias-free linear layer |
| Positions | Learned 32 x 32 grid plus a class-token embedding, and 2D RoPE (theta 10000, interleaved pairs, x then y, patches offset by one, class token at 0) |
| Merge | 2x2 neighbouring patch tokens concatenated top-left, top-right, bottom-left, bottom-right, class token excluded, 4096 wide |
| Projector | `Linear(4096, 4096)`, exact GELU, `Linear(4096, 4608)`, no biases |

The native resolution is 640 px, a 32 x 32 patch grid. There the Tower gives 1024 patch tokens of width 1024, the merge 256 tokens of width 4096, and the Projector 256 tokens of width 4608, the language model's hidden size.

The parent normalizes pixels with the CLIP mean and standard deviation, rounds to BF16, maps the result to [-1, 1] and rounds to BF16 again. `normalize` in `inkling2_vision.py` does the same.

## Usage

See [`examples/inference.py`](examples/inference.py), which needs `timm`, `torch`, `torchvision` and `huggingface_hub`.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/pe.py) compares every released tensor with the pinned parent using `torch.equal`. It then compares the Tower depths, the merge and the Projector with the eager per-image reference in vLLM's Inkling 2 vision test (`tests/models/inkling2/test_vision.py`, [vllm-project/vllm#67325](https://github.com/vllm-project/vllm/pull/67325)), run on the parent's tensors.

## Reproduction

The [export](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/pe.py) range-reads only the `model.visual.*` tensors from the shard that the pinned `model.safetensors.index.json` names. Before writing, it checks that the remap uses all 422 tensors exactly once, that the reshaped shapes are as listed above, and that no Tower tensor equals PE-Lang L's.

## Credits

Thinking Machines released the Inkling 2 Small weights. Meta released PE-Lang L and the Perception Encoder code. The vision forward follows the vLLM implementation.

## License

Apache-2.0, as declared on the source model card.
