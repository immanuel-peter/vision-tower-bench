---
license: apache-2.0
pipeline_tag: image-feature-extraction
library_name: timm
tags:
- vision
- feature-extraction
- perception-encoder
base_model:
- thinkingmachines/Inkling-2-Mini
---

# Inkling-2-Mini-Vision

This repository packages the vision Tower and Projector from [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). The Tower was initialized from Meta's [PE-Lang L](https://huggingface.co/facebook/vit_pe_lang_large_patch14_448_timm) and trained further with the language model. It has the same architecture and initialization as [Inkling-2-Small-Vision](https://huggingface.co/immanuelpeter/Inkling-2-Small-Vision), but its weights differ because each Tower was co-trained with a different language model.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 420 | Tower, the parent's `model.visual.backbone.*` with the prefix `model.visual.` removed |
| `projector.safetensors` | 2 | Projector, the parent's `model.visual.proj_in` and `proj_out` |
| `inkling2_vision.py` | | Loads the Tower onto timm's `vit_pe_lang_large_patch14_448` at patch size 20, plus the 2x2 merge, Projector and pixel normalization |
| `config.json` | | The parent's `vision_config`, the language model width and the native resolution |
| `export_manifest.json` | | Source pin, tensor counts and the vision reference |
| `LICENSE` | | Apache-2.0 |

## Architecture

| Component | Details |
|---|---|
| Tower | 23 blocks, 1024 hidden, 16 heads, patch size 20, 2D RoPE, exact GELU, LayerScale |
| Token compression | 2x2 spatial grouping of 1024-wide patches to 4096, no learned parameters |
| Projector | `Linear(4096, 4096)`, exact GELU, `Linear(4096, 2304)`, no biases |

The native resolution is 640 px, a 32 x 32 patch grid. There the Tower gives 1024 patch tokens of width 1024, the merge 256 tokens of width 4096, and the Projector 256 tokens of width 2304, the language model's hidden size.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/pe.py) compares every released tensor with the pinned parent using `torch.equal`, and checks that none equals the matching tensor in PE-Lang L or Inkling 2 Small. It then compares the Tower depths, the merge and the Projector with the eager per-image reference in vLLM's Inkling 2 vision [test](https://github.com/vllm-project/vllm/pull/67325), run on the parent's tensors.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/pe.py) reads the `model.visual.*` tensors from the pinned parent and writes the original BF16 tensors.

## Credits

Thinking Machines released the Inkling 2 Mini weights. Meta released PE-Lang L and the Perception Encoder code. The vision forward follows the vLLM implementation.

## License

Apache-2.0, as declared on the source model card.
