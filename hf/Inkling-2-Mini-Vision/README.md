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

This repository packages the vision Tower and Projector from [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). Initialized from Meta's [PE-Lang L](https://huggingface.co/facebook/vit_pe_lang_large_patch14_448_timm), its weights differ from both PE-Lang L and [Inkling-2-Small-Vision](https://huggingface.co/immanuelpeter/Inkling-2-Small-Vision) after co-training with the language model.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 420 | Tower tensors from `model.visual.backbone.*` |
| `projector.safetensors` | 2 | Projector (`proj_in`, `proj_out`) |
| `inkling2_vision.py` | | timm Tower loader, 2x2 merge and Projector |
| `config.json` | | Vision configuration and native resolution |
| `export_manifest.json` | | Source pin and tensor counts |

## Architecture

| Component | Details |
|---|---|
| Tower | 23 blocks, 1024 hidden, 16 heads, patch size 20, 2D RoPE, exact GELU, LayerScale |
| Token compression | 2x2 spatial grouping to 4096, no learned parameters |
| Projector | `Linear(4096, 4096)`, exact GELU, `Linear(4096, 2304)`, no biases |

At 640 px this gives 1024 Tower tokens (1024-d), 256 merged tokens (4096-d), and 256 projected tokens (2304-d).

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/pe.py) verifies all 422 tensors against the pinned parent with `torch.equal`, confirms they differ from PE-Lang L and Inkling 2 Small, and checks forward outputs against vLLM's [reference](https://github.com/vllm-project/vllm/pull/67325).

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/pe.py) reads `model.visual.*` from the pinned parent and writes the original BF16 tensors.

## Credits

Thinking Machines released the Inkling 2 Mini weights; Meta released PE-Lang L. The vision forward follows vLLM.

## License

[Apache-2.0](LICENSE), as declared on the source model card.
