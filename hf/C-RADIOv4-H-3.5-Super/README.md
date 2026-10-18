---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
pipeline_tag: image-feature-extraction
library_name: pytorch
tags:
- vision
- feature-extraction
- nemotron
base_model:
- nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16
---

# C-RADIOv4-H-3.5-Super

This repository packages the C-RADIOv4-H vision Tower, final LayerNorm, and
Projector from
[Nemotron Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)
at revision `97e009758ae21747ee3752d2e43b005fb1e23a61`. No language model
tensors are included.

## Lineage

Nemotron 3 Nano Omni introduced this C-RADIOv4-H Tower
([immanuelpeter/C-RADIOv4-H-3-Omni](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3-Omni)).
Super keeps its architecture and all 390 shared tensor shapes, and
small-tensor comparisons support further training from the Nano Omni Tower.
That is a lineage inference, not a verified training history. These are
Super's own weights, not Nano Omni or standalone
[nvidia/C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H) weights.
[Nemotron Bolt](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3.5-Bolt)
starts from this Tower in turn.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 391 | RADIO Tower from Super `vision_model.*`, including the new `summary_idxs` |
| `final_norm.safetensors` | 2 | `vision_projector.vision_final_layernorm` |
| `projector.safetensors` | 3 | Super `mlp1` Projector |
| `config.json` | | Source and code pins, widths, and epsilons |
| `inventory.json` | | Source shards and the shape, dtype, and SHA256 of every tensor |
| `vision.py` | | Loader used by `examples/inference.py` |

The 390 Nano-shaped Tower tensors span `model-00061-of-00063.safetensors` and
`model-00062-of-00063.safetensors`. `summary_idxs` comes from
`model-summary-idxs.safetensors`, and the final LayerNorm from
`model-00063-of-00063.safetensors`.

## Architecture

| Component | Details |
|---|---|
| Tower | C-RADIOv4-H, 32 layers, 1280 hidden, patch size 16, ten class/register prefix tokens |
| Final LayerNorm | `LayerNorm(1280)`, eps `1e-6`, enabled by Super's MTP setting and applied before the shuffle |
| Token compression | InternVL v2 2x2 pixel shuffle, scale 0.5, no learned parameters |
| Projector (`mlp1`) | `RMSNorm(5120)` eps `1e-5`, `Linear(5120, 20480)` no bias, SquaredReLU, `Linear(20480, 4096)` no bias |

Nano Omni's Projector ends at width 2688; Super's ends at 4096.

## Stages

At a 448 square crop, the Tower emits 784 tokens of width 1280 after the prefix
tokens are dropped. Tower outputs precede the final LayerNorm. `merged` is the
normalized grid after pixel shuffle, 196 tokens of width 5120. `projected` is
196 tokens of width 4096. The benchmark records Tower depths 4, 8, ..., 32
plus `merged` and `projected`. Its fixed crop differs from the parent's dynamic
image tiling.

Pass RGB float32 pixels in `[0, 1]`. The checkpoint's FP32 conditioner
normalizes them before the BF16 forward.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.
`AutoModel.from_pretrained` will not load this repository. The example uses
[`vision.py`](vision.py), which builds RADIO from pinned Nano Omni configuration
and pinned `nvidia/C-RADIOv4-H` remote code. It downloads code and configuration
only; no Nano Omni or standalone C-RADIO weights are loaded.

## Validation

The [release parity test](https://github.com/immanuel-peter/vision-tower-bench/blob/main/tests/test_release_parity.py)
compares every exported tensor with the pinned parent using `torch.equal`. It
then compares all eight Tower depths, `merged`, `projected`, and `vision.py`
against an independent reference built from the parent's own `RadioModel` and
`VisionProjector` code, without a language model. At 448 and 512 pixels in BF16,
all of these outputs were bit exact.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_nemotron35_vision.py)
reads the pinned weight index, checks the prefix counts, and downloads only the
`vision_model.*`, `mlp1.*`, and `vision_projector.vision_final_layernorm.*`
tensor byte ranges. It writes the original tensors and `inventory.json`.

## Credits

NVIDIA released
[Nemotron Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16),
[Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16),
and [C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H).

## License

[OpenMDW-1.1](LICENSE), the license linked from the source model card. The Super
repository has no LICENSE file, so this repository includes the unmodified
official text. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for its
source and the remote code notices.
