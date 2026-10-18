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
- nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16
---

# C-RADIOv4-H-3.5-Bolt

This repository packages the C-RADIOv4-H vision Tower and Projector from
[Nemotron Bolt](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16)
at revision `a6a1e5030819a130a1ea08aadb1fe42f4cc86777`. No language model
tensors are included.

## Lineage

Nemotron 3 Nano Omni introduced this C-RADIOv4-H Tower
([immanuelpeter/C-RADIOv4-H-3-Omni](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3-Omni)).
[Nemotron Super](https://huggingface.co/immanuelpeter/C-RADIOv4-H-3.5-Super)
keeps its architecture, and small-tensor comparisons support further training
from the Nano Omni Tower. Bolt's parent card describes taking the Super
Tower, retraining its Projector, and applying vision-heavy supervised fine-tuning
that also updates the encoder. These are Bolt's own weights, not Super, Nano
Omni, or standalone [nvidia/C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H)
weights.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 391 | RADIO Tower from Bolt `vision_model.*`, including the new `summary_idxs` |
| `projector.safetensors` | 3 | Bolt `mlp1` Projector |
| `config.json` | | Source and code pins, widths, and epsilons |
| `inventory.json` | | Source shards and the shape, dtype, and SHA256 of every tensor |
| `vision.py` | | Loader used by `examples/inference.py` |

The 390 Nano-shaped Tower tensors span `model-00061-of-00063.safetensors` and
`model-00062-of-00063.safetensors`. `summary_idxs` comes from
`model-summary-idxs.safetensors`. Bolt disables `vision_final_layernorm` and
ships no tensors for it.

## Architecture

| Component | Details |
|---|---|
| Tower | C-RADIOv4-H, 32 layers, 1280 hidden, patch size 16, ten class/register prefix tokens |
| Final LayerNorm | None. Super applies one before the shuffle; Bolt does not |
| Token compression | InternVL v2 2x2 pixel shuffle, scale 0.5, no learned parameters |
| Projector (`mlp1`) | `RMSNorm(5120)` eps `1e-5`, `Linear(5120, 20480)` no bias, SquaredReLU, `Linear(20480, 4096)` no bias |

## Stages

At a 448 square crop, the Tower emits 784 tokens of width 1280 after the prefix
tokens are dropped. `merged` is the raw final Tower grid after pixel shuffle,
196 tokens of width 5120. `projected` is 196 tokens of width 4096. The
benchmark evaluates only `merged` and `projected` for Bolt, plus one final
Tower sanity cell against Super. Its fixed crop differs from the parent's
dynamic image tiling.

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
`vision_model.*` and `mlp1.*` tensor byte ranges. It writes the original tensors
and `inventory.json`.

## Credits

NVIDIA released
[Nemotron Bolt](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16),
[Nemotron Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16),
[Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16),
and [C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H).

## License

[OpenMDW-1.1](LICENSE), the license linked from the source model card. Bolt's
pinned LICENSE file is empty, so this repository includes the unmodified
official text. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for its
source and the remote code notices.
