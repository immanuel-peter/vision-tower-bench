---
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
---

# C-RADIOv4-H

This repository packages the C-RADIOv4-H copy inside
[Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16).
It is not a clone of NVIDIA's standalone encoder
[nvidia/C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 390 | RADIO Tower, extracted from Omni `vision_model.*` |
| `projector.safetensors` | 3 | Omni `mlp1` Projector |
| `config.json` | | Vision-only `RADIOModel` config, with source and revision |
| `projector_config.json`, `projector.py` | | Projector shapes and loader |

## Architecture

| Component | Details |
|---|---|
| Tower | C-RADIOv4-H, 1280 hidden, patch size 16 |
| Token compression | InternVL v2 2x2 pixel shuffle, scale 0.5, no learned parameters |
| Projector (`mlp1`) | `RMSNorm(5120)`, `Linear(5120, 20480)` no bias, SquaredReLU, `Linear(20480, 2688)` no bias |
| Extra | `video_embedder` Linear on the RADIO patch generator (2-frame tubelets) |

The standalone encoder does not ship this Omni Projector path. Omni SFT later stages
train more than the Projector, so the ViT tensors can differ from
[nvidia/C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H).

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.
`AutoModel.from_pretrained` on this repository will fail. The example builds RADIO
from the Omni `vision_config` (`trust_remote_code=True`) and loads the weights here.

## Validation

The [`parity script`](https://github.com/immanuel-peter/vision-tower-bench/blob/main/tests/test_parity.py)
compares all 390 Tower tensors and three Projector tensors with the pinned Omni
parent using `torch.equal`. That is bit-identity with Omni `vision_model.*` and
`mlp1.*`, not with [nvidia/C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H).

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_nemotron_omni_vision.py)
reads `vision_model.*` (390 tensors) and `mlp1.*` (3 tensors) from shard 1 of
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`. It removes the prefixes and
writes the original BF16 tensors.

## Credits

NVIDIA released [C-RADIOv4-H](https://huggingface.co/nvidia/C-RADIOv4-H) and
[Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16).
The Omni paper is [arXiv:2604.24954](https://arxiv.org/abs/2604.24954).

## License

[NVIDIA Open Model Agreement](LICENSE), the same license as the source model.
