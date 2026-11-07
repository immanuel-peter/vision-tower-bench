---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: transformers
tags:
- vision
- feature-extraction
- qwen3_5
base_model: Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next Vision

Standalone Tower and Projector extracted from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540).
The source revision is `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
The 333 `model.visual.*` tensors are copied exactly in BF16, with the prefix removed.

The Tower has 27 blocks, width 1152, 16 attention heads, 16-pixel patches,
two-frame temporal patches, and 2 by 2 spatial merging. The Projector input is
4608 wide and its output is 2560 wide.

The release uses `Qwen3_5VisionModel`. The source config's `model_type` is
rewritten to `qwen3_5_vision` and `out_hidden_size` is set to 2560. No tensor is
reshaped or initialized. Strict loading confirms that the model requires no
additional persistent buffer.

## Load

```python
import torch
from transformers import AutoModel, AutoImageProcessor

model = AutoModel.from_pretrained(
    "immanuelpeter/Qwen3.8-Flash-Next-Vision", dtype=torch.bfloat16
).eval()
processor = AutoImageProcessor.from_pretrained(
    "immanuelpeter/Qwen3.8-Flash-Next-Vision"
)
```

Use an immutable release revision for reproducible experiments.
`last_hidden_state` contains Tower tokens in the parent's merge-block order.
`pooler_output` contains projected tokens. The bench adapter restores Tower
tokens to raster order and groups final Tower tokens into the 4608-wide
`merged` Stage. The bench uses a 448-pixel square crop and normalization with
mean and standard deviation 0.5. Parent processor dynamic resizing is separate.

## Validation

All 333 release tensors match the pinned parent exactly. Independent forward
validation uses the parent's `Qwen4ExpVisionModel`, its original config and its
pinned source tensors. At 96 and 448 pixels, two fixed random images match
exactly in CPU FP32, CPU BF16 and CUDA BF16. All eight Tower depths, the merged
Stage and the projected Stage match with `rtol=0` and `atol=0`.
The release also loads independently through `AutoModel`.

Validation used Transformers 5.17.0. Source-code hashes, image hashes and
per-tensor identity are recorded with the bench's issue 22 evidence.
Run `python -m scripts.validate_qwen38_flash_next --out parity.json` from
[vision-tower-bench](https://github.com/immanuel-peter/vision-tower-bench).

All 331 shared-shaped tensors differ from the pinned Qwen3.8-27B checkpoint.
Tower and merged Stage comparisons isolate weight changes. Projected Stage
comparisons also change output width, from 5120 to 2560.

## License

The weights retain the parent's Qwen Community License 1.0, copied verbatim in
[LICENSE](LICENSE). These weights are licensed under qwen-community-1.0.

