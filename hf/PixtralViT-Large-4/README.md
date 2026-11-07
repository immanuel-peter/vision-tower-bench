---
license: other
license_name: mistral-ecosystem-open-weight-license
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
- feature-extraction
- pixtral
base_model:
- mistralai/Mistral-Large-4-1T-A52B
---

# PixtralViT-Large-4

This repository packages the vision Tower and Projector from
[Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B).

Attribution notice: 2026 - Mistral

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 272 | Pixtral vision encoder, from the parent's `vision_encoder.*` |
| `projector.safetensors` | 4 | `pre_mm_projector_norm`, `patch_merger` and `vision_language_adapter` |
| `modeling_pixtral_large4.py` | | `PixtralVisionModel` subclass whose forward follows the parent's numeric order; loaded with `trust_remote_code` |
| `projector.py`, `projector_config.json` | | Projector loader built on HF `Mistral3MultiModalProjector` |
| `config.json`, `preprocessor_config.json` | | Vision configuration with `auto_map`, and the parent's `PixtralImageProcessor` settings |
| `export_manifest.json` | | Source pin, shard SHA256, name map and the SHA256 of every source tensor |
| `LICENSE`, `NOTICE` | | The parent's license, copied byte for byte, and the attribution notice |
| `LICENSE-CODE` | | MIT license for the Python files |

## Architecture

| Component | Details |
|---|---|
| Tower | 30 blocks, 2048 hidden, 16 heads of 128, SwiGLU 6144, RMSNorm (eps 1e-5), 14 px patches, axial 2D RoPE (theta 10000), no class token, no learned position embedding |
| Final norm | None. The last block's output is the raw residual stream |
| Projector | `RMSNorm(2048)`, 2x2 patch merge to 8192, `Linear(8192, 2048)`, then `Linear(2048, 6144)`, GELU, `Linear(6144, 6144)`, no biases |

The Tower has 1.64B parameters and the Projector 67M. At 448 px the bench
reports 1,024 Tower tokens of width 2048, 256 `merged` tokens of width 2048
(the patch-merger output) and 256 `projected` tokens of width 6144.

The final Tower layer has a few very large channels. On one ImageNet image at
448 px, channels 484, 1657 and 70 reach RMS 35,000 to 57,000 against a median
of 19, and the largest value is 2.7e5. Consider standardizing channels before
fitting a linear readout or PCA on that layer.

## Conversion

The parent ships only Mistral-native `consolidated-*` shards. The export reads
the 276 vision tensors from `consolidated-00053-of-00054.safetensors` and
renames them to HF Pixtral and Mistral3 names. Mistral applies RoPE to
interleaved channel pairs; HF Pixtral rotates halves. Each head's rows of
`wq` and `wk` are therefore reordered from pair `(2i, 2i+1)` to `(i, i+64)`.
No value is cast or rounded.

## Usage

```python
from transformers import AutoModel

tower = AutoModel.from_pretrained("immanuelpeter/PixtralViT-Large-4", trust_remote_code=True, dtype="bfloat16")
```

See [`examples/inference.py`](examples/inference.py) for the Tower, `merged`
and `projected` outputs. `projector.py` builds the Projector.

## Numerics: the supplied code and stock transformers

`trust_remote_code` loads `PixtralLarge4VisionModel` from
`modeling_pixtral_large4.py`. It is a `PixtralVisionModel` subclass with the
same weights and a forward that follows the parent's reference code,
mistral-inference, in BF16:

- patch embeddings are made contiguous before `ln_pre`, as the parent's
  `torch.cat` does;
- the rotary table is built in float32 with `torch.polar`, and q/k are
  rotated by complex multiplication in float32 before a cast back to BF16;
- attention runs on q/k in the parent's interleaved channel order, with
  PyTorch's memory-efficient SDPA kernel, which PyTorch adapted from the
  xformers kernel mistral-inference calls.

Stock `PixtralVisionModel.from_pretrained` (without `trust_remote_code`)
loads the same weights and agrees in float32, but in BF16 it rounds
differently. It casts cos/sin to BF16, rotates in BF16, attends in the
half-split channel order and normalizes a strided view. Against the parent's
BF16 forward, its outputs reach cosine 0.9937 at the deepest Stages, against
bit-identical outputs for the supplied code. It also warns about the
`pixtral_large4_vision` model type. Use the supplied code for features that
should match the parent in BF16.

## Validation

[`scripts/validate_mistral_large4.py`](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/validate_mistral_large4.py)
loads this repository with `trust_remote_code` and compares it with an
independent forward. The reference is mistral-inference 1.6.0, from its
SHA256-pinned wheel, run with the parent's native tensor names. xformers is
not installed, so its `memory_efficient_attention` is replaced by PyTorch's
memory-efficient SDPA kernel; a second run uses explicit float32 attention.
The check covers four fixed images at 448 and 1540 px, all eight sampled
Tower depths plus `merged` and `projected`, 80 comparisons per run.

- The source shard matches its pinned LFS SHA256. Every released tensor,
  after undoing the RoPE reordering, equals its source tensor bit for bit.
- Against the reference with the memory-efficient kernel, this release is
  bit-identical in float32 and in BF16, in all 80 comparisons.
- Against the reference with explicit float32 attention, float32 relative
  error is at most 1.7e-5. In BF16 the cosine is at least 0.9947. That gap is
  the reference's own sensitivity to the attention kernel: this release's
  BF16 output equals the reference's memory-efficient BF16 output exactly.
- The parent's BF16 forward itself drifts from its float32 forward at deep
  Stages (cosine down to 0.981 at Tower L30, `merged` and `projected`), because
  of the large final-layer channels. The predeclared check of BF16 against
  float32 at cosine 0.999 therefore fails in 29 of 80 comparisons, for the
  parent and this release alike.
- A plain rename without the RoPE reordering, a merger that reads its input
  kernel-major, and an added final norm each miss by 95% to 125%.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_mistral_large4_vision.py)
range-reads the vision tensors from the shard that the pinned
`consolidated.safetensors.index.json` names, then writes this repository,
including `modeling_pixtral_large4.py` from the bench's
`vtb/models/pixtral_large4.py`.

## Credits

Mistral AI released the Mistral Large 4 weights and the mistral-inference
reference code.

## License

The weights keep the parent's
[Mistral Ecosystem Open-Weight License](LICENSE), a modified MIT license.
Keep the copyright and permission notice in copies. A business that operates
a Model as a Service needs a separate commercial license from Mistral before
using the Model or derivative works for any commercial purpose. Read the
LICENSE file for the full terms. `modeling_pixtral_large4.py`, `projector.py`
and the example are MIT, from the vision-tower-bench repository; see
`LICENSE-CODE`.
