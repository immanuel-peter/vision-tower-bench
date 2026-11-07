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
| `projector.py`, `projector_config.json` | | Projector loader built on HF `Mistral3MultiModalProjector` |
| `config.json`, `preprocessor_config.json` | | `PixtralVisionModel` configuration and the parent's `PixtralImageProcessor` settings |
| `export_manifest.json` | | Source pin, shard SHA256, name map and the SHA256 of every source tensor |
| `LICENSE`, `NOTICE` | | The parent's license, copied byte for byte, and the attribution notice |

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

See [`examples/inference.py`](examples/inference.py). The Tower loads with
`AutoModel.from_pretrained`; `projector.py` builds the Projector.

## Validation

[`scripts/validate_mistral_large4.py`](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/validate_mistral_large4.py)
checks this release against an independent forward:

- The source shard matches its pinned LFS SHA256. Every released tensor,
  after undoing the RoPE reordering, equals its source tensor bit for bit.
- The reference is mistral-inference 1.6.0, from its SHA256-pinned wheel. It
  reads the parent's native tensor names and applies RoPE as complex
  multiplication.
- Four fixed images, at 448 and 1540 px, compared at all eight sampled Tower
  depths plus `merged` and `projected`. In FP32 the largest relative error is
  8.6e-6 at 448 px and 2.4e-5 at 1540 px.
- In BF16, the release's error against the FP32 reference is at most 1.10
  times the reference's own BF16 error. The large final-layer channels make
  that floor 17% to 18% relative at Tower L30, for the reference as well.
- A plain rename without the RoPE reordering, a merger that reads its input
  kernel-major, and an added final norm each miss by 95% to 125%.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_mistral_large4_vision.py)
range-reads the vision tensors from the shard that the pinned
`consolidated.safetensors.index.json` names, then writes this repository.

## Credits

Mistral AI released the Mistral Large 4 weights.

## License

The weights keep the parent's
[Mistral Ecosystem Open-Weight License](LICENSE), a modified MIT license.
Keep the copyright and permission notice in copies. A business that operates
a Model as a Service needs a separate commercial license from Mistral before
using the Model or derivative works for any commercial purpose. Read the
LICENSE file for the full terms.
