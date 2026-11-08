---
license: other
license_name: mistral-ecosystem-open-weight-license
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- mistralai/Mistral-Large-4-1T-A52B
---

# PixtralViT-Large-4

This repository packages the Tower and Projector from
[Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B). Attribution notice: 2026 - Mistral.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 272 | Pixtral vision encoder, from the parent's `vision_encoder.*` |
| `projector.safetensors` | 4 | `pre_mm_projector_norm`, `patch_merger` and `vision_language_adapter` |
| `modeling_pixtral_large4.py` | | `PixtralVisionModel` subclass following the parent's numeric order |
| `projector.py`, `projector_config.json` | | Projector shapes and loader |
| `config.json`, `preprocessor_config.json` | | Vision configuration and `PixtralImageProcessor` settings |
| `examples/inference.py` | | Tower, merged and projected features for a 448 x 448 crop |
| `export_manifest.json`, `.gitattributes` | | Source pin, tensor name map, file SHA256; Hub LFS rules |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | | Weight license, code license and attribution |

## Architecture

| Component | Details |
|---|---|
| Tower | 30 blocks, 2048 hidden, 16 heads of 128, SwiGLU 6144, RMSNorm (eps 1e-5), 14 px patches, axial 2D RoPE, no class token, no final norm |
| Projector | `RMSNorm(2048)`, 2x2 patch merge to 8192, `Linear(8192, 2048)`, then `Linear(2048, 6144)`, GELU, `Linear(6144, 6144)`, no biases |

At 448 pixels the bench reports 1,024 Tower tokens of width 2048, 256 merged tokens
of width 2048 and 256 projected tokens of width 6144. Three last-layer Tower channels
exceed RMS 2.5e4; standardize channels before fitting a readout there.

## Usage

[`examples/inference.py`](examples/inference.py) prints those shapes for a 448 x 448 crop.
`AutoModel.from_pretrained("immanuelpeter/PixtralViT-Large-4", trust_remote_code=True)` loads
`modeling_pixtral_large4.py`, which follows the parent's BF16 numeric order; stock
`PixtralVisionModel` loads the same weights but rounds differently there.

## Validation

The release tests compare all 276 tensors with the pinned parent checkpoint using
`torch.equal`. Fixed-image Tower, merged and projected outputs also match the
mistral-inference 1.6.0 reference bit-for-bit in FP32 and BF16 at 448 and 1540
pixels, in 80 of 80 comparisons. The parent itself drifts between BF16 and FP32
at the deepest Stages; this release matches it in each precision ([evidence](https://github.com/immanuel-peter/vision-tower-bench/tree/main/results/diagnostics/mistral-large4-parity)).

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_mistral_large4_vision.py)
range-reads the 276 vision tensors from the shard the pinned index names, renames them to HF Pixtral
names and reorders each `wq` and `wk` head from interleaved to rotate-half order, keeping BF16 values.

## Credits

Mistral AI released the Mistral Large 4 weights and the mistral-inference reference code.

## License

The weights keep the [Mistral Ecosystem Open-Weight License](LICENSE) from the
source model. The Python files are MIT; see [LICENSE-CODE](LICENSE-CODE).
