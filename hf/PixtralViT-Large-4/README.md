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

This repository packages the Tower and Projector from [Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B).

## Contents

| File | Contents |
|---|---|
| `model.safetensors` | BF16 Tower |
| `projector.safetensors`, `projector.py`, `projector_config.json` | Projector weights, configuration and loader |
| `config.json`, `preprocessor_config.json` | Vision and image-processing configuration |
| `modeling_pixtral_large4.py` | Vision modeling and processing code |
| `examples/inference.py` | Single-image inference |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | Licenses and attribution |
| `README.md`, `export_manifest.json` | Card, pinned source and file hashes |
| `.gitattributes` | Hub-managed storage rules |

Hub-only `.gitattributes` controls storage; the manifest identifies it separately from content hashes.

## Architecture

| Component | Details |
|---|---|
| Tower | 30 blocks, 2048 hidden, 16 heads of 128, SwiGLU 6144, RMSNorm (eps 1e-5), 14 px patches, axial 2D RoPE, no class token, no final norm |
| Token compression | 2x2 patch merge inside the Projector |
| Projector | `RMSNorm(2048)`, 2x2 patch merge to 8192, `Linear(8192, 2048)`, `Linear(2048, 6144)`, GELU, `Linear(6144, 6144)`, no biases |

At 448 pixels the bench reports 1,024 Tower tokens of width 2048, 256 merged tokens
of width 2048 and 256 projected tokens of width 6144.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.
Use `trust_remote_code=True` for the source-compatible BF16 forward.

## Validation

The release tests compare all 276 tensors with the pinned parent checkpoint using
`torch.equal`. Fixed-image Tower, merged and projected outputs also match the parent
implementation bit-for-bit with the same attention kernel at 448 and 1540 pixels.
The parent itself fails 29/80 BF16-versus-FP32 gates; different kernels also change BF16 outputs.
See [numerical evidence](https://github.com/immanuel-peter/vision-tower-bench/tree/t3/resolve-pixtral-vit-reference-parity/results/diagnostics/mistral-large4-parity).

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_mistral_large4_vision.py)
range-reads the 276 vision tensors from the shard the pinned index names, renames
them to HF Pixtral names and reorders each `wq` and `wk` head from interleaved
rotary pairs to rotate-half order. It writes the original BF16 tensors. The script
pins the parent revision.

## Credits

Mistral AI released the weights and mistral-inference reference code.

## License

The weights keep the [Mistral Ecosystem Open-Weight License](LICENSE). The Python files are MIT; see [LICENSE-CODE](LICENSE-CODE).
