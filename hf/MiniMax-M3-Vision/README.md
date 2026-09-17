---
license: other
license_name: minimax-community
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- MiniMaxAI/MiniMax-M3
---

# MiniMax-M3 Vision

This repository packages the Tower and Projector from
[MiniMax M3](https://huggingface.co/MiniMaxAI/MiniMax-M3).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 515 | MiniMax-M3 vision tower |
| `projector.safetensors` | 8 | `multi_modal_projector` and remapped `patch_merge_mlp` |
| `projector_config.json`, `projector.py` | | Projector shapes and loader |
| `config.json`, `preprocessor_config.json` | | Vision-only model and image-processing configuration |
| `LICENSE` | | MiniMax Community License |

## Architecture

| Component | Details |
|---|---|
| Tower | CLIP-style ViT, 32 layers, 1280 hidden, 16 heads, 5120 intermediate, patch size 14, Conv3d patch embed, 3D RoPE, GELU, image size 2016 |
| Token compression | 2x2 spatial merge, temporal patch size 2 |
| Projector | `Linear(1280, 6144)`, GELU, `Linear(6144, 6144)`, flatten four 6144-wide patches, `Linear(24576, 6144)`, GELU, `Linear(6144, 6144)` |

The class is `MiniMaxM3VLVisionModel` (`model_type` `minimax_m3_vl_vision`).
`last_hidden_state` is the raw 1280-wide Tower tokens. At 448 the bench reports 1280-wide
Tower tokens, 5120-wide merged tokens (2x2 of 1280), and 6144-wide projected tokens.
`projector_config.json` records `input_size` 1280, `hidden_size` 6144, `output_size` 6144,
and `merged_hidden_size` 24576.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.

## Validation

The [parity script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/tests/test_parity.py)
compares all 515 Tower tensors and eight Projector tensors with the pinned parent
checkpoint using `torch.equal`.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_minimax_m3_vision.py)
reads `vision_tower.vision_model.*` from shard 59 of `MiniMaxAI/MiniMax-M3`.
It remaps `encoder.layers` to `layers` and `embeddings.patch_embedding` to `embeddings.proj`, 
then writes 515 Tower tensors. The Projector comes from `multi_modal_projector.*` and `patch_merge_mlp.*` 
in shards 26 and 59, with merge linear names remapped, for eight tensors. The original BF16 weights 
are preserved.

## Credits

MiniMax released the MiniMax-M3 weights and the native Transformers implementation.

## License

[MiniMax Community License](LICENSE), the same license as the source model.
