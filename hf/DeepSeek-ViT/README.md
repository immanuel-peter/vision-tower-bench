---
license: mit
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
---

# DeepSeek-ViT

This repository packages DeepSeek-ViT and the two-layer MLP projector from
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

Earlier revisions of this repository packaged the original
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) release. The
architecture is unchanged, but every Tower and Aligner tensor differs, so pin a revision
when you need the original weights.

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 259 | DeepSeek-ViT Tower, extracted from source shard 1 |
| `projector.safetensors` | 4 | Two-layer MLP Aligner |
| `config.json` | | Vision-only `DeepSeekV41ViT` |
| `vision.py` | | Standalone ViT and Aligner used by `examples/inference.py` |

This repository does not include Transformers modeling files or a preprocessor
config. `AutoModel.from_pretrained` will not work.

## Architecture

| Component | Details |
|---|---|
| Tower | 32 layers, 1024 hidden, 16 heads, 2816 intermediate, patch size 14, 2D-RoPE |
| Token compression | 3x3 unfold / pixel-unshuffle with padding |
| Aligner | `Linear(9216, 5120)`, GELU, `Linear(5120, 5120)`, both with bias |

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.
`AutoModel.from_pretrained` will not load this repository. The example uses
[`vision.py`](vision.py) in this repo.

## Validation

The [parity check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/parity/deepseek_v41.py)
compares all 259 Tower tensors and 4 Aligner tensors with the pinned parent checkpoint
using `torch.equal`.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/deepseek_v41.py)
reads `vision.*` and `aligner.*` from shard 1 of `deepseek-ai/DeepSeek-V4.1-Flash-1017`, 
removes the prefixes, and writes the original BF16 tensors.

## Credits

DeepSeek released the DeepSeek-V4.1-Flash-1017 weights. DeepSeek-ViT is described in
[DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://www.alphaxiv.org/abs/2609.deepseek-v4-1-flash).

## License

[MIT License](LICENSE), the same license as the source model.
