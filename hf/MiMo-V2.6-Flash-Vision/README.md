---
license: mit
pipeline_tag: image-feature-extraction
library_name: transformers
tags:
- vision
- feature-extraction
- mimo
base_model:
- XiaomiMiMo/MiMo-V2.6-Flash-RL
---

# MiMo-V2.6-Flash Vision

This repository packages the vision Tower and merger from
[MiMo V2.6 Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) at revision
`dcd3fd83fb181622fbc606fbb1709901982a62c2`. It contains all 364 `visual.*` tensors and no language or audio
tensors. The benchmark reports it as MiMo 2.6 Flash (`mimo26_flash`).

MiMo 2.6 Pro and MiMo 2.6 Flash share the vision architecture but not their
weights: 361 of the 364 tensors differ. Only `blocks.24.norm2.weight`,
`blocks.25.norm2.weight` and `blocks.26.norm2.weight` are equal. The
Pro Tower is at
[immanuelpeter/MiMo-V2.6-Pro-Vision](https://huggingface.co/immanuelpeter/MiMo-V2.6-Pro-Vision).

## Contents

| File | Tensors | What it holds |
|---|---|---|
| `model.safetensors` | 364 | Tower blocks, patch embedding and merger, from the parent's `visual.*` |
| `modeling_mimo_vision.py` | | Vision classes extracted from the pinned parent, loaded with `trust_remote_code` |
| `config.json`, `preprocessor_config.json` | | Vision configuration and the parent's `Qwen2VLImageProcessor` settings |
| `export_manifest.json` | | Source pin, byte count and the SHA256 of every tensor |
| `LICENSE`, `LICENSE-CODE`, `NOTICE` | | Weight and code licenses |

## Architecture

| Component | Details |
|---|---|
| Tower | 28 blocks, 1280 hidden, SwiGLU 4608, 32 query heads, 8 KV heads, head dimension 64, RMSNorm, 2D RoPE |
| Patches | 2 frames x 16 x 16 pixels; a still image is repeated as two frames |
| Attention | Full attention at blocks 0, 9, 18 and 27 (zero-based). The other blocks use a ±64-token window and a per-head sink bias added to key column 0 |
| Token order | Tokens are grouped by 2x2 merge block. Blocks 5-8, 14-17 and 23-26 reorder the blocks column-major; the others keep row-major order |
| Merger | `LayerNorm(1280)` without bias, 2x2 concatenation to 5120, `Linear(5120, 5120)`, GELU, `Linear(5120, 4096)`, no biases |

## Stages

At a 448 square crop, each Tower depth has 784 tokens of width 1280. The
benchmark restores every captured depth to raster order: it undoes that depth's
row- or column-major block order for each image, then the 2x2 order inside each
block. `merged` is the normalized 5120-wide input to the merger MLP, 196
tokens. `projected` is 196 tokens of width 4096. The benchmark records
Tower depths 4, 7, 10, 14, 18, 21, 24 and 28 plus `merged` and `projected`.

The benchmark resizes the short side with bicubic interpolation, center-crops,
then applies the parent's ImageNet mean and standard deviation. Resolutions must
be multiples of 32. The parent processor's dynamic aspect-ratio resizing is a
different protocol.

## Usage

See [`examples/inference.py`](examples/inference.py) for image feature extraction.
`AutoModel.from_pretrained(..., trust_remote_code=True)` loads
`modeling_mimo_vision.py`. The forward takes the processor's `pixel_values` and
`image_grid_thw` and returns packed projected tokens in the parent's merge-block
order. For raster-ordered Tower depths, use `MiMo26FlashAdapter` in the
[benchmark](https://github.com/immanuel-peter/vision-tower-bench/blob/main/vtb/adapters/mimo26.py).

## Parent loading defect

The pinned parent's vision class creates `merger.ln_q.bias`,
`merger.mlp.0.bias` and `merger.mlp.2.bias`, but the checkpoint has none of
them, so the unmodified parent fails a strict load. This release omits the three
parameters. The parity reference loads the parent class, sets only those biases
to zero, and leaves its forward untouched. SGLang
(`b234b5c32e0d65e04a61912f85fc09e95e74a328`, `mimo_vl.py`) also keeps a
bias-free LayerNorm and zero merger biases. vLLM
(`03ee2e134d6cc60c4cc84096c8492fcc971f0526`, `mimo_v2_omni.py`) normalizes the
merger input with RMSNorm instead. Swapping only that norm on the parity images
at 448 px moves projected outputs by up to 0.101 (RMS 0.0135, cosine 0.9735). This release
follows HF and SGLang; projected parity with vLLM is not established.

The vision attention matches vLLM's sink handling: both add the sink to key
column 0 inside the window, rather than appending a separate sink column. See the
[attention sink check](https://github.com/immanuel-peter/vision-tower-bench/blob/main/results/diagnostics/mimo26-attention-sinks/README.md).

## Validation

The [release parity test](https://github.com/immanuel-peter/vision-tower-bench/blob/main/tests/test_release_parity.py) compares
every exported tensor with the pinned parent using `torch.equal`. It then runs
two noise images with different aspect ratios as one batch, at 96 and 448 px,
through the zero-completed parent and the benchmark adapter. A coordinate
scatter, written independently of the adapter, restores the parent's tokens to
raster order. In BF16 on an NVIDIA L40S, all eight Tower depths and `merged`
were bit exact. Projected outputs from the adapter and from `AutoModel` loading
of this release differed by at most 0.0078125 at both resolutions. The parent's zero biases can select
different BF16 kernels than the bias-free modules. Adapter preprocessing matches
the parent's image processor to within 2.4e-7.

## Reproduction

The [export script](https://github.com/immanuel-peter/vision-tower-bench/blob/main/scripts/export_mimo26_vision.py) reads the
pinned weight index and downloads only the `visual.*` byte ranges, 1,457,188,864
bytes from `model_pp0_ep0_shard0.safetensors`. It checks every tensor for equality with the
parent, writes the original BF16 tensors, and copies `modeling_mimo_vision.py`
from `vtb/models/mimo26.py`.

## Credits

Xiaomi released the MiMo V2.6 Flash weights and the vision implementation.

## License

The weights are MIT, as declared on the pinned model card; the parent repository
has no LICENSE file, so [`LICENSE`](LICENSE) reproduces the MIT terms with
Xiaomi's copyright. `modeling_mimo_vision.py` keeps its Apache-2.0 header from
Xiaomi and Hugging Face; see [`LICENSE-CODE`](LICENSE-CODE) and [`NOTICE`](NOTICE).
