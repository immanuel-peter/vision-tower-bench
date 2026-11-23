---
license: other
license_name: tencent-hy-license
license_link: LICENSE
base_model: tencent/Hy4
pipeline_tag: image-feature-extraction
library_name: transformers
---

# Hy4 Vision

The standalone BF16 Tower and Projector extracted from [Tencent Hy4](https://huggingface.co/tencent/Hy4/tree/c10c0de5b6f850703fb0a90e65d1c32b558b9167) for frozen-feature probing. The release preserves the 649 vision tensors and removes the language model.

The benchmark uses 512×512 RGB crops normalized with mean and standard deviation 0.5. The 40-block Tower produces a 32×32 raster grid at width 1536. The Projector produces 16×16 grids at width 6144. The `merged` Stage groups RMS-normalized 2×2 neighborhoods. The `projected` Stage emits the convolution output without row delimiters.

Load with `AutoModel.from_pretrained("immanuelpeter/Hy4-Vision", trust_remote_code=True)`. Pass a batch of normalized RGB tensors and the desired one-indexed block depths. The model returns the requested Tower outputs, merged features, and projected features.

The [Tencent Hy License 1.0](LICENSE) governs the weights. See [NOTICE](NOTICE) for attribution and changes.
