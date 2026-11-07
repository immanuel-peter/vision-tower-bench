---
license: other
license_name: kimi-k3-1-pro
license_link: LICENSE
pipeline_tag: image-feature-extraction
library_name: transformers
base_model:
- moonshotai/Kimi-K3.1-Pro
---

# MoonViT K3.1 Pro

The Tower and Projector from Moonshot AI's Kimi K3.1 Pro at
`6d0a0202956796543364d666ab19c6ad1a494ee1`. This bundle contains 165 Tower
tensors and 3 Projector tensors in their original BF16 dtype. The export checks
every tensor against the pinned parent.

The architecture matches Kimi K3. The Tower has 27 blocks of width 1024.
The spatial merger groups each 2x2 patch square into a 4096-wide token.
The Projector produces 7168-wide tokens. The final Tower extraction includes
the final RMSNorm.

Use batch size 1 for reproducible attention. The bench uses a 448px bicubic
short-side resize and center crop. This square-crop protocol differs from the
full model's dynamic image preprocessing.

`export_manifest.json` records checkpoint, code and file provenance.
The standalone modeling code comes from AI4Industry/MoonViT-V2 at
`c5acb88baf02a94e68ca8a225f7eebee152c1fc6`.
Independent parity uses official vLLM vision source registered for Pro,
with only engine dependencies replaced by single-rank PyTorch operations.
See the bench's reproducibility evidence for numerical errors and substitutions.

## License

Weights retain Moonshot AI's custom Kimi K3.1-Pro License, copied byte for byte
from the pinned parent into LICENSE. They are not covered by the bench's MIT
license. Vendored standalone code retains its original copyright notices.
See NOTICE for source attribution.

## Interpretation

K3 versus K3.1 Pro is a matched vision-architecture checkpoint comparison.
The released checkpoints do not isolate which training stage caused weight
changes. Frozen-feature measurements cannot establish an RL-only causal
effect or explain full-model coding, finance, CAD, or factuality scores.

## Numerical backend scope

Independent validation executes official Pro-supported vLLM source at
`1ecc9cc4ce6626352df39e4364579ec376c10761`, with Pro tensors loaded directly
from the pinned source shards. The official packed Triton rotary kernel on CUDA, or official native rotary
on CPU, with the bench's masked SDPA math attention gives bit-exact BF16 outputs for three fixed 448px images, every tapped Tower depth, merged and
projected. K3 regression passes the same comparison.

The directly executed official vLLM TORCH_SDPA wrapper agrees within the
recorded FP32 tolerances but differs in BF16. With the official packed rotary
kernel held fixed, changing only the attention execution causes this difference.
The maximum per-slice relative L2 difference in the Pro BF16 diagnostic is
about 8.1%. Do not interpret this bundle as bit-identical to production vLLM
BF16 execution. The original tolerances and failed diagnostics are preserved.
The bench's primary attention behavior has not changed.

The release includes the exact-parent tensor manifest, pinned source hashes
and numerical validation payloads. Full frozen-feature capability measurements
are pending and no model-quality advantage is claimed.
