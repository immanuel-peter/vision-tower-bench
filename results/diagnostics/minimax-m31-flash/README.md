# MiniMax M3.1 Flash validation

The Flash checkpoint is `MiniMaxAI/MiniMax-M3.1-Flash` at
`54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`. M3 is pinned at
`f0e1c1e04d40177e4673a22097036854f536e9c0`.

`validation.json` records exact source identity for all 515 Tower tensors and
eight Projector tensors in each bundle, including hashes, shapes and source dtypes.
It also records every tensor's cosine and relative L2 difference between M3 and
Flash. `drift-summary.json` groups the arithmetic means of those per-tensor
statistics by layer and Projector module. These means are not concatenated-weight
cosines.

The independent reference loads source tensors directly into native parent HF
classes, without the adapter's loaders or remapping helpers. Patch packing and
raster restoration use separate coordinate loops. At 56 and 448 pixels all eight
Tower depths, merged tokens and projected tokens match exactly in BF16. The
reference's Transformers version and source hash are recorded. The pinned parent
repository ships no modeling Python file, so this is an HF implementation check,
not parity against a second modeling implementation.

No occurrence of `image_size` exists in the native HF modeling file. Patches and
RoPE positions use runtime `grid_thw`. Changing the nominal field between 672 and
2016 produces exactly equal outputs on both test grids. Architecture and
preprocessing therefore support a training-only sibling comparison at 448.
The Projector's raw tensors retain their source precision despite LM quantization.

`expected-inventory.json` declares the required experiment inventory.
Benchmark completion and per-Stage deltas must be established from the resulting
measurements. Validation alone supplies no capability result.

The [concatenated-weight sweep](layer-drift.json) also pools float64 dot products
and norms across all parameters in each of the 32 layers. This differs from the
arithmetic per-tensor means above. Large-norm parameters can dominate it; the
aggregate layer drift is not monotonic with depth. The full per-tensor data
remain the source for individual modules. The standalone figure is generated
by the retained layer-drift.py script.
