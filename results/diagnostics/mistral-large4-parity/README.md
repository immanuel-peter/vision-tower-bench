# Mistral Large 4 Tower parity, November 7 2026

`immanuelpeter/PixtralViT-Large-4` is a renamed copy of the vision tensors in
`mistralai/Mistral-Large-4-1T-A52B` at revision
`f12dabdc0c1e99e7f7924461a6f308859b05af4d`, plus a forward
(`vtb/models/pixtral_large4.py`, shipped as `modeling_pixtral_large4.py`)
that follows the parent's numeric order. This directory holds the checks
against an independent implementation, the first failure, and the root
cause. A shape check or a comparison against the converted HF module would
miss a wrong rotary layout or a numeric-order difference.

## Source identity

- `consolidated-00053-of-00054.safetensors` has SHA256
  `d3289cad598653fec78adfd49f939504f92c4c0e9d8b9e66c8750b2e917dde90`, equal to
  its pinned LFS hash. The pinned `consolidated.safetensors.index.json` places
  all 276 vision tensors in that shard: 272 Tower and 4 Projector, all BF16,
  1,637,107,712 and 67,110,912 parameters.
- The byte-range fetch the exporter uses returns the same 276 tensors as the
  full downloaded shard, bit for bit.
- Every released tensor, after undoing the RoPE reordering, equals its source
  tensor bit for bit, and its SHA256 matches `export_manifest.json`; cosine
  is 1 and relative L2 is 0 for all 276. A fresh download of release
  revision `204c1fe3f84c6f017ca12c22732bbde26e6cb182` passes the same check
  ([hf_reload_identity_204c1fe.json](hf_reload_identity_204c1fe.json)).
  Revision `3cd67e852deac482eda4a09f71934f02582f21d8` adds the corrected
  forward and card; its weight files are byte-identical and a fresh download
  passes the same check
  ([hf_reload_identity_3cd67e8.json](hf_reload_identity_3cd67e8.json)).

## Reference

The reference is mistral-inference 1.6.0, from the wheel with SHA256
`ab55f27381836d81e67b323eea21bc0f5f483a9eff8c603d04d9fabde925e836`. It loads
the parent's native names with a strict state dict and applies RoPE as
complex multiplication on interleaved channel pairs. The vision path follows
`Transformer.embed_vision_language_features`: `VisionTransformer`, then
`pre_mm_projector_norm`, `PatchMerger` and `VisionLanguageAdapter`.

mistral-inference calls xformers `memory_efficient_attention`, which is not
installed. Each comparison runs twice, with two substitutes:

- `efficient`: PyTorch's memory-efficient SDPA kernel, which PyTorch adapted
  from the xformers kernel, in the input dtype;
- `explicit`: softmax(q k^T / sqrt(d)) v in float32, head by head.

`tests/test_mistral_large4.py` checks both substitutes, and xformers'
`BlockDiagonalMask` semantics, against direct formulas.

## Open questions from issue 23

- **RoPE permutation.** Mistral rotates channel pairs `(2i, 2i+1)`; HF Pixtral
  rotates `i` with `i + 64`. A plain rename is wrong. The export reorders each
  head's `wq` and `wk` rows, as HF's `PermuteForRope` does for Pixtral. A
  plain rename misses the reference by 95% (448 px) and 125% (1540 px)
  relative error.
- **Merger order.** Both implementations `unfold` a channel-first grid, so
  the 8192 inputs are channel-major over the 2x2 block. A merger that reads
  them kernel-major misses by 120% and 117%.
- **Eps.** Every norm uses 1e-5 on both sides. Setting all of them to 1e-6
  moves outputs by 0.10% to 0.12%.
- **No final norm.** The reference Tower returns the last block's output with
  no further module. Adding an RMSNorm there misses by 99.8%.

## Gates

Fixed before the first run, unchanged since: FP32 relative error at most
1e-4 against the FP32 reference; BF16 relative error against the FP32
reference at most 1.5 times the reference's own BF16 error plus 0.005, with
cosine at least 0.999; negative controls must miss by at least 5%. The
validator also reports the direct comparison of BF16 outputs with the
reference's BF16 outputs against the same 0.999 cosine.

## First failure

The first run (`history/1-first-run-a2277c8-fixed-cosine.json`) passed FP32
everywhere and failed the BF16 gate at deep Stages. A later run loosened the
cosine bound to the reference's own BF16 cosine minus 0.002
(`history/2-old-forward-release-relaxed-bound.json`); that change was
withdrawn and the original bound restored. Both files stay as recorded.

## Root cause

Comparing the two BF16 quantities separately showed that the conversion
itself disagreed with the parent in BF16: converted against parent BF16
reached only cosine 0.9939. Op-by-op checks inside block 1, each op fed
the reference's own input (`cpu_ops_448.json`, then `history/3-*`), found
patch conv, every RMSNorm, the q/k/v/o projections and the MLP bit-exact,
and three numeric-order differences in the forward:

1. **RoPE.** HF Pixtral casts cos/sin to BF16 and rotates in BF16, changing
   25% to 30% of rotated q/k values. Rotating in float32 with float32
   rotate-half left 1-ulp differences (`history/4-*`). The forward now builds
   the parent's `torch.polar` table in float32 and multiplies as complex
   numbers: bit-exact.
2. **Channel order in attention.** In HF's half-split q/k layout the kernel
   sums each q.k product in a different order; 0.03% of attention outputs
   differed (`block_probe.json`). The forward now attends in the parent's
   interleaved layout: bit-exact (`history/5-*`).
3. **Strided `ln_pre` input.** The parent's `torch.cat` makes the patch
   embeddings contiguous; the converted forward normalized a transposed
   view, which changes the RMSNorm reduction order for 0.002% of block-1
   inputs (`input_probe.json`). Thirty blocks amplify that. With a contiguous
   input, all 30 chained blocks match bit for bit (`chain_probe.json`).

## Outcomes at 3bb47be

[parity-3bb47be-efficient.json](parity-3bb47be-efficient.json) and
[parity-3bb47be-explicit.json](parity-3bb47be-explicit.json): four fixed
images (two synthetic in `parity-images/`, ImageNet-100 `validation_000000`
and `validation_006500`), 448 and 1540 px, eight Tower depths plus `merged`
and `projected`, 80 comparisons per run.

| Quantity | `efficient` reference | `explicit` reference |
|---|---|---|
| Converted FP32 vs reference FP32, worst relative error | 0 (bit-exact) | 1.7e-5 |
| Converted BF16 vs reference BF16, bit-exact | 80 of 80 | 0 of 80 |
| Converted BF16 vs reference BF16, cosine at least 0.999 | 80 of 80 | 58 of 80, worst 0.9947 |
| Reference BF16 vs reference FP32, cosine at least 0.999 | 51 of 80, worst 0.9813 | 51 of 80, worst 0.9808 |
| Original BF16 gate (converted BF16 vs reference FP32) | 51 of 80 pass | 51 of 80 pass |
| Negative controls detected | 6 of 6 | 6 of 6 |

- **Conversion parity.** With the xformers-derived kernel, the converted
  Tower, merger and adapter reproduce the reference bit for bit in both
  precisions.
- **Attention-kernel sensitivity.** Against the explicit float32 attention,
  the converted BF16 outputs reach cosine 0.9947. They equal the reference's
  memory-efficient BF16 outputs exactly, so this column is the reference's
  own change when only its attention kernel changes.
- **Checkpoint numeric sensitivity.** The reference's own BF16 forward drifts
  from its FP32 forward at Tower L22, L26 and L30, `merged` and `projected`:
  29 of 80 comparisons fall below cosine 0.999, down to 0.981.
  [bf16_floor_breakdown.json](bf16_floor_breakdown.json) shows why: Tower L30
  has no final norm, and three channels (484, 1657 and 70) reach RMS 35,000
  to 57,000 against a median of 19, with a maximum of 2.7e5. The original
  BF16 gate fails in exactly these 29 comparisons, for the parent and the
  release alike. The gate is not met, and it is not loosened.

The stock HF path, `PixtralVisionModel` without the supplied forward,
reaches cosine 0.9937 against the parent's BF16 forward with the same
kernel (`diagnosis-3bb47be.json`, variant `hf`/`default`).

## Files

- `parity-3bb47be-{efficient,explicit}.json`, `diagnosis-3bb47be.json`: final
  outcomes and the op-level and whole-forward variant comparisons.
- `history/`: every earlier report, in order, named by commit.
- `cpu_ops_448.json`, `block_probe.json`, `chain_probe.json`,
  `input_probe.json`, `probes/`: the localization runs and their scripts
  (arguments: reference wheel directory, ImageNet-100 validation directory,
  output path).
- `bf16_floor_breakdown.json`: channel statistics behind the checkpoint's BF16
  sensitivity.
- `hf_reload_identity_{204c1fe,3cd67e8}.json`: identity checks of fresh
  downloads of both release revisions.

Reproduce with `scripts/validate_mistral_large4.py` (`--reference-attention
efficient|explicit`), `scripts/mistral_large4_bf16_diagnosis.py` and
`scripts/mistral_large4_bf16_floor.py`. `tests/test_release_parity.py` runs
the validator on CUDA.
