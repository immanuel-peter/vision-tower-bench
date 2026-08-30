# Packed-sequence Towers need flash attention, or batch size one

MoonViT-V2 returns one flat packed sequence for a whole batch, and its published modeling
code dispatches attention three ways. Only `flash_attention_2` reads `cu_seqlens`. Both
`sdpa_attention` and `eager_attention` build a dense `[1, seq, seq]` boolean mask across
the entire packed batch and then mask out everything outside each image. Packing N images
costs N times the attention work, then masks out all but one N-th of it.

Measured on an A100 80GB with sdpa, throughput falls as batch size rises: 15.5 images per
second at batch 1, 13.4 at 2, 9.6 at 4, and 5.3 at 8
(`docs/measurements/moonvit-v2-a100-batch-sweep.json`). Batch 64 asks for a 192 GiB
allocation and fails. Run extraction for this Tower at batch size 1 unless flash attention
is installed. Increasing the batch size on the fallback path makes the run slower.

Throughput numbers are not comparable between a Tower that packs sequences this way and
one that does not. Inspect each Tower's attention dispatch before choosing a batch size;
model family alone does not determine it.

Try installing flash attention before the full extraction. At batch 1 with sdpa, an A100
runs MoonViT-V2 at 15.5 images per second, compared with 3.0 on the M4 Max. DINOv2 is
7.6 times faster on the same A100, while MoonViT-V2 is 5.2 times faster. Flash attention
should remove the packed-mask cost and permit larger batches, but batch 1 is fast enough
to keep v1 moving if setup fails.

## MoonViT attention dispatch causes the penalty

The roster run measured the other two packed Towers over 768 ImageNet-100 images, one
model at a time. Throughput rises with batch size for both:

| batch | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| qwen3_5 img/s | 14.33 | 24.93 | 40.08 | **42.52** | 36.84 |
| muse_glimmer img/s | 9.93 | 13.04 | **14.79** | 11.70 | |

So run qwen3_5 at batch 8 and muse_glimmer at batch 4. Batch size 1 costs qwen3_5 a factor
of three. Read the muse batch-8 point with care: it overlapped extraction on the other
three GPUs, while batch 4 was measured clean.

The ADR's original rule was too broad. Packing alone does not cause the penalty.
MoonViT-V2 and Kimi K2.6 fall back to an implementation that materializes a dense mask
over the whole packed batch; the transformers implementations of the other two do not.
Test the Tower rather than infer from the family.

CPU contention mattered more than batch size on the rented box. Torch takes one intra-op
thread per core in every process, so four unbounded extraction lanes on 46 cores drove
load average to 90 and left the GPUs under 20 percent, running MoonViT-V2 at 1.5 img/s
against the 15.5 above. Capping threads per lane took MoonViT-V2 to 12.3, Kimi K2.6 to
16.4 and SigLIP2 to 21.5. Workers and Torch threads compete for the same cores, so cap
the threads before raising `--workers` (`docs/measurements/roster-run-2026-08-29.md`).
