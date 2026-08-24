# Packed-sequence Towers need flash attention, or batch size one

MoonViT-V2 returns one flat packed sequence for a whole batch, and its published modeling
code dispatches attention three ways. Only `flash_attention_2` consumes `cu_seqlens`.
Both `sdpa_attention` and `eager_attention` instead build a dense `[1, seq, seq]` boolean
mask across the entire packed batch and mask out everything outside each image. Packing N
images therefore costs N times the attention work and throws away all but one N-th of it.

Measured on an A100 80GB with sdpa, throughput falls as batch size rises: 15.5 images per
second at batch 1, 13.4 at 2, 9.6 at 4, and 5.3 at 8
(`docs/measurements/moonvit-v2-a100-batch-sweep.json`). Batch 64 asks for a 192 GiB
allocation and fails. This inverts the usual rule, so extraction for this Tower runs at
batch 1 until flash attention is installed, and any burst that raises the batch size to
"use the GPU properly" makes the run slower rather than faster.

The wider rule for the roster: a Tower that packs sequences is not comparable to one that
does not on throughput alone, and its batch size is a correctness-adjacent setting rather
than a tuning knob. Qwen3.8-27B uses the same native-resolution family, so check its
attention dispatch before sizing its burst.

Installing flash attention before the full extraction is worth it. At batch 1 with sdpa,
an A100 gives 15.5 images per second against 3.0 on the M4 Max, a five times gain on a
model where DINOv2 gains seven and a half. Flash attention would restore large batches and
remove the quadratic term. It is not on the critical path for v1, because batch 1 works and
the cost is bounded, so it is an optimization to attempt before the burst rather than a
gate on it.
