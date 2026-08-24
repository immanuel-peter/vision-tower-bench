# Packed-sequence Towers need flash attention, or batch size one

MoonViT-V2 returns one flat packed sequence for a whole batch, and its published modeling
code dispatches attention three ways. Only `flash_attention_2` reads `cu_seqlens`. Both
`sdpa_attention` and `eager_attention` build a dense `[1, seq, seq]` boolean mask across
the entire packed batch and then mask out everything outside each image. Packing N images
costs N times the attention work and discards all but one N-th of the result.

Measured on an A100 80GB with sdpa, throughput falls as batch size rises: 15.5 images per
second at batch 1, 13.4 at 2, 9.6 at 4, and 5.3 at 8
(`docs/measurements/moonvit-v2-a100-batch-sweep.json`). Batch 64 asks for a 192 GiB
allocation and fails. Extraction for this Tower therefore runs at batch 1. This inverts
the usual instinct, so anyone who raises the batch size to keep the GPU busy will slow the
run down. The adapter docstring says so at the point where someone would change it.

Batch size is a correctness question here, not a performance setting, and the rule
generalizes past this one model. A Tower that packs sequences cannot be compared to one
that does not on throughput alone. Qwen3.8-27B belongs to the same native-resolution
family, so read its attention dispatch before sizing its burst.

Installing flash attention before the full extraction is worth an hour of setup. At batch
1 with sdpa an A100 runs this Tower at 15.5 images per second against 3.0 on the M4 Max.
DINOv2 gains 7.6 times on the same instance and MoonViT-V2 gains 5.2, and the gap is the
quadratic term. Flash attention removes it and restores large batches. Batch 1 works and
its cost is bounded, so this is an optimization to attempt before the burst, not a blocker
for v1.
