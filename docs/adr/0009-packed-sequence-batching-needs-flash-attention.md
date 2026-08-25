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
one that does not. Qwen3.8-27B belongs to the same native-resolution family, so inspect
its attention dispatch before choosing a batch size.

Try installing flash attention before the full extraction. At batch 1 with sdpa, an A100
runs MoonViT-V2 at 15.5 images per second, compared with 3.0 on the M4 Max. DINOv2 is
7.6 times faster on the same A100, while MoonViT-V2 is 5.2 times faster. Flash attention
should remove the packed-mask cost and permit larger batches, but batch 1 is fast enough
to keep v1 moving if setup fails.
