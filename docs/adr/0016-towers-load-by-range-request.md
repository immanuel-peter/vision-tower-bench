# Towers load by range request

A Tower is a small part of the checkpoint that ships it. Qwen's is 0.92 GB inside a
3.97 GB shard, and Muse Glimmer's is 3.84 GB spread across two shards totalling 59.55 GB.
Downloading whole shards to reach them moves 63.52 GB to use 4.76 GB.

Every safetensors file starts with a header that gives each tensor a dtype, a shape, and a
byte range. `vtb/shards.py` reads that header with one range request and then reads only
the tensors whose names match a prefix. ADR-0007 used the same idea by hand to find the
Kimi K3 Projector; this makes it the way every adapter loads.

One request per tensor does not work. Qwen's Tower is 333 tensors and Muse's is 806, and
issuing that many requests earns a 429 from the CDN partway through. Tensors sharing a
prefix sit next to each other in the file, so the reader groups them into spans and reads
one span per request, tolerating gaps up to 8 MB. Qwen's 333 tensors become one request,
0.92 GB in 18.7 seconds. Requests retry with exponential backoff on 429 and on 5xx.

The cost is that the parent repository has to keep serving range requests, which
`hf_hub_download` would have cached locally. That trade is worth it while the roster is
still moving; a Tower that gets probed repeatedly should be republished instead, the way
ADR-0015 handles MoonViT-V2.
