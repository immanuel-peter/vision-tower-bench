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
0.92 GB in 18.7 seconds.

Spans are then capped at 256 MB, which undoes part of that grouping on purpose. The CDN
resets long transfers, and a reset costs the whole request, so one 3.7 GB span for Muse
threw away gigabytes on a broken socket. At 256 MB Muse takes about 15 requests and Qwen
about 4, both far below the rate limit that 333 hit, and a reset costs at most 256 MB.
Requests retry with exponential backoff on 429, on 5xx, and on transport errors. The
transport case was missing at first, so a reset socket raised `ChunkedEncodingError` and
killed the read outright.

## Cache range reads

`hf_hub_download` caches; raw range requests do not. Every construction of an adapter
re-read the whole Tower. Muse's 3.71 GB took 758 seconds each time, which is most of why
the parity suite ran for 31 minutes and the merge tests for 23.

`load_prefixed` now writes its result to `$HF_HOME/vtb-shards`, so a scratch disk holds it
beside the Hugging Face cache, and `VTB_SHARD_CACHE` moves it. The same Muse read returns
in 0.33 seconds, 2306 times faster, with tensors verified identical.

The cache key is a hash of the repository, the prefix, and the etag of every shard read.
A new upstream revision changes the etags and misses the cache, preventing stale weights
from being served silently. Entries are written to a temporary file and renamed, so a
killed process cannot leave a half-written entry that a later run would trust.

The remaining cost is that a first read still depends on the parent repository serving
range requests. A Tower that gets probed repeatedly should be republished instead, the way
ADR-0015 handles MoonViT-V2.
