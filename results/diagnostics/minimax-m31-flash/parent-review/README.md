# Parent delivery audit

The parent checked all 103 canonical payload hashes, 49 unique paired payloads,
30 figure exports and 582 figure-source snapshot files. CPU validators replayed
the recorded bootstrap intervals and checked actual IDs, labels, grids and seeds.

The source audit downloaded only the four pinned vision-containing parent shards,
verified their complete LFS SHA256s, then read selected vision tensors directly
with safe_open. All 523 tensors per checkpoint and all drift rows match.

The first source attempt completed all four shard hashes, then stopped because
the baseline bundle was in scratch rather than hf/. Its failure is retained in
parent-source-audit.first-run.txt. The resumed script uses that recorded baseline
path and the completed shard checkpoint, without downloading again.
The seconds field covers the resumed phase, not the preceding downloads.
Original scratch .log files are retained here as .txt.

Commands and paths are encoded in the audit script and logs. CUDA was hidden;
PyTorch and BLAS used two threads. No experiments were retrained.
