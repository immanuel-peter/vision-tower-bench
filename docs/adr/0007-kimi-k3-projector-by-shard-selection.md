# The Kimi K3 Projector comes from two shards, not from surgery

PLAN.md listed MoonViT-V2's missing Projector as the top risk, on the assumption that
reaching it meant downloading and reassembling a checkpoint comparable to Kimi K2.6's
64 shards. Reading `model.safetensors.index.json` alone, with no weight download, shows
otherwise. Of 497,220 tensors across 96 shards, three carry the `mm_projector` prefix and
all three sit in `model-00095-of-000096.safetensors`, 0.09 GB. The 165 `vision_tower`
tensors sit entirely in `model-00096-of-000096.safetensors`, 0.80 GB. Both are ordinary
`hf_hub_download` calls, 0.89 GB in total, so the risk is retired rather than mitigated.

The Projector is Kimi K3's `patchmergerv2`: `Linear(4096, 4096)` without bias, GELU,
`Linear(4096, 7168)` without bias, then `RMSNorm(7168)`. Its input is the `merged` Stage
tensor the adapter already produces, so `projected` is one call on a tensor the pipeline
holds. The adapter rebuilds it from the checkpoint shapes rather than from a config, so a
changed upstream checkpoint fails at `load_state_dict` instead of mismatching quietly.

The parity gate in ADR-0003 is satisfied at a stronger bar than it asks for. All 165
standalone Tower tensors are bit-identical to the `vision_tower` tensors inside Kimi K3,
not merely equal within BF16 tolerance, so features extracted from the standalone Tower
describe the model this Projector was trained against.

The same shard-index-first check should run before every remaining shard-surgery estimate
on the roster, in particular Qwen3.8-27B, where PLAN.md currently assumes an 18-shard,
57 GB download.
