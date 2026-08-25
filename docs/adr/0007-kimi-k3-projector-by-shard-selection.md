# The Kimi K3 Projector needs one small shard

PLAN.md listed MoonViT-V2's missing Projector as the top risk, on the assumption that
reaching it meant downloading and reassembling a checkpoint comparable to Kimi K2.6's
64 shards. Reading `model.safetensors.index.json` alone, with no weight download, shows
otherwise. Of 497,220 tensors across 96 shards, three carry the `mm_projector` prefix and
all three sit in `model-00095-of-000096.safetensors`, 0.09 GB. The 165 `vision_tower`
tensors sit entirely in `model-00096-of-000096.safetensors`, 0.80 GB. Both are ordinary
`hf_hub_download` calls totaling 0.89 GB. A full checkpoint reconstruction is unnecessary.

The Projector is Kimi K3's `patchmergerv2`: `Linear(4096, 4096)` without bias, GELU,
`Linear(4096, 7168)` without bias, then `RMSNorm(7168)`. Its input is the `merged` Stage
tensor the adapter already produces. The adapter builds the module from checkpoint shapes
rather than a separate config. If the upstream shapes change, `load_state_dict` fails.

All 165 standalone Tower tensors are bit-identical to the `vision_tower` tensors inside
Kimi K3. This exceeds ADR-0003's BF16 tolerance requirement and confirms that the
standalone Tower matches the one used to train the Projector.

Check shard indices before estimating any remaining extraction. PLAN.md currently assumes
that Qwen3.8-27B requires an 18-shard, 57 GB download, which this method may avoid.
