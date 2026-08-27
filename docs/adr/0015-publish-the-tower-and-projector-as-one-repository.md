# Publish the Tower and Projector together

Publish the MoonViT-V2 Tower and Kimi K3 Projector in one Hugging Face repository. The
adapter currently loads the Tower from `AI4Industry/MoonViT-V2` and the three Projector
tensors from shard 95 of `moonshotai/Kimi-K3`. Co-locating them removes that shard-specific
download path.

Keep the weights in separate files. `model.safetensors` contains the 165 unprefixed Tower
tensors, preserving `AutoModel.from_pretrained` compatibility with
`AI4Industry/MoonViT-V2`. `projector.safetensors` contains the three Projector tensors.
`projector_config.json` records their shapes because `MoonViTV2Config` does not describe the
Projector.

Vendor AI4Industry's modeling and image-processing files unchanged and credit them in the
model card. Ship the upstream Kimi K3 `LICENSE`, whose grant permits redistribution. Section
1 requires the copyright and permission notice. Sections 2 and 3 apply only above $20
million in Model-as-a-Service revenue or 100 million monthly active users.

`scripts/export_moonvit_v2.py` verifies all 165 Tower tensors against the standalone
checkpoint before writing. Upload remains a manual `hf upload`. After publication, the
adapter uses the combined repository and drops its Projector shard constants. This replaces
ADR-0007's download path, not its parity result.
