import json
import os

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Mistral3Config, PixtralImageProcessor, PixtralVisionConfig, PixtralVisionModel
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector
from transformers.models.pixtral.modeling_pixtral import apply_rotary_pos_emb

from scripts import validate_mistral_large4 as validate
from vtb.adapters import mistral_large4 as ml4

needs_weights = pytest.mark.skipif(os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1")


def native_names(layers: int = 30) -> list[str]:
    names = ["vision_encoder.patch_conv.weight", "vision_encoder.ln_pre.weight"]
    for n in range(layers):
        stem = f"vision_encoder.transformer.layers.{n}."
        names += [stem + f"attention.{w}.weight" for w in ("wq", "wk", "wv", "wo")]
        names += [stem + f"feed_forward.{w}.weight" for w in ("w1", "w2", "w3")]
        names += [stem + "attention_norm.weight", stem + "ffn_norm.weight"]
    return names + [
        "pre_mm_projector_norm.weight",
        "patch_merger.merging_layer.weight",
        "vision_language_adapter.w_in.weight",
        "vision_language_adapter.w_out.weight",
    ]


def test_remap_is_one_to_one_and_names_hf_modules():
    names = native_names()
    renamed = [ml4.remap_key(name) for name in names]
    assert len(names) == 276 and len(set(renamed)) == 276
    assert sum(name.startswith("vision_tower.") for name in renamed) == ml4.TOWER_TENSORS
    assert ml4.remap_key("vision_encoder.transformer.layers.7.feed_forward.w1.weight").endswith(
        "layers.7.feed_forward.gate_proj.weight"
    )
    assert ml4.remap_key("vision_encoder.transformer.layers.7.feed_forward.w2.weight").endswith("down_proj.weight")
    assert ml4.remap_key("vision_encoder.transformer.layers.7.feed_forward.w3.weight").endswith("up_proj.weight")
    assert ml4.remap_key("vision_language_adapter.w_out.weight") == "multi_modal_projector.linear_2.weight"
    assert sum(ml4.is_rotary(name) for name in renamed) == 60
    with pytest.raises(KeyError):
        ml4.remap_key("vision_encoder.transformer.norm.weight")


def tiny_tower_config(**overrides) -> PixtralVisionConfig:
    settings = dict(
        hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=2, head_dim=16,
        image_size=112, patch_size=14, hidden_act="silu", rope_parameters={"rope_theta": 10000.0, "rope_type": "axial"},
    )
    settings.update(overrides)
    return PixtralVisionConfig(**settings)


def mistral_rotary(x: torch.Tensor, positions: torch.Tensor, head_dim: int, side: int) -> torch.Tensor:
    """Mistral's complex rotation on interleaved pairs, written from mistral_inference.rope."""
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    h = torch.outer(torch.arange(side), freqs[::2])
    w = torch.outer(torch.arange(side), freqs[1::2])
    grid = torch.cat([h[:, None, :].repeat(1, side, 1), w[None, :, :].repeat(side, 1, 1)], dim=-1)
    cis = torch.polar(torch.ones_like(grid), grid)[positions[:, 0], positions[:, 1]]
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * cis[:, None, :]).flatten(-2)


def test_rope_permutation_turns_mistral_rotation_into_hf_rotate_half():
    torch.manual_seed(0)
    config = tiny_tower_config()
    heads, head_dim, side = 2, 16, 4
    rotary = PixtralVisionModel(config).patch_positional_embedding
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    positions = torch.stack((ys.flatten(), xs.flatten()), dim=-1)
    tokens = torch.randn(side * side, 32)
    weight = torch.randn(heads * head_dim, 32)

    native = (tokens @ weight.T).view(-1, heads, head_dim)
    expected = mistral_rotary(native, positions, head_dim, side)

    cos, sin = rotary(tokens[None], positions)
    permuted = (tokens @ ml4.interleaved_to_halves(weight, heads).T).view(1, -1, heads, head_dim).transpose(1, 2)
    rotated, _ = apply_rotary_pos_emb(permuted, permuted, cos, sin, unsqueeze_dim=0)
    # Bring HF's halves back to interleaved order to compare channel by channel.
    actual = rotated[0].transpose(0, 1).reshape(-1, heads, 2, head_dim // 2).transpose(2, 3).reshape(-1, heads, head_dim)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    plain = (tokens @ weight.T).view(1, -1, heads, head_dim).transpose(1, 2)
    wrong, _ = apply_rotary_pos_emb(plain, plain, cos, sin, unsqueeze_dim=0)
    wrong = wrong[0].transpose(0, 1)
    assert (wrong - expected).norm() / expected.norm() > 0.1


def test_permutation_round_trips_exactly():
    weight = torch.randn(2048, 64, dtype=torch.bfloat16)
    assert torch.equal(ml4.halves_to_interleaved(ml4.interleaved_to_halves(weight)), weight)
    assert not torch.equal(ml4.interleaved_to_halves(weight), weight)


def tiny_projector_config(tower: PixtralVisionConfig) -> Mistral3Config:
    return Mistral3Config(
        vision_config=tower.to_dict(),
        text_config={"model_type": "mistral", "hidden_size": 40, "rms_norm_eps": 1e-5},
        spatial_merge_size=2, projector_hidden_act="gelu", multimodal_projector_bias=False, vision_feature_layer=-1,
    )


def test_merger_input_is_channel_major_over_the_2x2_block_as_in_mistral():
    torch.manual_seed(0)
    config = tiny_tower_config()
    projector = Mistral3MultiModalProjector(tiny_projector_config(config))
    side, width = 4, 32
    tokens = torch.randn(side * side, width)
    captured = {}
    projector.patch_merger.merging_layer.register_forward_pre_hook(lambda _m, args: captured.setdefault("x", args[0]))
    projector.patch_merger(tokens, [(side * 14, side * 14)])
    grid = tokens.view(side, side, width)
    expected = []
    for y in range(0, side, 2):
        for x in range(0, side, 2):
            # mistral_inference get_sub_grids: d x (row, col) of the block, flattened channel-major.
            block = torch.stack([grid[y, x], grid[y, x + 1], grid[y + 1, x], grid[y + 1, x + 1]], dim=-1)
            expected.append(block.reshape(-1))
    assert torch.equal(captured["x"], torch.stack(expected))


def tiny_parts():
    torch.manual_seed(0)
    config = tiny_tower_config()
    tower = PixtralVisionModel(config).eval()
    projector = Mistral3MultiModalProjector(tiny_projector_config(config)).eval()
    return tower, projector


def test_batched_tower_matches_the_packed_block_diagonal_forward():
    tower, projector = tiny_parts()
    pixels = torch.randn(3, 3, 112, 112)
    with torch.inference_mode():
        final, captured = ml4.tower_forward(tower, pixels, (2, 4))
        packed = tower(pixels, image_sizes=[(112, 112)] * 3).last_hidden_state.view(3, 64, 32)
        merged, projected = ml4.projector_stages(projector, final, 8)
        reference = projector(packed.reshape(-1, 32), [(112, 112)] * 3).view(3, 16, 40)
    torch.testing.assert_close(final, packed, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(captured[4], final)
    torch.testing.assert_close(projected, reference, rtol=1e-5, atol=1e-5)
    assert merged.shape == (3, 16, 32)


def test_merged_and_projected_follow_the_raster_merged_grid():
    tower, projector = tiny_parts()
    side = 8
    tokens = torch.zeros(1, side * side, 32)
    # Mark one Tower token; it must land in merged token (row // 2) * 4 + col // 2.
    for row, col in ((0, 0), (0, 5), (3, 2), (7, 7)):
        tokens.zero_()
        tokens[0, row * side + col] = 1.0
        merged, _ = ml4.projector_stages(projector, tokens, side)
        hit = merged[0].abs().sum(-1).nonzero().flatten().tolist()
        assert hit == [(row // 2) * (side // 2) + col // 2]


def test_preprocess_matches_the_parent_processor_and_rejects_other_grids():
    with pytest.raises(ValueError):
        ml4.Preprocess(440)
    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 256, (448, 448, 3), dtype=np.uint8))
    ours = ml4.Preprocess(448)(image)
    parent = PixtralImageProcessor(
        size={"longest_edge": 448}, patch_size=14, image_mean=list(ml4.IMAGE_MEAN), image_std=list(ml4.IMAGE_STD),
        resample=3, do_pad=True,
    )(images=image, return_tensors="pt")
    assert parent["pixel_values"].shape == (1, 3, 448, 448)
    torch.testing.assert_close(ours, parent["pixel_values"][0], rtol=0, atol=1e-5)
    assert ml4.collate([ours, ours])["pixel_values"].shape == (2, 3, 448, 448)


def test_depth_points_match_the_issue():
    adapter = object.__new__(ml4.MistralLarge4Adapter)
    adapter.num_layers = 30
    assert adapter.depth_points() == [4, 8, 11, 15, 19, 22, 26, 30]


def test_attention_substitute_matches_softmax_attention_with_block_masks():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 9, 3, 8) for _ in range(3))
    mask = validate.BlockDiagonalMask.from_seqlens([4, 5])
    actual = validate.memory_efficient_attention(q, k, v, mask)
    for start, stop in ((0, 4), (4, 9)):
        qs, ks, vs = (t[:, start:stop].transpose(1, 2) for t in (q, k, v))
        weights = torch.softmax(qs @ ks.transpose(-1, -2) / 8**0.5, dim=-1)
        torch.testing.assert_close(actual[:, start:stop], (weights @ vs).transpose(1, 2), rtol=1e-5, atol=1e-6)
    unmasked = validate.memory_efficient_attention(q, k, v)
    expected = torch.softmax(q.transpose(1, 2) @ k.permute(0, 2, 3, 1) / 8**0.5, -1) @ v.transpose(1, 2)
    torch.testing.assert_close(unmasked, expected.transpose(1, 2), rtol=1e-5, atol=1e-6)
    half = validate.memory_efficient_attention(q.bfloat16(), k.bfloat16(), v.bfloat16())
    assert half.dtype == torch.bfloat16


@needs_weights
def test_remap_covers_every_vision_tensor_in_the_pinned_index_once():
    index = ml4.source_index()
    assert len(index) == 276
    assert set(index.values()) == {"consolidated-00053-of-00054.safetensors"}
    assert sorted(index) == sorted(native_names())


@needs_weights
def test_source_shapes_and_dtypes():
    from vtb.shards import read_header

    header, _, _ = read_header(ml4.SOURCE_REPO, "consolidated-00053-of-00054.safetensors", ml4.SOURCE_REVISION)
    vision = {name: entry for name, entry in header.items() if name.startswith(ml4.SOURCE_PREFIXES)}
    assert len(vision) == 276 and {entry["dtype"] for entry in vision.values()} == {"BF16"}
    shapes = {name: tuple(entry["shape"]) for name, entry in vision.items()}
    assert shapes["vision_encoder.patch_conv.weight"] == (2048, 3, 14, 14)
    assert shapes["patch_merger.merging_layer.weight"] == (2048, 8192)
    assert shapes["vision_language_adapter.w_in.weight"] == (6144, 2048)
    assert shapes["vision_language_adapter.w_out.weight"] == (6144, 6144)
    tower = sum(int(np.prod(s)) for n, s in shapes.items() if n.startswith("vision_encoder."))
    assert round(tower / 1e9, 2) == 1.64


@needs_weights
def test_bundle_manifest_and_configs_record_the_pinned_parent():
    bundle = ml4.RELEASE_DIR
    if not (bundle / "export_manifest.json").is_file():
        pytest.skip("run scripts/export_mistral_large4_vision.py first")
    manifest = json.loads((bundle / "export_manifest.json").read_text())
    assert (manifest["source_repo"], manifest["source_revision"]) == (ml4.SOURCE_REPO, ml4.SOURCE_REVISION)
    assert manifest["tensors"] == 276 and len(manifest["rope_permutation"]["tensors"]) == 60
    projector = json.loads((bundle / "projector_config.json").read_text())
    assert projector["text_config"]["rms_norm_eps"] == 1e-5
    assert projector["projector_hidden_act"] == "gelu"
    assert json.loads((bundle / "config.json").read_text())["hidden_act"] == "silu"


def test_bf16_rotation_follows_mistral_float32_order():
    torch.manual_seed(3)
    heads, head_dim, side = 2, 16, 6
    config = tiny_tower_config()
    rotary = PixtralVisionModel(config).patch_positional_embedding
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    positions = torch.stack((ys.flatten(), xs.flatten()), dim=-1)
    native = (torch.randn(side * side, heads, head_dim) * 40).bfloat16()
    expected = mistral_rotary(native, positions, head_dim, side).bfloat16()

    halves = native.view(-1, heads, head_dim // 2, 2).transpose(2, 3).reshape(-1, heads, head_dim)
    table = ml4.rotary_table(side, side, head_dim, 10000.0)
    assert table.dtype == torch.complex64
    actual = ml4.rotate(halves.transpose(0, 1)[None], table)[0].transpose(0, 1)
    assert actual.dtype == torch.bfloat16
    # Same table, same complex multiplication, same cast: identical BF16 values.
    assert torch.equal(actual, expected)
    cos, sin = rotary(torch.zeros(1, side * side, 32), positions)

    hf_cos, hf_sin = cos.bfloat16(), sin.bfloat16()
    q = halves.transpose(0, 1)[None]
    hf, _ = apply_rotary_pos_emb(q, q, hf_cos, hf_sin, unsqueeze_dim=0)
    hf = hf[0].transpose(0, 1).view(-1, heads, 2, head_dim // 2).transpose(2, 3).reshape(-1, heads, head_dim)
    assert (hf != expected).float().mean() > 0.05


def test_block_forward_matches_the_hf_layer_in_float32():
    tower, _ = tiny_parts()
    hidden = torch.randn(2, 64, 32)
    ys, xs = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    positions = torch.stack((ys.flatten(), xs.flatten()), dim=-1)
    cos, sin = tower.patch_positional_embedding(hidden, positions)
    layer = tower.transformer.layers[0]
    with torch.inference_mode():
        expected = layer(hidden, attention_mask=None, position_embeddings=(cos, sin))
        actual = ml4.block_forward(layer, hidden, ml4.rotary_table(8, 8, 16, 10000.0))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_release_ships_the_shared_modeling_file_and_maps_auto_classes():
    bundle = ml4.RELEASE_DIR
    shared = ml4.ROOT / "vtb/models/pixtral_large4.py"
    assert (bundle / "modeling_pixtral_large4.py").read_bytes() == shared.read_bytes()
    config = json.loads((bundle / "config.json").read_text())
    assert config["model_type"] == "pixtral_large4_vision"
    assert config["architectures"] == ["PixtralLarge4VisionModel"]
    assert config["auto_map"] == {
        "AutoConfig": "modeling_pixtral_large4.PixtralLarge4VisionConfig",
        "AutoModel": "modeling_pixtral_large4.PixtralLarge4VisionModel",
    }


def tiny_corrected_bundle(tmp_path):
    from transformers import AutoModel

    torch.manual_seed(0)
    config = ml4.PixtralLarge4VisionConfig(**{k: v for k, v in tiny_tower_config().to_dict().items() if k != "model_type"})
    config.auto_map = {
        "AutoConfig": "modeling_pixtral_large4.PixtralLarge4VisionConfig",
        "AutoModel": "modeling_pixtral_large4.PixtralLarge4VisionModel",
    }
    model = ml4.PixtralLarge4VisionModel(config).eval()
    model.save_pretrained(tmp_path)
    (tmp_path / "modeling_pixtral_large4.py").write_bytes((ml4.ROOT / "vtb/models/pixtral_large4.py").read_bytes())
    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True, local_files_only=True).eval()
    return model, loaded


def test_remote_code_bundle_runs_the_shared_forward(tmp_path):
    model, loaded = tiny_corrected_bundle(tmp_path)
    assert type(loaded).__name__ == "PixtralLarge4VisionModel"
    assert type(loaded).__module__ != ml4.PixtralLarge4VisionModel.__module__
    pixels = torch.randn(2, 3, 112, 112)
    for dtype in (torch.float32, torch.bfloat16):
        loaded_d, model_d = loaded.to(dtype), model.to(dtype)
        with torch.inference_mode():
            released = loaded_d(pixels.to(dtype), output_hidden_states=True)
            shared, captured = ml4.encode(model_d, pixels.to(dtype), (2, 4))
        assert torch.equal(released.last_hidden_state[0], shared.reshape(-1, 32))
        assert torch.equal(released.hidden_states[1][0], captured[2].reshape(-1, 32))
    with torch.inference_mode():
        stock = PixtralVisionModel.forward(model.float(), pixels).last_hidden_state
        corrected = model.float()(pixels).last_hidden_state
    torch.testing.assert_close(corrected, stock, rtol=1e-4, atol=1e-4)


def test_corrected_forward_packs_mixed_sizes_like_pixtral():
    tower, _ = tiny_parts()
    corrected = ml4.PixtralLarge4VisionModel(ml4.PixtralLarge4VisionConfig(
        **{k: v for k, v in tower.config.to_dict().items() if k != "model_type"})).eval()
    corrected.load_state_dict(tower.state_dict())
    pixels = torch.randn(2, 3, 112, 112)
    sizes = [(112, 84), (56, 112)]
    with torch.inference_mode():
        packed = corrected(pixels, image_sizes=sizes).last_hidden_state[0]
        first, _ = ml4.encode(corrected, pixels[:1, :, :112, :84])
        second, _ = ml4.encode(corrected, pixels[1:, :, :56, :112])
        stock = PixtralVisionModel.forward(tower, pixels, image_sizes=sizes).last_hidden_state[0]
    assert packed.shape == (8 * 6 + 4 * 8, 32)
    assert torch.equal(packed, torch.cat([first[0], second[0]]))
    torch.testing.assert_close(packed, stock, rtol=1e-4, atol=1e-4)


def test_validator_release_path_equals_the_adapter_path(tmp_path):
    _, loaded = tiny_corrected_bundle(tmp_path)
    _, projector = tiny_parts()
    pixels = torch.randn(2, 3, 112, 112).bfloat16()
    loaded, projector = loaded.bfloat16(), projector.bfloat16()
    with torch.inference_mode():
        released = validate.release_stages(loaded, projector, pixels, (2, 4))
        single = validate.release_stages(loaded, projector, pixels[:1], (2, 4))
        shared = validate.candidate_stages(loaded, projector, pixels[:1], (2, 4))
    assert set(released) == {"tower_L2", "tower_L4", "merged", "projected"}
    for name, tensor in single.items():
        assert torch.equal(tensor, shared[name])
        assert torch.equal(released[name][0], tensor)


def test_bundle_card_contents_and_manifest_hashes_match_the_files():
    import hashlib
    import re

    bundle = ml4.RELEASE_DIR
    manifest = json.loads((bundle / "export_manifest.json").read_text())
    assert manifest["hash_excludes"] == ["export_manifest.json", ".gitattributes"]
    assert manifest["source_shards"] == ["consolidated-00053-of-00054.safetensors"]
    assert (manifest["tower_tensors"], manifest["projector_tensors"], manifest["dtype"]) == (272, 4, "bfloat16")
    present = {str(p.relative_to(bundle)) for p in bundle.rglob("*")
               if p.is_file() and p.name != "export_manifest.json" and "__pycache__" not in p.parts}
    weights = {"model.safetensors", "projector.safetensors"}
    hashed = set(manifest["files_sha256"])
    # Weight files are not in git; check them only where an export wrote them.
    assert present - weights <= hashed and hashed - present <= weights
    for name in present & hashed - weights:
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == manifest["files_sha256"][name], name

    card = (bundle / "README.md").read_text()
    contents = card.split("## Contents", 1)[1].split("\n## ", 1)[0]
    rows = [line.split("|")[1] for line in contents.splitlines() if line.startswith("| `")]
    listed = {name for cell in rows for name in re.findall(r"`([^`]+)`", cell)}
    # The card is README.md itself.
    assert listed | {"README.md"} == hashed | {"export_manifest.json"}
    assert len(card.splitlines()) <= 70 and card.count("\n## ") == 7
