"""Layout and processor checks that need no checkpoint downloads."""

import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, Qwen2VLImageProcessor

from vtb import cache, extract
from vtb.adapters import mimo26
from vtb.models.mimo26 import MiMoVisionAttention, MiMoVisionConfig, MiMoVisionModel


@pytest.mark.parametrize("resolution", [32, 96, 448])
def test_preprocessing_matches_pinned_processor_contract(resolution):
    rng = np.random.default_rng(7)
    image = Image.fromarray(rng.integers(0, 256, (213, 317, 3), dtype=np.uint8))
    reference = Qwen2VLImageProcessor(
        patch_size=16, temporal_patch_size=2, merge_size=2,
        image_mean=mimo26.IMAGE_MEAN, image_std=mimo26.IMAGE_STD,
    )
    cropped = mimo26.square_crop(resolution)(image)
    expected = reference(images=cropped, do_resize=False, return_tensors="pt")
    actual = mimo26.Preprocess(resolution)(image)
    torch.testing.assert_close(actual, expected["pixel_values"], rtol=0, atol=5e-7)
    # Reconstruct each C,T,H,W patch and check the two temporal frames.
    patches = actual.reshape(-1, 3, 2, 16, 16)
    assert torch.equal(patches[:, :, 0], patches[:, :, 1])


@pytest.mark.parametrize("resolution", [0, -32, 224 + 16, 447])
def test_invalid_resolution_fails_early(resolution):
    with pytest.raises(ValueError, match="multiple of 32"):
        mimo26.Preprocess(resolution)


@pytest.mark.parametrize("column", [False, True])
def test_raster_restores_rectangular_grids_without_mixing_images(column):
    grid = torch.tensor([[1, 4, 6], [1, 6, 8]])
    expected, packed = [], []
    for image, (_, h, w) in enumerate(grid.tolist()):
        raster = torch.arange(h * w).reshape(h, w) + image * 1000
        expected.append(raster.flatten())
        # Construct the upstream layout by explicit coordinate enumeration,
        # independently of the adapter's reshape/transpose implementation.
        outer = [(y, x) for y in range(0, h, 2) for x in range(0, w, 2)]
        if column:
            outer = sorted(outer, key=lambda yx: (yx[1], yx[0]))
        packed.extend(raster[y + dy, x + dx] for y, x in outer for dy in range(2) for dx in range(2))
    restored = mimo26.raster(torch.stack(packed).reshape(-1, 1), grid, column)
    for actual, reference in zip(restored, expected, strict=True):
        assert torch.equal(actual.flatten(), reference)


class Passthrough(nn.Module):
    def forward(self, x, **kwargs):
        return x


def tiny_config(**kwargs):
    return MiMoVisionConfig(hidden_size=8, intermediate_size=16, num_heads=2,
                            num_key_value_heads=1, qk_channels=4, out_hidden_size=12, **kwargs)


def test_every_sampled_depth_tracks_its_current_layout(monkeypatch):
    model = MiMoVisionModel(tiny_config())
    model.visual.patch_embed = nn.Identity()
    # Parent dtype property refers to patch_embed.proj.weight.
    model.visual.patch_embed.proj = nn.Linear(8, 8, bias=False)
    model.visual.blocks = nn.ModuleList([Passthrough() for _ in range(28)])
    model.visual.merger.ln_q = nn.Identity()
    monkeypatch.setattr(mimo26, "load_model", lambda *a, **kw: model)
    adapter = mimo26.MiMo26ProAdapter(resolution=64, device="cpu", dtype=torch.float32)
    # Exercise all depths, including every row/column transition, then verify
    # the production sampler is a subset of those validated depths.
    points = adapter.depth_points()
    assert points == [4, 7, 10, 14, 18, 21, 24, 28]
    monkeypatch.setattr(adapter, "depth_points", lambda: list(range(1, 29)))
    raster = torch.arange(32).reshape(2, 4, 4, 1).expand(-1, -1, -1, 8).float()
    packed = raster.reshape(2, 2, 2, 2, 2, 8).permute(0, 1, 3, 2, 4, 5).reshape(-1, 8)
    batches = list(adapter.extract({"pixel_values": packed, "grid_thw": torch.tensor([[1, 4, 4]] * 2)}, ["a", "b"]))
    tower = [batch for batch in batches if batch.stage == "tower"]
    assert len(tower) == 28
    for batch in tower:
        assert torch.equal(batch.tokens, raster.reshape(2, 16, 8)), batch.layer_index
    assert torch.equal(batches[-2].tokens, packed.reshape(2, 4, 32))
    assert not any(block._forward_hooks for block in model.visual.blocks)


def test_sink_biases_key_zero_and_respects_window():
    attention = MiMoVisionAttention(4, num_heads=1, num_kv_heads=1, head_dim=4, use_sinks=True, window_size=1)
    with torch.no_grad():
        attention.qkv.weight.zero_(); attention.qkv.bias.zero_()
        attention.qkv.weight[8:] = torch.eye(4)
        attention.proj.weight.copy_(torch.eye(4)); attention.proj.bias.zero_()
        attention.sinks.fill_(2)
    x = torch.eye(4)
    actual = attention(x, torch.tensor([0, 4]), (torch.ones(4, 4), torch.zeros(4, 4)))
    logits = torch.zeros(4, 4)
    logits[:, 0] += 2
    for q in range(4):
        for k in range(4):
            if abs(q-k) > 1:
                logits[q, k] = -torch.inf
    torch.testing.assert_close(actual, logits.softmax(-1) @ x)
    # A separate sink would steal probability mass and produce a different row.
    separate = torch.cat([torch.zeros(1, 2), torch.full((1, 1), 2.0)], -1).softmax(-1)[0, :2]
    assert not torch.allclose(actual[0, :2], separate)


def test_bundle_roundtrip_is_offline_and_visual_only(tmp_path):
    model = MiMoVisionModel(tiny_config(depth=1, fullatt_block_indexes=[0], vit_window_attn_types=[-1])).eval()
    model.config.auto_map = {"AutoConfig": "modeling_mimo_vision.MiMoVisionConfig", "AutoModel": "modeling_mimo_vision.MiMoVisionModel"}
    model.save_pretrained(tmp_path)
    (tmp_path / "modeling_mimo_vision.py").write_text((mimo26.ROOT / "vtb/models/mimo26.py").read_text())
    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True, local_files_only=True)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    assert torch.equal(loaded.visual.rotary_pos_emb.inv_freq, model.visual.rotary_pos_emb.inv_freq)
    assert all(name.startswith("visual.") for name in loaded.state_dict())
    x = torch.randn(16, 1536)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(x, torch.tensor([[1, 4, 4]])), model(x, torch.tensor([[1, 4, 4]])), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("checkpoint,width", [(mimo26.PRO, 6144), (mimo26.FLASH, 4096)], ids=lambda c: getattr(c, "name", c))
def test_bundle_metadata_pins_real_variants(checkpoint, width):
    bundle = checkpoint.release_dir
    config = json.loads((bundle / "config.json").read_text())
    manifest = json.loads((bundle / "export_manifest.json").read_text())
    assert config["out_hidden_size"] == width
    assert config["vit_window_attn_types"] == MiMoVisionConfig().vit_window_attn_types
    assert (manifest["source_repo"], manifest["source_revision"]) == (checkpoint.repo, checkpoint.revision)
    assert manifest["tensors"] == len(manifest["tensor_sha256"]) == mimo26.VISUAL_TENSORS
    assert checkpoint.model_id in (bundle / "examples/inference.py").read_text()


@pytest.mark.parametrize("checkpoint", [mimo26.PRO, mimo26.FLASH], ids=lambda c: c.name)
def test_bundle_modeling_code_is_a_byte_copy_of_the_source(checkpoint):
    source = mimo26.ROOT / "vtb/models/mimo26.py"
    assert (checkpoint.release_dir / "modeling_mimo_vision.py").read_bytes() == source.read_bytes()


def test_generic_extraction_writes_every_stage_in_raster_order(monkeypatch, tmp_path):
    torch.manual_seed(0)
    model = MiMoVisionModel(tiny_config()).eval()
    monkeypatch.setattr(mimo26, "load_model", lambda *a, **kw: model)
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for name in ("a", "b", "c"):
        Image.fromarray(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)).save(images / f"{name}.png")
    monkeypatch.setattr(sys, "argv", [
        "extract", "--model", "mimo26_flash", "--images", str(images), "--out", str(tmp_path / "features"),
        "--batch-size", "2", "--workers", "0", "--device", "cpu", "--resolution", "64",
    ])
    extract.main()

    run = tmp_path / "features/mimo26_flash_64_full"
    summary = json.loads((run / "summary.json").read_text())
    assert summary["model_id"] == "immanuelpeter/MiMoViT-2.6-Flash"
    assert summary["images"] == 3 and summary["slices_per_image"] == 10
    assert sorted(cache.slices(run)) == sorted(
        [("tower", layer) for layer in (4, 7, 10, 14, 18, 21, 24, 28)] + [("merged", 28), ("projected", 28)]
    )
    widths = {"tower": 8, "merged": 32, "projected": 12}
    for stage, layer in cache.slices(run):
        tokens, ids, _ = cache.load(run, stage, layer)
        assert ids == ["a", "b", "c"]
        assert tuple(tokens.shape) == (3, 16 if stage == "tower" else 4, widths[stage])
    # Tower tokens must be raster order: the merged Stage regroups 2x2 raster
    # blocks of the normalized final Tower layer.
    tower, _, _ = cache.load(run, "tower", 28)
    merged, _, _ = cache.load(run, "merged", 28)
    blocks = tower.reshape(3, 2, 2, 2, 2, 8).permute(0, 1, 3, 2, 4, 5).reshape(3, 4, 4, 8)
    expected = model.visual.merger.ln_q(blocks.to(model.dtype)).reshape(3, 4, 32)
    torch.testing.assert_close(merged, expected.to(merged.dtype))
