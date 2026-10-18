"""Raster, shuffle, normalization, and explicit Stage contracts without weights."""

import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vtb import cache, extract
from vtb.adapters import nemotron35 as family


def test_fixed_raster_matches_benchmark_crop_and_preserves_rgb():
    h, w = 480, 640
    y, x = np.mgrid[:h, :w]
    image = Image.fromarray(np.stack((x % 256, y % 256, (x + 3 * y) % 256), axis=-1).astype(np.uint8))
    pixels = family.Preprocess(448)(image)
    expected = image.resize((597, 448), Image.Resampling.BICUBIC).crop((74, 0, 522, 448))
    assert torch.equal(pixels, torch.from_numpy(np.array(expected)).permute(2, 0, 1).float() / 255)
    assert pixels.shape == (3, 448, 448)


def test_pixel_shuffle_keeps_raster_order_and_parent_channel_packing():
    # Parent v2 packs each 2x2 cell in row-major order.
    pixels = torch.arange(16).reshape(1, 4, 4, 1)
    actual = family.pixel_shuffle(pixels).flatten(1, 2)
    assert actual.tolist() == [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]]


def test_super_layernorm_runs_before_shuffle_and_bolt_omits_it():
    x = torch.randn(1, 4, 4, 8, generator=torch.Generator().manual_seed(42))
    norm = nn.LayerNorm(8, eps=1e-6)
    norm.weight.data.copy_(torch.linspace(0.5, 1.5, 8))
    merged = family.pixel_shuffle(norm(x))
    expected = torch.stack([norm(x[0, row, col]) for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))]).flatten()
    assert torch.equal(merged[0, 0, 0], expected)
    assert not torch.equal(merged, family.pixel_shuffle(x))
    assert family.SUPER.final_layernorm and not family.BOLT.final_layernorm


def test_projector_activation_is_squared_relu_and_output_width_is_4096():
    with torch.device("meta"):
        projector = family.make_projector()
    assert projector[0].weight.shape == (5120,)
    assert projector[1].weight.shape == (20480, 5120)
    assert projector[3].weight.shape == (4096, 20480)
    assert torch.equal(family.SquaredReLU()(torch.tensor([-2., -1., 0., 2.])), torch.tensor([0., 0., 0., 4.]))


def test_build_restores_original_fp32_conditioner_statistics(monkeypatch):
    from types import SimpleNamespace

    class Radio(nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.radio_model = nn.Module()
            self.radio_model.register_buffer("summary_idxs", torch.zeros(3, dtype=torch.int64))
            self.radio_model.input_conditioner = nn.Module()
            for name in ("norm_mean", "norm_std"):
                self.radio_model.input_conditioner.register_buffer(name, torch.zeros(3, 1, 1))

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
    monkeypatch.setattr(family.AutoConfig, "from_pretrained", lambda *_a, **_kw: SimpleNamespace(vision_config=None))
    monkeypatch.setattr(family, "get_class_from_dynamic_module", lambda *_a, **_kw: Radio)
    monkeypatch.setattr(family, "attach_video_embedder", lambda _: None)
    monkeypatch.setattr(family, "make_projector", nn.Identity)
    parts = {"tower": {"summary_idxs": torch.tensor([0, 1]),
                       "radio_model.input_conditioner.norm_mean": mean,
                       "radio_model.input_conditioner.norm_std": std},
             "projector": {}, "final_norm": {}}
    radio, _, _ = family.build_parts(family.BOLT, parts, torch.bfloat16)
    conditioner = radio.radio_model.input_conditioner
    assert conditioner.norm_mean.dtype == torch.float32
    assert torch.equal(conditioner.norm_mean, mean)
    assert torch.equal(conditioner.norm_std, std)
    assert not torch.equal(mean, mean.bfloat16().float())


class FakeRadio(nn.Module):
    def __init__(self):
        super().__init__()
        from types import SimpleNamespace
        blocks = nn.ModuleList([nn.Identity() for _ in range(32)])
        self.radio_model = SimpleNamespace(model=SimpleNamespace(blocks=blocks, patch_generator=SimpleNamespace(num_skip=10)))

    def forward(self, pixels):
        from types import SimpleNamespace
        assert pixels.dtype == torch.float32
        x = torch.arange(14 * 1280, dtype=torch.float32).reshape(1, 14, 1280).expand(len(pixels), -1, -1)
        for block in self.radio_model.model.blocks:
            x = block(x)
        return SimpleNamespace(features=x[:, 10:])


@pytest.fixture
def fake_parts(monkeypatch):
    monkeypatch.setattr(family, "load_parts", lambda *_: (FakeRadio(), nn.Identity(), nn.Identity()))


def test_bolt_default_stage_filter_and_explicit_tower_sanity(fake_parts):
    adapter = family.Nemotron35BoltAdapter(resolution=32, device="cpu", dtype=torch.float32)
    inputs = {"pixel_values": torch.zeros(1, 3, 32, 32)}
    batches = list(adapter.extract(inputs, ["sample"]))
    assert [batch.stage for batch in batches] == ["merged", "projected"]
    assert batches[0].tokens.shape == (1, 1, 5120)
    adapter.selected_stages = ("tower",)
    batches = list(adapter.extract(inputs, ["sample"]))
    assert [batch.layer_index for batch in batches] == list(range(4, 33, 4))
    assert all(batch.tokens.shape == (1, 4, 1280) for batch in batches)
    assert all(not block._forward_hooks for block in adapter.model.radio_model.model.blocks)


def test_extract_writes_only_the_final_bolt_tower_when_requested(fake_parts, monkeypatch, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "sample.png")
    monkeypatch.setattr(sys, "argv", [
        "extract", "--model", "nemotron35_bolt", "--images", str(images), "--out", str(tmp_path / "features"),
        "--workers", "0", "--device", "cpu", "--resolution", "32",
        "--stages", "tower", "--final-stages", "tower",
    ])
    extract.main()
    run = tmp_path / "features/nemotron35_bolt_32_full"
    assert cache.slices(run) == [("tower", 32)]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["model_id"] == "immanuelpeter/C-RADIOv4-H-3.5-Bolt"
    assert summary["stages"] == ["tower"]
    assert summary["tokens_per_image"] == 4


@pytest.mark.parametrize("resolution", [0, 16, 33, 447])
def test_shuffle_rejects_invalid_resolution_without_loading_weights(resolution):
    with pytest.raises(ValueError, match="multiple of 32"):
        family.Nemotron35SuperAdapter(resolution=resolution)
