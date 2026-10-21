"""DINOv3's token contract and independent preprocessing/order checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTEmbeddings

from vtb.adapters import dinov3
from vtb.adapters.dinov2 import Preprocess as DINOv2Preprocess


@pytest.mark.parametrize("resolution,side", [(448, 28), (512, 32)])
def test_preprocess_uses_bench_crop_and_parent_channel_norm(resolution, side):
    y, x = np.mgrid[:63, :97]
    picture = Image.fromarray(np.stack([x * 2, y * 3, (x + y) % 256], -1).astype("uint8"))
    # Independent PIL construction of the bench's short-side resize/crop.
    width = int(97 * resolution / 63)
    resized = picture.resize((width, resolution), Image.Resampling.BICUBIC)
    left = round((width - resolution) / 2)
    cropped = resized.crop((left, 0, left + resolution, resolution))
    raw = torch.from_numpy(np.array(cropped)).permute(2, 0, 1).float() / 255
    expected = (raw - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    actual = dinov3.Preprocess(resolution)(picture)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual, DINOv2Preprocess(resolution)(picture))
    assert actual.shape == (3, side * 16, side * 16)


@pytest.mark.parametrize("resolution", [0, -16, 449, 511])
def test_rejects_partial_patch_crops(resolution):
    with pytest.raises(ValueError, match="multiple of 16"):
        dinov3.Preprocess(resolution)


def test_official_patch_embedding_is_row_major_after_five_prefix_tokens():
    config = DINOv3ViTConfig(hidden_size=16, num_attention_heads=2, patch_size=16, num_register_tokens=4)
    embeddings = DINOv3ViTEmbeddings(config)
    with torch.no_grad():
        embeddings.patch_embeddings.weight.zero_()
        embeddings.patch_embeddings.weight[:, 0, 0, 0] = 1
        embeddings.patch_embeddings.bias.zero_()
    pixels = torch.zeros(1, 3, 32, 48)
    pixels[0, 0, ::16, ::16] = torch.tensor([[11., 12., 13.], [21., 22., 23.]])
    result = embeddings(pixels)
    assert result.shape == (1, 11, 16)
    torch.testing.assert_close(result[0, 5:, 0], torch.tensor([11., 12., 13., 21., 22., 23.]))


class SentinelParent:
    config = SimpleNamespace(num_hidden_layers=24, hidden_size=1024, patch_size=16, num_register_tokens=4)
    dtype = torch.float32

    def to(self, _):
        return self

    def eval(self):
        return self

    def __call__(self, pixels, output_hidden_states):
        assert output_hidden_states
        batch = len(pixels)
        # Prefix values cannot be mistaken for spatial patches; each depth is distinct.
        row = torch.arange(9).float()[None, :, None].expand(batch, 9, 1024)
        self.hidden_states = tuple(row + depth * 100 for depth in range(25))
        self.final = row - 999
        return SimpleNamespace(hidden_states=self.hidden_states, last_hidden_state=self.final)


def test_adapter_pins_source_drops_registers_and_retains_parent_final_norm(monkeypatch):
    parent = SentinelParent()
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return parent

    monkeypatch.setattr(dinov3.DINOv3ViTModel, "from_pretrained", load)
    monkeypatch.delenv(dinov3.BUNDLE_ENV, raising=False)
    adapter = dinov3.DINOv3Adapter(resolution=32, dtype=torch.float32, device="cpu")
    assert calls == [((dinov3.SOURCE_REPO,), {"revision": dinov3.SOURCE_REVISION, "dtype": torch.float32})]
    batches = list(adapter.extract(torch.zeros(2, 3, 32, 32), ["first", "second"]))
    assert [b.layer_index for b in batches] == [3, 6, 9, 12, 15, 18, 21, 24]
    for batch in batches:
        expected = parent.final if batch.layer_index == 24 else parent.hidden_states[batch.layer_index]
        assert torch.equal(batch.tokens, expected[:, 5:])
        assert batch.tokens.shape == (2, 4, 1024)
        assert batch.image_ids == ["first", "second"]
        assert batch.model_id == dinov3.SOURCE_REPO
        assert batch.relative_depth == batch.layer_index / 24
    with pytest.raises(ValueError, match="expected pixels"):
        list(adapter.extract(torch.zeros(1, 3, 48, 32), ["wrong"]))
    with pytest.raises(ValueError, match="same length"):
        list(adapter.extract(torch.zeros(2, 3, 32, 32), ["wrong"]))


def test_parent_preprocessor_normalization_on_the_bench_crop():
    from transformers import DINOv3ViTImageProcessor

    picture = Image.fromarray(np.random.default_rng(3).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    parent = DINOv3ViTImageProcessor(do_resize=False)
    expected = parent(images=picture, return_tensors="pt")["pixel_values"][0]
    actual = dinov3.Preprocess(32)(picture)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
