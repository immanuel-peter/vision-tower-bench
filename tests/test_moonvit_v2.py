import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torchvision import transforms
from transformers import AutoImageProcessor

from vtb.adapters import moonvit_v2
from vtb.adapters.moonvit_v2 import CHECKPOINTS, K3, K31_PRO, KimiK31ProAdapter, MoonViTV2Adapter


def test_checkpoint_selection_preserves_k3():
    assert MoonViTV2Adapter.checkpoint == K3
    assert KimiK31ProAdapter.checkpoint == K31_PRO
    assert MoonViTV2Adapter.model_id == "immanuelpeter/MoonViT-V2"
    assert KimiK31ProAdapter.model_id == "immanuelpeter/MoonViT-K3.1-Pro"
    assert K3.repo != K31_PRO.repo
    assert K3.revision != K31_PRO.revision
    assert MoonViTV2Adapter.extract is KimiK31ProAdapter.extract


@pytest.mark.parametrize("adapter_class", [MoonViTV2Adapter, KimiK31ProAdapter])
def test_tower_and_projector_share_immutable_revision_without_changing_k3(monkeypatch, adapter_class):
    calls = []
    class Model(torch.nn.Module):
        config = SimpleNamespace(num_hidden_layers=27)
    def load(model_id, **kwargs):
        calls.append(("tower", model_id, kwargs.get("revision")))
        return Model()
    def projector(dtype, model_id, revision=None):
        calls.append(("projector", model_id, revision))
        return torch.nn.Identity()
    monkeypatch.setattr(moonvit_v2.AutoModel, "from_pretrained", load)
    monkeypatch.setattr(moonvit_v2, "load_projector_weights", projector)
    adapter_class(device="cpu")
    revision = adapter_class.checkpoint.bundle_revision
    assert calls == [("tower", adapter_class.model_id, revision), ("projector", adapter_class.model_id, revision)]
    expected = "bcaedfccd1f07d23ab20de8a0a168668c27cacb3" if adapter_class is MoonViTV2Adapter else "129e85c01ce4db85044bc49352ad359fc12dd5ff"
    assert revision == expected


@pytest.mark.parametrize("checkpoint", CHECKPOINTS.values())
def test_discovery_uses_pinned_index_not_shard_constants(tmp_path, monkeypatch, checkpoint):
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"weight_map": {
        "vision_tower.a": "tower.safetensors", "vision_tower.b": "other.safetensors",
        "mm_projector.a": "projector.safetensors", "language_model.a": "lm.safetensors",
    }}))
    def download(repo, name, revision):
        assert (repo, revision) == (checkpoint.repo, checkpoint.revision)
        assert name == "model.safetensors.index.json"
        return str(index)
    monkeypatch.setattr(moonvit_v2, "hf_hub_download", download)
    assert moonvit_v2.discover_shards(checkpoint) == {
        "vision_tower.": ["other.safetensors", "tower.safetensors"],
        "mm_projector.": ["projector.safetensors"],
    }


def test_read_parent_cannot_reuse_k3_for_pro(monkeypatch):
    def load(repo, shards, prefix, revision):
        assert revision == (K3.revision if repo == K3.repo else K31_PRO.revision)
        return {"a": torch.tensor(1 if repo == K3.repo else 2)}
    monkeypatch.setattr(moonvit_v2, "load_prefixed", load)
    assert moonvit_v2.read_parent(K3, ["any"], "vision_tower.")["a"].item() == 1
    assert moonvit_v2.read_parent(K31_PRO, ["any"], "vision_tower.")["a"].item() == 2


@pytest.mark.parametrize("adapter_class", [MoonViTV2Adapter, KimiK31ProAdapter])
def test_kimi_batch_one_is_enforced_before_forward(adapter_class):
    adapter = adapter_class.__new__(adapter_class)
    with pytest.raises(ValueError, match="batch size 1"):
        list(adapter.extract({}, ["a", "b"]))


spec = importlib.util.spec_from_file_location("kimi_projector", Path(__file__).parents[1] / "hf/MoonViT-K3.1-Pro/projector.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Projector = module.Projector
from vtb.adapters.moonvit_v2 import load_projector_weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_example_projector_matches_adapter_and_exact_gelu(tmp_path, dtype):
    config = dict(input_size=4, hidden_size=4, output_size=6, norm_eps=1e-5)
    weights = {
        "proj.0.weight": torch.eye(4, dtype=dtype),
        "proj.2.weight": torch.arange(24, dtype=dtype).reshape(6, 4) / 24,
        "post_norm.weight": torch.linspace(.5, 1.5, 6, dtype=dtype),
    }
    save_file(weights, tmp_path / "projector.safetensors")
    example = Projector(config).to(dtype).eval()
    example.load_state_dict(weights, strict=True)
    adapter = load_projector_weights(dtype, str(tmp_path))
    assert example.proj[1].approximate == adapter.proj[1].approximate == "none"
    assert set(example.state_dict()) == set(adapter.state_dict()) == set(weights)
    tokens = torch.tensor([[[-2., -.5, .5, 2.], [-1., 0., 1., 3.]]], dtype=dtype)
    with torch.inference_mode():
        expected = adapter.post_norm(adapter.proj(tokens))
        torch.testing.assert_close(example(tokens), expected, atol=0, rtol=0)


@pytest.mark.skipif(__import__("os").environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1")
@pytest.mark.parametrize("checkpoint", moonvit_v2.CHECKPOINTS.values(), ids=moonvit_v2.CHECKPOINTS)
def test_preprocess_matches_the_published_processor(checkpoint):
    processor = AutoImageProcessor.from_pretrained(checkpoint.model_id, revision=checkpoint.bundle_revision, trust_remote_code=True)
    crop = transforms.Compose([transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(448)])
    image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (480, 640, 3), dtype=np.uint8))
    reference = processor(crop(image), return_tensors="pt")
    ours = moonvit_v2.Preprocess(448)(image)
    assert ours.shape == reference["pixel_values"].shape
    assert ours.allclose(reference["pixel_values"], atol=1e-6)
    assert (moonvit_v2.collate([ours])["grid_thws"] == reference["grid_thws"]).all()
