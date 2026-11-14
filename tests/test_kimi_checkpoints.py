import json
from types import SimpleNamespace

import pytest
import torch

from scripts import export_moonvit_v2 as export
from vtb import probe
from vtb.adapters.moonvit_v2 import CHECKPOINTS, K3, K31_PRO, KimiK31ProAdapter, MoonViTV2Adapter
from vtb.probe_run import Split, fit_reducer
from vtb import probe_run
from vtb.adapters import moonvit_v2


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
    expected = "bcaedfccd1f07d23ab20de8a0a168668c27cacb3" if adapter_class is MoonViTV2Adapter else "6ab95e2daf86e8100be2e712a7a71c0fba26dac4"
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
    monkeypatch.setattr(export, "hf_hub_download", download)
    assert export.discover_shards(checkpoint) == {
        "vision_tower.": ["other.safetensors", "tower.safetensors"],
        "mm_projector.": ["projector.safetensors"],
    }


def test_read_prefix_cannot_reuse_k3_for_pro(monkeypatch):
    def download(repo, shard, revision):
        return repo
    monkeypatch.setattr(export, "hf_hub_download", download)
    monkeypatch.setattr(export, "load_file", lambda repo: {
        "vision_tower.a": torch.tensor(1 if repo == K3.repo else 2),
        "mm_projector.a": torch.tensor(99),
    })
    assert export.read_prefix("any", "vision_tower.", K3)["a"].item() == 1
    assert export.read_prefix("any", "vision_tower.", K31_PRO)["a"].item() == 2


@pytest.mark.parametrize("adapter_class", [MoonViTV2Adapter, KimiK31ProAdapter])
def test_kimi_batch_one_is_enforced_before_forward(adapter_class):
    adapter = adapter_class.__new__(adapter_class)
    with pytest.raises(ValueError, match="batch size 1"):
        list(adapter.extract({}, ["a", "b"]))


def test_diagnostic_implements_archived_formula_and_primary_is_unchanged():
    torch.manual_seed(0)
    tokens = torch.randn(32, 4, 6)
    split = Split(torch.arange(24), torch.arange(24, 28), torch.arange(28, 32))
    primary = fit_reducer(tokens, split, 3)
    assert type(primary) is probe.Reducer
    torch.testing.assert_close(primary(tokens), (tokens - primary.mean) @ primary.basis, rtol=0, atol=0)
    diagnostic = fit_reducer(tokens, split, 3, standardize_channels=True)
    flat = tokens[split.train].flatten(0, 1).float()
    torch.testing.assert_close(diagnostic.mean, flat.mean(0, keepdim=True))
    torch.testing.assert_close(diagnostic.scale, flat.std(0, keepdim=True, correction=1).clamp_min(1e-6))
    massive = tokens * torch.tensor([1., 1., 1e4, 1., 1., 1.])
    scaled = fit_reducer(massive, split, 3, standardize_channels=True)(massive).flatten(0, 1)
    base = diagnostic(tokens).flatten(0, 1)
    torch.testing.assert_close(scaled @ scaled.T, base @ base.T, rtol=1e-3, atol=1e-3)


def test_learning_rate_protocol_and_actual_seed_evidence(monkeypatch):
    calls = []
    def train(features, labels, split, classes, readout, rate, seed, device, epochs, batch_size):
        calls.append((rate, seed))
        return None, 1.0 if rate == .003 else 0.0, .5 + .01 * seed
    monkeypatch.setattr(probe_run, "train_once", train)
    args = SimpleNamespace(learning_rates=probe_run.LEARNING_RATES, readout="attention",
                           device="cpu", epochs=20, batch_size=256, seeds=3)
    result = probe_run.run_cell(None, None, None, 100, args)
    assert calls[:11] == [(rate, 0) for rate in probe_run.LEARNING_RATES]
    assert calls[11:] == [(.003, seed) for seed in range(3)]
    assert result["learning_rate_search_seed"] == 0
    assert result["seed_ids"] == [0, 1, 2]
    assert result["test_accuracy_by_seed"] == [.5, .51, .52]
