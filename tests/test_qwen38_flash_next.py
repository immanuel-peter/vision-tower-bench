import pytest
import torch
from transformers import Qwen3_5VisionConfig, Qwen3_5VisionModel

from vtb.adapters import qwen3_5 as qwen
from vtb.extract import ADAPTERS


def test_flash_descriptor_and_registry():
    assert ADAPTERS["qwen38_flash_next"] is qwen.Qwen38FlashNextAdapter
    assert qwen.FLASH_NEXT.out_hidden_size == 2560
    assert qwen.FLASH_NEXT.license == "qwen-community-1.0"
    assert qwen.FLASH_NEXT.revision == "de4b8e4d43b917e7706784d8bb445c9af86a3540"
    assert qwen.QWEN27B.out_hidden_size == 5120


@pytest.mark.parametrize("checkpoint", qwen.CHECKPOINTS.values())
def test_source_discovery_uses_pinned_index(monkeypatch, checkpoint):
    state = {f"tensor{i}": torch.ones(1) for i in range(333)}
    index = {qwen.VISION_PREFIX + k: "discovered.safetensors" for k in state}
    index["model.language.weight"] = "do-not-download.safetensors"
    monkeypatch.setattr(qwen, "source_json", lambda c, f: {"weight_map": index})
    calls = []
    def load(repo, shards, prefix, revision):
        calls.append((repo, shards, prefix, revision))
        return state
    monkeypatch.setattr(qwen, "load_prefixed", load)
    assert qwen.source_weights(checkpoint) is state
    assert calls == [(checkpoint.repo, ["discovered.safetensors"], qwen.VISION_PREFIX, checkpoint.revision)]
    state.pop("tensor0")
    with pytest.raises(ValueError, match="incomplete"):
        qwen.source_weights(checkpoint)


def test_parent_model_type_is_rewritten_and_width_overridden(monkeypatch):
    config = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                  out_hidden_size=8, num_position_embeddings=4, patch_size=2,
                  temporal_patch_size=1, spatial_merge_size=1)
    expected = Qwen3_5VisionModel(Qwen3_5VisionConfig(**dict(config, out_hidden_size=2560)))
    monkeypatch.setattr(qwen, "source_json", lambda c, f: {"vision_config": dict(config, model_type="qwen4_exp")})
    monkeypatch.setattr(qwen, "source_weights", lambda c: expected.state_dict())
    loaded = qwen.load_source_tower(qwen.FLASH_NEXT, torch.float32)
    assert loaded.config.model_type == "qwen3_5_vision"
    assert loaded.merger.linear_fc2.weight.shape == (2560, 16)
    assert set(loaded.state_dict()) == set(expected.state_dict())


@pytest.mark.parametrize("resolution", [0, -32, 447, 16])
def test_invalid_resolution_fails_before_model_load(resolution):
    with pytest.raises(ValueError, match="multiple of 32"):
        qwen.Preprocess(resolution)

