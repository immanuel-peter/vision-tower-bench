"""Verify the weights each adapter loads against their parent checkpoints.

Four adapters now load a republished Tower, so these tests are what ties the repository
the bench probes back to the checkpoint it came out of.

These tests compare state dictionaries. They do not run the parent language models.
"""

import os

import pytest
import torch

from vtb.adapters import kimi_k26, muse_glimmer, qwen3_5, siglip2
from vtb.shards import load_prefixed

pytestmark = pytest.mark.skipif(
    os.environ.get("VTB_SKIP_WEIGHTS") == "1", reason="VTB_SKIP_WEIGHTS=1"
)


def assert_bit_exact(
    module_weights: dict[str, torch.Tensor],
    checkpoint_weights: dict[str, torch.Tensor],
    count: int,
) -> None:
    assert len(checkpoint_weights) == count
    assert set(module_weights) == set(checkpoint_weights)
    for name, weight in checkpoint_weights.items():
        assert torch.equal(module_weights[name], weight), name


def test_siglip2_tower_matches_published_checkpoint():
    tower = siglip2.SigLIP2Adapter(device="cpu", dtype=torch.float32).model
    checkpoint_weights = load_prefixed(siglip2.MODEL_ID, ["model.safetensors"], "vision_model.")
    assert_bit_exact(tower.state_dict(), checkpoint_weights, 448)


def test_qwen_tower_matches_parent_checkpoint():
    tower = qwen3_5.load_tower(torch.bfloat16)
    checkpoint_weights = load_prefixed(
        qwen3_5.SOURCE_REPO, [qwen3_5.VISION_SHARD], qwen3_5.VISION_PREFIX
    )
    assert_bit_exact(tower.state_dict(), checkpoint_weights, 333)


def test_muse_tower_and_projector_match_parent_checkpoint():
    tower, projector = muse_glimmer.load_parts(torch.bfloat16)
    checkpoint_weights = load_prefixed(
        muse_glimmer.SOURCE_REPO, muse_glimmer.SHARDS, muse_glimmer.TOWER_PREFIX
    )
    assert_bit_exact(tower.state_dict(), checkpoint_weights, 806)

    for module, prefix, count in (
        (projector.adapter, muse_glimmer.ADAPTER_PREFIX, 2),
        (projector.projection, muse_glimmer.PROJECTION_PREFIX, 1),
    ):
        checkpoint_weights = load_prefixed(muse_glimmer.SOURCE_REPO, muse_glimmer.SHARDS, prefix)
        assert_bit_exact(module.state_dict(), checkpoint_weights, count)


def test_kimi_k26_republished_weights_match_parent_checkpoint():
    tower, projector = kimi_k26.load_parts(torch.bfloat16)
    for module, prefix, count in (
        (tower, kimi_k26.TOWER_PREFIX, 329),
        (projector, kimi_k26.PROJECTOR_PREFIX, 6),
    ):
        checkpoint_weights = load_prefixed(kimi_k26.SOURCE_REPO, kimi_k26.SOURCE_SHARDS, prefix)
        # Ignore buffers the standalone model rebuilds at load time.
        loaded = {k: v for k, v in module.state_dict().items() if k in checkpoint_weights}
        assert_bit_exact(loaded, checkpoint_weights, count)
