"""Verify the weights each adapter loads against their parent checkpoints.

Four adapters now load a republished Tower, so these tests are what ties the repository
the bench probes back to the checkpoint it came out of.

These tests compare state dictionaries. They do not run the parent language models.
"""

import os

import pytest
import torch

from vtb.adapters import deepseek_v41, gemma4, glm5, kimi_k26, minimax_m3, muse_glimmer, nemotron_omni, qwen3_5, siglip2
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


def test_gemma4_tower_and_projector_match_parent_checkpoint():
    tower, projector = gemma4.load_source_parts(torch.bfloat16)
    checkpoint_weights = load_prefixed(
        gemma4.SOURCE_REPO,
        [gemma4.VISION_SHARD],
        gemma4.TOWER_PREFIX,
        revision=gemma4.SOURCE_REVISION,
    )
    loaded = {k: v for k, v in tower.state_dict().items() if k in checkpoint_weights}
    assert_bit_exact(loaded, checkpoint_weights, 355)

    checkpoint_weights = load_prefixed(
        gemma4.SOURCE_REPO,
        [gemma4.VISION_SHARD],
        gemma4.PROJECTOR_PREFIX,
        revision=gemma4.SOURCE_REVISION,
    )
    loaded = {k: v for k, v in projector.state_dict().items() if k in checkpoint_weights}
    assert_bit_exact(loaded, checkpoint_weights, 1)


def test_glm5_tower_matches_parent_checkpoint():
    tower = glm5.load_source_tower(torch.bfloat16)
    checkpoint_weights = load_prefixed(
        glm5.SOURCE_REPO,
        [glm5.VISION_SHARD],
        glm5.VISION_PREFIX,
        revision=glm5.SOURCE_REVISION,
    )
    loaded = {k: v for k, v in tower.state_dict().items() if k in checkpoint_weights}
    assert_bit_exact(loaded, checkpoint_weights, 347)


def test_minimax_m3_tower_and_projector_match_parent_checkpoint():
    tower, projector = minimax_m3.load_source_parts(torch.bfloat16)
    checkpoint_weights = minimax_m3.remap_tower(
        load_prefixed(
            minimax_m3.SOURCE_REPO,
            [minimax_m3.VISION_SHARD],
            minimax_m3.TOWER_PREFIX,
            revision=minimax_m3.SOURCE_REVISION,
        )
    )
    loaded = {k: v for k, v in tower.state_dict().items() if k in checkpoint_weights}
    assert set(loaded) == set(checkpoint_weights)
    assert len(checkpoint_weights) == 515
    for name, weight in checkpoint_weights.items():
        assert torch.equal(loaded[name], weight.to(loaded[name].dtype)), name

    checkpoint_weights = minimax_m3.remap_projector(
        load_prefixed(
            minimax_m3.SOURCE_REPO,
            list(minimax_m3.PROJECTOR_SHARDS),
            minimax_m3.PROJECTOR_PREFIX,
            revision=minimax_m3.SOURCE_REVISION,
        ),
        load_prefixed(
            minimax_m3.SOURCE_REPO,
            list(minimax_m3.PROJECTOR_SHARDS),
            minimax_m3.MERGE_MLP_PREFIX,
            revision=minimax_m3.SOURCE_REVISION,
        ),
    )
    loaded = {k: v for k, v in projector.state_dict().items() if k in checkpoint_weights}
    assert set(loaded) == set(checkpoint_weights)
    assert len(checkpoint_weights) == 8
    for name, weight in checkpoint_weights.items():
        assert torch.equal(loaded[name], weight.to(loaded[name].dtype)), name


def test_nemotron_omni_tower_and_projector_match_parent_checkpoint():
    radio, projector = nemotron_omni.load_source_parts(torch.bfloat16)
    checkpoint_weights = load_prefixed(
        nemotron_omni.SOURCE_REPO,
        [nemotron_omni.VISION_SHARD],
        nemotron_omni.TOWER_PREFIX,
        revision=nemotron_omni.SOURCE_REVISION,
    )
    loaded = radio.state_dict()
    matched = {k: v for k, v in loaded.items() if k in checkpoint_weights}
    assert len(checkpoint_weights) == 390
    for name, weight in checkpoint_weights.items():
        assert name in loaded, name
        assert torch.equal(loaded[name], weight.to(loaded[name].dtype)), name

    checkpoint_weights = load_prefixed(
        nemotron_omni.SOURCE_REPO,
        [nemotron_omni.VISION_SHARD],
        nemotron_omni.PROJECTOR_PREFIX,
        revision=nemotron_omni.SOURCE_REVISION,
    )
    loaded = {k: v for k, v in projector.state_dict().items() if k in checkpoint_weights}
    assert len(checkpoint_weights) == 3
    for name, weight in checkpoint_weights.items():
        assert torch.equal(loaded[name], weight.to(loaded[name].dtype)), name


def test_deepseek_v41_tower_and_aligner_match_parent_checkpoint():
    tower, aligner = deepseek_v41.load_source_parts(torch.bfloat16)
    checkpoint_weights = load_prefixed(
        deepseek_v41.SOURCE_REPO,
        [deepseek_v41.VISION_SHARD],
        deepseek_v41.TOWER_PREFIX,
        revision=deepseek_v41.SOURCE_REVISION,
    )
    loaded = {k: v for k, v in tower.state_dict().items() if k in checkpoint_weights}
    assert len(checkpoint_weights) == 259
    for name, weight in checkpoint_weights.items():
        assert torch.equal(loaded[name], weight.to(loaded[name].dtype)), name

    checkpoint_weights = load_prefixed(
        deepseek_v41.SOURCE_REPO,
        [deepseek_v41.VISION_SHARD],
        deepseek_v41.ALIGNER_PREFIX,
        revision=deepseek_v41.SOURCE_REVISION,
    )
    loaded = {k: v for k, v in aligner.state_dict().items() if k in checkpoint_weights}
    assert len(checkpoint_weights) == 4
    for name, weight in checkpoint_weights.items():
        assert torch.equal(loaded[name], weight.to(loaded[name].dtype)), name


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
