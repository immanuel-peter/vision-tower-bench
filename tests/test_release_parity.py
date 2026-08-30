"""Verify local release bundles against their pinned parent checkpoints."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel

from scripts.export_moonvit_k26 import SOURCE_REVISION as KIMI_REVISION
from scripts.export_muse_glimmer_vision import SOURCE_REVISION as MUSE_REVISION
from scripts.export_qwen3_8_vision import SOURCE_REVISION as QWEN_REVISION
from vtb.adapters import kimi_k26, muse_glimmer, qwen3_5
from vtb.shards import load_prefixed

ROOT = Path(__file__).parents[1]
QWEN_BUNDLE = ROOT / "hf/Qwen3.8-27B-Vision"
MUSE_BUNDLE = ROOT / "hf/Muse-Glimmer-Vision"
KIMI_BUNDLE = ROOT / "hf/MoonViT-K2.6"
DEVICE = torch.device(os.environ.get("VTB_RELEASE_DEVICE", "cpu"))
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

pytestmark = pytest.mark.skipif(
    os.environ.get("VTB_SKIP_WEIGHTS") == "1",
    reason="VTB_SKIP_WEIGHTS=1",
)


def assert_bit_exact(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert set(actual) == set(expected)
    for name, weight in expected.items():
        assert torch.equal(actual[name], weight), name


def load_projector_module(bundle: Path):
    spec = importlib.util.spec_from_file_location("release_projector", bundle / "projector.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_projector(bundle)


def test_qwen_release_weights_match_the_pinned_parent():
    source = load_prefixed(
        qwen3_5.MODEL_ID,
        [qwen3_5.VISION_SHARD],
        qwen3_5.VISION_PREFIX,
        revision=QWEN_REVISION,
    )
    assert len(source) == 333
    assert_bit_exact(load_file(QWEN_BUNDLE / "model.safetensors"), source)


def test_muse_release_weights_match_the_pinned_parent():
    source_tower = load_prefixed(
        muse_glimmer.MODEL_ID,
        muse_glimmer.SHARDS,
        muse_glimmer.TOWER_PREFIX,
        revision=MUSE_REVISION,
    )
    assert len(source_tower) == 806
    assert_bit_exact(load_file(MUSE_BUNDLE / "model.safetensors"), source_tower)

    source_projector = {
        "adapter." + name: weight
        for name, weight in load_prefixed(
            muse_glimmer.MODEL_ID,
            muse_glimmer.SHARDS,
            muse_glimmer.ADAPTER_PREFIX,
            revision=MUSE_REVISION,
        ).items()
    }
    source_projector.update({
        "projection." + name: weight
        for name, weight in load_prefixed(
            muse_glimmer.MODEL_ID,
            muse_glimmer.SHARDS,
            muse_glimmer.PROJECTION_PREFIX,
            revision=MUSE_REVISION,
        ).items()
    })
    assert len(source_projector) == 3
    assert_bit_exact(load_file(MUSE_BUNDLE / "projector.safetensors"), source_projector)


def test_kimi_release_weights_match_the_pinned_parent():
    for filename, prefix, count in (
        ("model.safetensors", kimi_k26.TOWER_PREFIX, 329),
        ("projector.safetensors", kimi_k26.PROJECTOR_PREFIX, 6),
    ):
        source = load_prefixed(
            "moonshotai/Kimi-K2.6",
            ("model-00063-of-000064.safetensors", "model-00064-of-000064.safetensors"),
            prefix,
            revision=KIMI_REVISION,
        )
        assert len(source) == count
        assert_bit_exact(load_file(KIMI_BUNDLE / filename), source)


def test_qwen_release_forward_matches_the_parent():
    source = qwen3_5.load_tower(DTYPE).to(DEVICE)
    released = AutoModel.from_pretrained(
        QWEN_BUNDLE,
        dtype=DTYPE,
        local_files_only=True,
    ).to(DEVICE)
    sample = qwen3_5.Preprocess(448)(Image.new("RGB", (640, 480), "gray"))
    inputs = qwen3_5.collate([sample])

    with torch.inference_mode():
        hidden_states = inputs["hidden_states"].to(DEVICE, DTYPE)
        grid_thw = inputs["grid_thw"].to(DEVICE)
        expected = source(hidden_states, grid_thw)
        actual = released(hidden_states, grid_thw)

    assert torch.equal(actual.last_hidden_state, expected.last_hidden_state)
    assert torch.equal(actual.pooler_output, expected.pooler_output)


def test_muse_release_forward_matches_the_parent():
    source, source_projector = muse_glimmer.load_parts(DTYPE)
    source = source.to(DEVICE)
    source_projector = source_projector.to(DEVICE)
    released = AutoModel.from_pretrained(
        MUSE_BUNDLE,
        dtype=DTYPE,
        local_files_only=True,
    ).to(DEVICE)
    released_projector = load_projector_module(MUSE_BUNDLE).to(DEVICE, DTYPE).eval()
    sample = muse_glimmer.Preprocess(448)(Image.new("RGB", (640, 480), "gray"))
    inputs = muse_glimmer.collate([sample])

    with torch.inference_mode():
        pixel_values = inputs["pixel_values"].to(DEVICE, DTYPE)
        grid_thw = inputs["grid_thw"].to(DEVICE)
        expected_merged = source(pixel_values, grid_thw).last_hidden_state
        actual_merged = released(pixel_values, grid_thw).last_hidden_state
        expected_projected = source_projector(expected_merged)
        actual_projected = released_projector(actual_merged)

    assert torch.equal(actual_merged, expected_merged)
    assert torch.equal(actual_projected, expected_projected)


def test_kimi_release_forward_matches_the_parent():
    source, source_projector = kimi_k26.load_parts(DTYPE, attention="eager")
    source = source.to(DEVICE)
    source_projector = source_projector.to(DEVICE)
    released = AutoModel.from_pretrained(
        KIMI_BUNDLE,
        dtype=DTYPE,
        local_files_only=True,
        trust_remote_code=True,
    ).to(DEVICE)
    released_projector = load_projector_module(KIMI_BUNDLE).to(DEVICE, DTYPE).eval()
    sample = kimi_k26.Preprocess(448)(Image.new("RGB", (640, 480), "gray"))
    inputs = kimi_k26.collate([sample])

    with torch.inference_mode():
        pixel_values = inputs["pixel_values"].to(DEVICE, DTYPE)
        grid_thws = inputs["grid_thws"].to(DEVICE)
        expected_merged = source(pixel_values, grid_thws)
        actual_merged = released(pixel_values, grid_thws)
        expected_projected = source_projector(expected_merged)
        actual_projected = released_projector(actual_merged)

    for actual, expected in zip(actual_merged, expected_merged, strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(actual_projected, expected_projected, strict=True):
        assert torch.equal(actual, expected)
