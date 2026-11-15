"""Verify local release bundles against their pinned parent checkpoints."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel

from scripts.export_moonvit_k26_vision import SOURCE_REVISION as KIMI_REVISION
from scripts.export_moonvit_k26_vision import load_source_parts as load_kimi_source
from scripts.export_muse_glimmer_vision import SOURCE_REVISION as MUSE_REVISION
from scripts.export_muse_glimmer_vision import load_source_parts as load_muse_source
from scripts.export_qwen3_8_vision import SOURCE_REVISION as QWEN_REVISION
from scripts.export_qwen3_8_vision import load_source_tower as load_qwen_source
from scripts.validate_mimo26 import assert_parity as assert_mimo26_parity, validate as validate_mimo26
from scripts.validate_nemotron35 import assert_parity, validate as validate_nemotron35
from scripts.validate_mistral_large4 import validate as validate_mistral_large4
from vtb.adapters import kimi_k26, mimo26, mistral_large4, muse_glimmer, nemotron35, qwen3_5
from vtb.shards import load_prefixed

ROOT = Path(__file__).parents[1]
QWEN_BUNDLE = ROOT / "hf/Qwen3.8-27B-Vision"
MUSE_BUNDLE = ROOT / "hf/Muse-Glimmer-Vision"
KIMI_BUNDLE = ROOT / "hf/MoonViT-K2.6"
NEMOTRON35_BUNDLES = {
    nemotron35.SUPER.name: ROOT / "hf/C-RADIOv4-H-3.5-Super",
    nemotron35.BOLT.name: ROOT / "hf/C-RADIOv4-H-3.5-Bolt",
}
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


@pytest.mark.parametrize("checkpoint", qwen3_5.CHECKPOINTS.values(), ids=qwen3_5.CHECKPOINTS)
def test_qwen_release_weights_match_the_pinned_parent(checkpoint):
    source = qwen3_5.source_weights(checkpoint)
    assert len(source) == 333
    actual = load_file(checkpoint.bundle / "model.safetensors")
    assert_bit_exact(actual, source)
    assert all(tensor.dtype == source[name].dtype == torch.bfloat16 for name, tensor in actual.items())
    assert all(torch.equal(tensor.view(torch.uint8), source[name].view(torch.uint8))
               for name, tensor in actual.items())
    assert actual["merger.linear_fc2.weight"].shape == (checkpoint.out_hidden_size, 4608)


def test_qwen_siblings_do_not_share_the_exported_weights():
    first = load_file(qwen3_5.QWEN27B.bundle / "model.safetensors")
    second = load_file(qwen3_5.FLASH_NEXT.bundle / "model.safetensors")
    assert set(first) == set(second)
    shared = [name for name in first if first[name].shape == second[name].shape]
    assert len(shared) == 331
    assert all(not torch.equal(first[name], second[name]) for name in shared)


def test_muse_release_weights_match_the_pinned_parent():
    source_tower = load_prefixed(
        muse_glimmer.SOURCE_REPO,
        muse_glimmer.SHARDS,
        muse_glimmer.TOWER_PREFIX,
        revision=MUSE_REVISION,
    )
    assert len(source_tower) == 806
    assert_bit_exact(load_file(MUSE_BUNDLE / "model.safetensors"), source_tower)

    source_projector = {
        "adapter." + name: weight
        for name, weight in load_prefixed(
            muse_glimmer.SOURCE_REPO,
            muse_glimmer.SHARDS,
            muse_glimmer.ADAPTER_PREFIX,
            revision=MUSE_REVISION,
        ).items()
    }
    source_projector.update({
        "projection." + name: weight
        for name, weight in load_prefixed(
            muse_glimmer.SOURCE_REPO,
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
            kimi_k26.SOURCE_REPO,
            kimi_k26.SOURCE_SHARDS,
            prefix,
            revision=KIMI_REVISION,
        )
        assert len(source) == count
        assert_bit_exact(load_file(KIMI_BUNDLE / filename), source)


@pytest.mark.parametrize("checkpoint", qwen3_5.CHECKPOINTS.values(), ids=qwen3_5.CHECKPOINTS)
def test_qwen_release_forward_matches_the_parent(checkpoint):
    source = load_qwen_source(DTYPE, checkpoint).to(DEVICE)
    released = AutoModel.from_pretrained(
        checkpoint.bundle,
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
    source, source_projector = load_muse_source(DTYPE)
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
    source, source_projector = load_kimi_source(DTYPE, attention="eager")
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


@pytest.mark.parametrize("name", sorted(NEMOTRON35_BUNDLES))
def test_nemotron35_release_matches_the_independent_parent(name):
    report = validate_nemotron35(name, NEMOTRON35_BUNDLES[name], DEVICE, dtype=DTYPE)

    assert report["weights_bit_exact"]
    assert len(report["cells"]) == 10
    assert set(report["release_cells"]) == {"tower", "merged", "projected"}
    assert_parity(report, DTYPE)


@pytest.mark.parametrize("name", sorted(mimo26.CHECKPOINTS))
def test_mimo26_release_matches_the_zero_completed_parent(name):
    report = validate_mimo26(name, mimo26.CHECKPOINTS[name].release_dir, DEVICE, dtype=DTYPE)

    assert report["weights_bit_exact"]
    assert report["zero_completed_biases"] == ["merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias"]
    assert [len(case["features"]) for case in report["cases"]] == [10, 10]
    assert_mimo26_parity(report, DTYPE)


def test_mistral_large4_release_matches_the_independent_mistral_forward(tmp_path):
    if DEVICE.type != "cuda":
        pytest.skip("the 1540 px FP32 reference needs VTB_RELEASE_DEVICE=cuda")
    report = validate_mistral_large4(
        tmp_path / "report.json", tmp_path / "reference", bundle=mistral_large4.RELEASE_DIR,
        resolutions=[448, 1540], device=str(DEVICE),
    )
    release = report["identity"]["release"]
    assert report["identity"]["shard_sha256"] == report["identity"]["shard_lfs_sha256"]
    assert release["bit_exact_after_inverse_permutation"] and release["manifest_sha256_all_match"]
    assert release["tensors"] == 276
    assert {case["resolution"] for case in report["cases"]} == {448, 1540}
    assert len(report["cases"]) == 80
    assert all(case["adapter_fp32"]["relative"] <= 1e-4 for case in report["cases"])
    # The release reproduces the reference BF16 forward exactly. The BF16-vs-FP32
    # gate is not asserted: the parent's own BF16 forward misses it at deep Stages.
    assert report["all_direct_bf16_bit_exact"]
    assert report["release_equals_adapter_all"]
    assert report["all_controls_detected"]
