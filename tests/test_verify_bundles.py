"""Bundle verification on tiny synthetic MiMo and Nemotron bundles."""

import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from scripts import verify_bundles
from vtb.models.mimo26 import MiMoVisionConfig

SOURCE = SimpleNamespace(repo="org/parent", revision="a" * 40, final_layernorm=True)


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.fixture
def mimo(tmp_path, monkeypatch):
    weights = {"blocks.0.norm1.weight": torch.randn(8).bfloat16(), "merger.mlp.0.weight": torch.randn(4, 8).bfloat16()}
    tracked = tmp_path / "tracked"
    tracked.mkdir()
    MiMoVisionConfig(depth=1, hidden_size=8).save_pretrained(tracked)
    for name in ("modeling_mimo_vision.py", "preprocessor_config.json", "LICENSE", "LICENSE-CODE", "NOTICE"):
        (tracked / name).write_text(name)
    (tracked / "export_manifest.json").write_text(json.dumps({
        "source_repo": SOURCE.repo, "source_revision": SOURCE.revision, "tensors": 2,
        "bytes": sum(t.nbytes for t in weights.values()),
        "tensor_sha256": {name: digest(t) for name, t in weights.items()},
    }))
    bundle = tmp_path / "bundle"
    shutil.copytree(tracked, bundle)
    # The release config carries explicit defaults; it must still compare equal.
    config = json.loads((bundle / "config.json").read_text())
    config["qk_channels"] = 64
    (bundle / "config.json").write_text(json.dumps(config, indent=1))
    save_file({"visual." + name: t for name, t in weights.items()}, bundle / "model.safetensors")
    monkeypatch.setattr(verify_bundles.mimo26, "VISUAL_TENSORS", 2)
    return SimpleNamespace(bundle=bundle, tracked=tracked, weights=weights)


def test_mimo_bundle_passes_with_prefixed_names_and_reformatted_config(mimo):
    result = verify_bundles.verify_mimo26(mimo.bundle, SOURCE, mimo.tracked)
    assert all(result["checks"].values()), result
    assert result["notes"] == {"config_bytes_equal_tracked": False}


def test_mimo_bundle_detects_a_changed_tensor_and_wrong_source(mimo):
    changed = {"visual." + name: t.clone() for name, t in mimo.weights.items()}
    changed["visual.blocks.0.norm1.weight"][0] += 1
    save_file(changed, mimo.bundle / "model.safetensors")
    result = verify_bundles.verify_mimo26(mimo.bundle, SimpleNamespace(repo="org/other", revision=SOURCE.revision),
                                          mimo.tracked)
    assert not result["checks"]["tensor_sha256"] and not result["checks"]["provenance"]
    assert result["mismatched_tensors"] == ["visual.blocks.0.norm1.weight"]


def test_mimo_bundle_detects_a_semantic_config_change(mimo):
    config = json.loads((mimo.bundle / "config.json").read_text())
    config["depth"] = 2
    (mimo.bundle / "config.json").write_text(json.dumps(config))
    assert not verify_bundles.verify_mimo26(mimo.bundle, SOURCE, mimo.tracked)["checks"][
        "config_equals_tracked_after_defaults"]


def nemotron(tmp_path, final_layernorm, monkeypatch):
    parts = {
        "tower": {"blocks.0.weight": torch.randn(3, 2).bfloat16()},
        "projector": {"0.weight": torch.randn(4).bfloat16()},
        "final_norm": {"weight": torch.randn(2).bfloat16()} if final_layernorm else {},
    }
    tracked = tmp_path / "tracked"
    tracked.mkdir()
    (tracked / "config.json").write_text(json.dumps({
        "source": SOURCE.repo, "revision": SOURCE.revision, "vision_final_layernorm": final_layernorm,
    }))
    (tracked / "inventory.json").write_text(json.dumps({
        part: {name: {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": digest(t)} for name, t in weights.items()}
        for part, weights in parts.items()
    } | {"source_shards": ["x"]}))
    (tracked / "LICENSE").write_text("OpenMDW\n")
    bundle = tmp_path / "bundle"
    shutil.copytree(tracked, bundle)
    (bundle / "THIRD_PARTY_NOTICES.md").write_text("notices\n")
    for part, filename in (("tower", "model.safetensors"), ("projector", "projector.safetensors"),
                           ("final_norm", "final_norm.safetensors")):
        if parts[part]:
            save_file(parts[part], bundle / filename)
    monkeypatch.setattr(verify_bundles, "NEMOTRON_COUNTS", {"tower": 1, "projector": 1, "final_norm": 1})
    monkeypatch.setattr(verify_bundles, "NEMOTRON_LICENSE_SHA256", hashlib.sha256(b"OpenMDW\n").hexdigest())
    return bundle, tracked


@pytest.mark.parametrize("final_layernorm", [True, False])
def test_nemotron_bundle_passes(tmp_path, monkeypatch, final_layernorm):
    bundle, tracked = nemotron(tmp_path, final_layernorm, monkeypatch)
    checkpoint = SimpleNamespace(repo=SOURCE.repo, revision=SOURCE.revision, final_layernorm=final_layernorm)
    result = verify_bundles.verify_nemotron35(bundle, checkpoint, tracked)
    assert all(result["checks"].values()), result
    assert ("final_norm_absent" in result["checks"]) is not final_layernorm


def test_nemotron_bundle_detects_dtype_changes_and_stray_final_norm(tmp_path, monkeypatch):
    bundle, tracked = nemotron(tmp_path, False, monkeypatch)
    save_file({"0.weight": torch.zeros(4)}, bundle / "projector.safetensors")
    save_file({"weight": torch.zeros(2)}, bundle / "final_norm.safetensors")
    checkpoint = SimpleNamespace(repo=SOURCE.repo, revision=SOURCE.revision, final_layernorm=False)
    checks = verify_bundles.verify_nemotron35(bundle, checkpoint, tracked)["checks"]
    assert not checks["projector_shape_dtype_sha256"]
    assert not checks["final_norm_absent"]
