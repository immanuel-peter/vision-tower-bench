"""The DINOv3 local-bundle option: provenance checks and load routing."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from vtb.adapters import dinov3


def digest(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A tiny DINOv3 saved like the restored bundle, with its own pinned hash."""
    monkeypatch.delenv(dinov3.BUNDLE_ENV, raising=False)
    torch.manual_seed(0)
    config = DINOv3ViTConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        patch_size=16, num_register_tokens=4,
    )
    model = DINOv3ViTModel(config).eval()
    path = tmp_path / "dinov3"
    model.save_pretrained(path)
    (path / "LICENSE.md").write_text("DINOv3 License\n")
    files = {name: digest(path / name) for name in ("config.json", "LICENSE.md", dinov3.MODEL_FILE)}
    write_manifest(path, files)
    monkeypatch.setattr(dinov3, "MODEL_BYTES", files[dinov3.MODEL_FILE]["bytes"])
    monkeypatch.setattr(dinov3, "MODEL_SHA256", files[dinov3.MODEL_FILE]["sha256"])
    return SimpleNamespace(path=path, model=model, files=files)


def write_manifest(path: Path, files: dict, repo=dinov3.SOURCE_REPO, revision=dinov3.SOURCE_REVISION):
    (path / "source_manifest.json").write_text(json.dumps({
        "source_repo": repo, "source_revision": revision, "files": files,
    }))


def test_verified_bundle_loads_locally_with_identical_weights(bundle):
    manifest = dinov3.verify_bundle(bundle.path)
    assert manifest["source_revision"] == dinov3.SOURCE_REVISION
    loaded = dinov3.load_parent(torch.float32, bundle.path)
    expected = bundle.model.state_dict()
    assert set(loaded.state_dict()) == set(expected)
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, expected[name]), name
    pixels = torch.randn(2, 3, 32, 48)
    with torch.inference_mode():
        torch.testing.assert_close(loaded(pixels).last_hidden_state, bundle.model(pixels).last_hidden_state,
                                   rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("repo", "facebook/dinov3-vit7b16-pretrain-lvd1689m"),
    ("revision", "0" * 40),
])
def test_rejects_other_provenance(bundle, field, value):
    write_manifest(bundle.path, bundle.files, **{field: value})
    with pytest.raises(ValueError, match="provenance"):
        dinov3.verify_bundle(bundle.path)


def test_rejects_same_size_weights_with_different_bytes(bundle):
    weights = bundle.path / dinov3.MODEL_FILE
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA256"):
        dinov3.verify_bundle(bundle.path)


def test_rejects_a_manifest_that_vouches_for_other_weights(bundle, monkeypatch):
    # Restore the real pins: the fixture's own hash is no longer accepted.
    monkeypatch.setattr(dinov3, "MODEL_BYTES", 1_212_559_808)
    monkeypatch.setattr(dinov3, "MODEL_SHA256", "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179")
    with pytest.raises(ValueError, match="pinned model.safetensors"):
        dinov3.verify_bundle(bundle.path)


def test_rejects_changed_or_missing_listed_files(bundle):
    (bundle.path / "LICENSE.md").write_text("changed licence\n")
    with pytest.raises(ValueError, match="LICENSE.md: size"):
        dinov3.verify_bundle(bundle.path)
    (bundle.path / "LICENSE.md").unlink()
    with pytest.raises(FileNotFoundError, match="missing LICENSE.md"):
        dinov3.verify_bundle(bundle.path)


def test_rejects_a_directory_without_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="source_manifest.json"):
        dinov3.verify_bundle(tmp_path)


class Parent:
    config = SimpleNamespace(num_hidden_layers=24, hidden_size=1024, patch_size=16, num_register_tokens=4)

    def to(self, _):
        return self

    def eval(self):
        return self


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(dinov3, "verify_bundle", lambda path: seen.append(("verify", Path(path))))

    def load(*args, **kwargs):
        seen.append(("load", args, kwargs))
        return Parent()

    monkeypatch.setattr(dinov3.DINOv3ViTModel, "from_pretrained", load)
    monkeypatch.delenv(dinov3.BUNDLE_ENV, raising=False)
    return seen


def test_default_stays_on_the_pinned_hub_revision(calls):
    adapter = dinov3.DINOv3Adapter(resolution=32, dtype=torch.float32, device="cpu")
    assert calls == [("load", (dinov3.SOURCE_REPO,), {"revision": dinov3.SOURCE_REVISION, "dtype": torch.float32})]
    assert adapter.model_id == dinov3.SOURCE_REPO


def test_environment_bundle_is_verified_then_loaded_offline(calls, monkeypatch, tmp_path):
    monkeypatch.setenv(dinov3.BUNDLE_ENV, str(tmp_path))
    adapter = dinov3.DINOv3Adapter(resolution=32, dtype=torch.float32, device="cpu")
    assert calls == [("verify", tmp_path), ("load", (tmp_path,), {"dtype": torch.float32, "local_files_only": True})]
    assert adapter.model_id == dinov3.SOURCE_REPO


def test_argument_overrides_environment(calls, monkeypatch, tmp_path):
    monkeypatch.setenv(dinov3.BUNDLE_ENV, str(tmp_path / "from-env"))
    dinov3.DINOv3Adapter(resolution=32, dtype=torch.float32, device="cpu", bundle=tmp_path / "explicit")
    assert calls[0] == ("verify", tmp_path / "explicit")


def test_empty_environment_value_means_the_hub(calls, monkeypatch):
    monkeypatch.setenv(dinov3.BUNDLE_ENV, "")
    dinov3.DINOv3Adapter(resolution=32, dtype=torch.float32, device="cpu")
    assert calls[0][1] == (dinov3.SOURCE_REPO,)


def test_failed_verification_never_loads(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(dinov3.DINOv3ViTModel, "from_pretrained", lambda *a, **k: loads.append(a))
    with pytest.raises(FileNotFoundError):
        dinov3.DINOv3Adapter(resolution=32, dtype=torch.float32, device="cpu", bundle=tmp_path)
    assert loads == []


@pytest.mark.skipif(
    os.environ.get("VTB_SKIP_WEIGHTS") == "1" or not os.environ.get(dinov3.BUNDLE_ENV),
    reason="needs VTB_DINOV3_BUNDLE and weights",
)
def test_restored_bundle_is_the_pinned_vit_large():
    path = Path(os.environ[dinov3.BUNDLE_ENV])
    dinov3.verify_bundle(path)
    parent = dinov3.load_parent(torch.float32, path)
    assert sum(p.numel() for p in parent.parameters()) == 303_129_600
    assert parent.config.num_register_tokens == 4
