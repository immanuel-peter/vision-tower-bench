import hashlib
import importlib.util
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import yaml

from scripts import export_minimax_m3_vision as exporter
from vtb.adapters.minimax_m3 import FLASH, Preprocess

ROOT = Path(__file__).parents[1]
BUNDLE = ROOT / "hf/MiniMax-M3.1-Flash-Vision"


def example(path):
    spec = importlib.util.spec_from_file_location("minimax_release_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_house_card_structure_and_shapes():
    text = (BUNDLE / "README.md").read_text()
    frontmatter = yaml.safe_load(text.split("---", 2)[1])
    assert frontmatter["pipeline_tag"] == "image-feature-extraction"
    assert frontmatter["library_name"] == "transformers"
    assert frontmatter["base_model"] == [FLASH.repo]
    assert "vision" in frontmatter["tags"]
    assert 55 <= len(text.splitlines()) <= 70
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert len(headings) == 8
    assert headings[1:] == ["## Contents", "## Architecture", "## Usage", "## Validation",
                            "## Reproduction", "## Credits", "## License"]
    for shape in ("(1, 1024, 1280)", "(1, 256, 5120)", "(1, 256, 6144)"):
        assert shape in text


def test_manifest_covers_nested_and_hidden_files_without_self_hash(monkeypatch, tmp_path):
    out = tmp_path / ".cache" / "release"
    fixtures = {"README.md": b"card", ".gitattributes": b"lfs",
                "model.safetensors": b"tower", "projector.safetensors": b"projector",
                "examples/inference.py": b"example"}
    for name, value in fixtures.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    for name in ("export_manifest.json", "__pycache__/module.pyc", ".cache/runtime"):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ignored")
    monkeypatch.setattr(exporter, "source_json", lambda *_: {"weight_map": {
        "vision_tower.vision_model.weight": "z.safetensors",
        "multi_modal_projector.weight": "a.safetensors",
        "patch_merge_mlp.weight": "z.safetensors",
        "language_model.weight": "unrelated.safetensors",
    }})
    from safetensors.torch import save_file
    save_file({"tower.weight": torch.ones(1, dtype=torch.bfloat16)}, out / "model.safetensors")
    save_file({"proj.weight": torch.ones(1, dtype=torch.bfloat16)}, out / "projector.safetensors")
    fixtures.update({name: (out / name).read_bytes() for name in ("model.safetensors", "projector.safetensors")})
    result = exporter.write_manifest(out, FLASH)
    assert result["source_shards"] == ["a.safetensors", "z.safetensors"]
    assert (result["tower_tensors"], result["projector_tensors"]) == (1, 1)
    assert result["dtype"] == "bfloat16"
    assert result["hash_excludes"] == ["export_manifest.json"]
    assert result["files_sha256"] == {name: hashlib.sha256(value).hexdigest()
                                      for name, value in fixtures.items() if name != ".gitattributes"}
    assert result["hub_only_metadata"] == [".gitattributes"]
    first = (out / "export_manifest.json").read_bytes()
    exporter.write_manifest(out, FLASH)
    assert (out / "export_manifest.json").read_bytes() == first


def test_example_patch_packing_matches_the_benchmark():
    module = example(BUNDLE / "examples/inference.py")
    pixels = np.arange(43 * 57 * 3, dtype=np.uint8).reshape(43, 57, 3)
    image = Image.fromarray(pixels)
    assert torch.equal(module.preprocess(image), Preprocess(448)(image))


def test_example_keeps_snapshot_root_when_script_is_a_symlink(tmp_path):
    (tmp_path / "examples").mkdir()
    path = tmp_path / "examples/inference.py"
    path.symlink_to(BUNDLE / "examples/inference.py")
    module = example(path)
    assert module.BUNDLE == tmp_path
