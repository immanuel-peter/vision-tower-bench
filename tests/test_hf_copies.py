import importlib.util
import runpy
import shutil
from pathlib import Path

import pytest
import torch

from scripts import render_hf_templates
from vtb.adapters import nemotron_omni

ROOT = Path(__file__).parents[1]
HF = ROOT / "hf"
SIBLING_EXAMPLES = {
    "C-RADIOv4-H-3-Omni": ("load_projector",),
    "C-RADIOv4-H-3.5-Bolt": ("load_vision",),
    "C-RADIOv4-H-3.5-Super": ("load_vision",),
    "DeepSeek-ViT": ("Aligner", "ViT"),
    "Gemma4-31B-Vision": ("load_projector",),
    "MiniMax-M3-Vision": ("load_projector",),
    "MiniMax-M3.1-Flash-Vision": ("main",),
    "MoonViT-K2.6": ("load_projector",),
    "MoonViT-K3.1-Pro": ("Projector",),
    "Muse-Glimmer-Vision": ("load_projector",),
    "PixtralViT-Large-4": ("load_projector", "merged_and_projected"),
}


def load_example(path):
    spec = importlib.util.spec_from_file_location("hf_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bundle", ["MiMoViT-2.6-Pro", "MiMoViT-2.6-Flash"])
def test_mimo_copies_match_vtb(bundle):
    assert (HF / bundle / "modeling_mimo_vision.py").read_bytes() == (ROOT / "vtb/models/mimo26.py").read_bytes()


def test_rendered_templates_are_checked_in():
    rendered = render_hf_templates.render()
    assert rendered
    for path, text in rendered.items():
        assert path.read_text() == text, f"run scripts/render_hf_templates.py to refresh {path.relative_to(ROOT)}"


def test_pixel_shuffle_copies_agree():
    copies = {
        "vtb": nemotron_omni.pixel_shuffle,
        "omni_example": load_example(HF / "C-RADIOv4-H-3-Omni/examples/inference.py").pixel_shuffle,
        **{name: runpy.run_path(str(HF / name / "vision.py"))["pixel_shuffle"] for name in ("C-RADIOv4-H-3.5-Bolt", "C-RADIOv4-H-3.5-Super")},
    }
    tokens = torch.randn(2, 6, 4, 8)
    expected = copies.pop("vtb")(tokens)
    assert expected.shape == (2, 3, 2, 32)
    for name, shuffle in copies.items():
        assert torch.equal(shuffle(tokens), expected), name


def test_examples_do_not_touch_sys_path():
    for path in HF.glob("*/examples/*.py"):
        assert "sys.path" not in path.read_text(), path


@pytest.mark.parametrize("bundle,names", SIBLING_EXAMPLES.items())
def test_examples_load_siblings_from_a_snapshot(bundle, names, tmp_path, monkeypatch):
    snapshot = tmp_path / bundle
    shutil.copytree(HF / bundle, snapshot, ignore=shutil.ignore_patterns("*.safetensors", "__pycache__"))
    monkeypatch.chdir(tmp_path)
    module = load_example(snapshot / "examples/inference.py")
    assert module.BUNDLE == snapshot
    assert all(callable(getattr(module, name)) for name in names)
