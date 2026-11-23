import importlib.util
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModel,
    AutoProcessor,
    MuseGlimmerConfig,
    MuseGlimmerTextConfig,
    MuseGlimmerVisionConfig,
    MuseGlimmerVisionModel,
    Qwen3_5VisionConfig,
    Qwen3_5VisionModel,
)

from scripts import render_hf_templates
from vtb.adapters import hy4, hy4_flash, minimax_m3, nemotron_omni
from vtb.adapters.kimi_k26 import BUNDLE_FILES as KIMI_BUNDLE_FILES
from vtb.adapters.moonvit_v2 import load_projector_weights
from vtb.adapters.muse_glimmer import Projector
from vtb.adapters.muse_glimmer import projector_config as muse_projector_config
from vtb.bundle import write_bundle
from vtb.bundle_manifest import write_manifest

ROOT = Path(__file__).parents[1]
HF = ROOT / "hf"
SIBLING_EXAMPLES = {
    "C-RADIOv4-H-3-Omni": ("load_projector",),
    "C-RADIOv4-H-3.5-Bolt": ("load_vision",),
    "C-RADIOv4-H-3.5-Super": ("load_vision",),
    "DeepSeek-ViT": ("Aligner", "ViT"),
    "Gemma4-31B-Vision": ("load_projector",),
    "Hy4-Flash-Vision": ("preprocess", "features"),
    "Hy4-Vision": ("preprocess", "main"),
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


@pytest.mark.parametrize(
    "bundle,shipped,source",
    [
        ("MiMoViT-2.6-Pro", "modeling_mimo_vision.py", "mimo26.py"),
        ("MiMoViT-2.6-Flash", "modeling_mimo_vision.py", "mimo26.py"),
        ("PixtralViT-Large-4", "modeling_pixtral_large4.py", "pixtral_large4.py"),
        ("Inkling-2-Small-Vision", "inkling2_vision.py", "inkling2.py"),
        ("Inkling-2-Mini-Vision", "inkling2_vision.py", "inkling2.py"),
        ("Hy4-Flash-Vision", "modeling_hy4_flash_vision.py", "hy4_flash.py"),
        ("Hy4-Vision", "modeling_hy4.py", "hy4.py"),
    ],
)
def test_shipped_modeling_code_is_a_byte_copy_of_vtb(bundle, shipped, source):
    assert (HF / bundle / shipped).read_bytes() == (
        ROOT / "vtb/models" / source
    ).read_bytes()


def test_rendered_templates_are_checked_in():
    rendered = render_hf_templates.render()
    assert rendered
    for path, text in rendered.items():
        assert path.read_text() == text, (
            f"run scripts/render_hf_templates.py to refresh {path.relative_to(ROOT)}"
        )


def test_pixel_shuffle_copies_agree():
    copies = {
        "vtb": nemotron_omni.pixel_shuffle,
        "omni_example": load_example(
            HF / "C-RADIOv4-H-3-Omni/examples/inference.py"
        ).pixel_shuffle,
        **{
            name: runpy.run_path(str(HF / name / "vision.py"))["pixel_shuffle"]
            for name in ("C-RADIOv4-H-3.5-Bolt", "C-RADIOv4-H-3.5-Super")
        },
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
    shutil.copytree(
        HF / bundle,
        snapshot,
        ignore=shutil.ignore_patterns("*.safetensors", "__pycache__"),
    )
    monkeypatch.chdir(tmp_path)
    module = load_example(snapshot / "examples/inference.py")
    assert module.BUNDLE == snapshot
    assert all(callable(getattr(module, name)) for name in names)


def test_qwen_bundle_loads_through_auto_model_without_the_parent_repo(tmp_path):
    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=4,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=1,
    )
    tower = Qwen3_5VisionModel(config)
    preprocessor = tmp_path / "source-preprocessor.json"
    preprocessor.write_text(
        json.dumps(
            {
                "image_processor_type": "Qwen2VLImageProcessorFast",
                "processor_class": "Qwen3VLProcessor",
            }
        )
    )
    license_path = tmp_path / "source-license"
    license_path.write_text("Apache License\n")
    out = tmp_path / "bundle"

    write_bundle(
        out,
        tower=tower,
        architecture="Qwen3_5VisionModel",
        preprocessor=json.loads(preprocessor.read_text()),
        license=license_path,
    )

    loaded = AutoModel.from_pretrained(out, local_files_only=True)
    processor = AutoProcessor.from_pretrained(out, local_files_only=True)
    assert isinstance(loaded, Qwen3_5VisionModel)
    assert processor.__class__.__name__ == "Qwen2VLImageProcessor"
    assert loaded.config.model_type == "qwen3_5_vision"
    exported_processor = json.loads((out / "preprocessor_config.json").read_text())
    assert exported_processor["image_processor_type"] == "Qwen2VLImageProcessorFast"
    assert exported_processor["processor_class"] == "AutoImageProcessor"
    assert (out / "LICENSE").read_text() == "Apache License\n"


def test_muse_bundle_loads_the_tower_and_keeps_the_projector_separate(tmp_path):
    vision_config = MuseGlimmerVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        patch_size=2,
        patch_temporal=1,
        merge_size=2,
        pos_emb_height=2,
        pos_emb_width=2,
        max_position_embeddings=4,
        layer_types=["full_attention"],
    )
    config = MuseGlimmerConfig(
        vision_config=vision_config,
        text_config=MuseGlimmerTextConfig(
            hidden_size=24,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["full_attention"],
        ),
        out_hidden_size=64,
        projector_hidden_size=32,
    )
    tower = MuseGlimmerVisionModel(vision_config)
    projector = Projector(config)
    processor_config = tmp_path / "source-processor.json"
    processor_config.write_text(
        json.dumps(
            {
                "image_processor": {
                    "image_processor_type": "MuseGlimmerImageProcessor",
                    "patch_size": 2,
                    "merge_size": 2,
                }
            }
        )
    )
    license_path = tmp_path / "source-license"
    license_path.write_text("Apache License\n")
    usage_policy = tmp_path / "source-usage-policy"
    usage_policy.write_text("Usage policy\n")
    out = tmp_path / "bundle"

    write_bundle(
        out,
        tower=tower,
        architecture="MuseGlimmerVisionModel",
        weights={"projector.safetensors": projector.state_dict()},
        configs={"projector_config.json": muse_projector_config(projector)},
        preprocessor=json.loads(processor_config.read_text())["image_processor"],
        license=license_path,
        files={"USAGE_POLICY.md": usage_policy},
    )

    loaded = AutoModel.from_pretrained(out, local_files_only=True)
    assert isinstance(loaded, MuseGlimmerVisionModel)
    assert loaded.config.model_type == "muse_glimmer_vision"
    assert set(load_file(out / "projector.safetensors")) == {
        "adapter.fc1.weight",
        "adapter.fc2.weight",
        "projection.weight",
    }
    assert json.loads((out / "preprocessor_config.json").read_text())["patch_size"] == 2
    assert json.loads((out / "preprocessor_config.json").read_text())[
        "processor_class"
    ] == ("AutoImageProcessor")
    assert (out / "USAGE_POLICY.md").read_text() == "Usage policy\n"


KIMI_BUNDLE = Path(__file__).parents[1] / "hf/MoonViT-K2.6"


def test_kimi_bundle_declares_local_model_and_processor_code():
    assert set(KIMI_BUNDLE_FILES) <= {path.name for path in KIMI_BUNDLE.iterdir()}
    config = json.loads((KIMI_BUNDLE / "config.json").read_text())
    processor = json.loads((KIMI_BUNDLE / "preprocessor_config.json").read_text())
    assert config["auto_map"] == {
        "AutoConfig": "configuration_moonvit.MoonViTConfig",
        "AutoModel": "modeling_moonvit.MoonViTModel",
    }
    assert processor["auto_map"] == {
        "AutoImageProcessor": "kimi_k25_vision_processing.KimiK25VisionProcessor"
    }


@pytest.mark.skipif(
    not (KIMI_BUNDLE / "model.safetensors").is_file(),
    reason="run python -m vtb.export kimi_k26 first",
)
def test_kimi_bundle_loads_offline_without_the_parent_repositories():
    loaded = AutoModel.from_pretrained(
        KIMI_BUNDLE,
        dtype="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        KIMI_BUNDLE,
        local_files_only=True,
        trust_remote_code=True,
    )
    assert loaded.config.model_type == "moonvit_k26"
    assert processor.__class__.__name__ == "KimiK25VisionProcessor"


def test_manifest_counts_real_tensor_components(tmp_path):
    save_file(
        {
            "tower.weight": torch.ones(2, dtype=torch.bfloat16),
            "merger.weight": torch.zeros(3, dtype=torch.bfloat16),
        },
        tmp_path / "model.safetensors",
    )
    m = write_manifest(
        tmp_path, "parent", "pinned", ["shard-b", "shard-a"], merger_in_model=True
    )
    assert (m["tower_tensors"], m["projector_tensors"]) == (1, 1)
    assert m["source_shards"] == ["shard-a", "shard-b"]
    assert (m["source_repo"], m["source_revision"]) == ("parent", "pinned")


def test_maintained_bundle_manifests_record_counts_and_shards():
    root = Path(__file__).parents[1]
    expected = {
        "MiniMax-M3.1-Flash-Vision": (515, 8),
        "Qwen3.8-Flash-Next-Vision": (327, 6),
        "PixtralViT-Large-4": (272, 4),
        "MoonViT-K3.1-Pro": (165, 3),
        "Hy4-Flash-Vision": (435, 6),
        "Hy4-Vision": (643, 6),
        "DeepSeek-ViT": (259, 4),
    }
    checked = 0
    for name, counts in expected.items():
        bundle = root / "hf" / name
        if not bundle.is_dir():
            continue
        checked += 1
        m = json.loads((bundle / "export_manifest.json").read_text())
        assert (m["tower_tensors"], m["projector_tensors"]) == counts
        assert isinstance(m["source_shards"], list) and m["source_shards"]
        assert m["dtype"] in {"mixed", "bfloat16"}
    assert checked


MINIMAX_BUNDLE = HF / "MiniMax-M3.1-Flash-Vision"


def test_minimax_example_patch_packing_matches_the_benchmark():
    module = load_example(MINIMAX_BUNDLE / "examples/inference.py")
    pixels = np.arange(43 * 57 * 3, dtype=np.uint8).reshape(43, 57, 3)
    image = Image.fromarray(pixels)
    assert torch.equal(module.preprocess(image), minimax_m3.Preprocess(448)(image))


def test_example_keeps_snapshot_root_when_script_is_a_symlink(tmp_path):
    (tmp_path / "examples").mkdir()
    path = tmp_path / "examples/inference.py"
    path.symlink_to(MINIMAX_BUNDLE / "examples/inference.py")
    module = load_example(path)
    assert module.BUNDLE == tmp_path


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kimi_projector_file_matches_adapter_and_exact_gelu(tmp_path, dtype):
    config = dict(input_size=4, hidden_size=4, output_size=6, norm_eps=1e-5)
    weights = {
        "proj.0.weight": torch.eye(4, dtype=dtype),
        "proj.2.weight": torch.arange(24, dtype=dtype).reshape(6, 4) / 24,
        "post_norm.weight": torch.linspace(0.5, 1.5, 6, dtype=dtype),
    }
    save_file(weights, tmp_path / "projector.safetensors")
    example = (
        load_example(HF / "MoonViT-K3.1-Pro/projector.py")
        .Projector(config)
        .to(dtype)
        .eval()
    )
    example.load_state_dict(weights, strict=True)
    adapter = load_projector_weights(dtype, str(tmp_path))
    assert example.proj[1].approximate == adapter.proj[1].approximate == "none"
    assert set(example.state_dict()) == set(adapter.state_dict()) == set(weights)
    tokens = torch.tensor(
        [[[-2.0, -0.5, 0.5, 2.0], [-1.0, 0.0, 1.0, 3.0]]], dtype=dtype
    )
    with torch.inference_mode():
        expected = adapter.post_norm(adapter.proj(tokens))
        torch.testing.assert_close(example(tokens), expected, atol=0, rtol=0)


HY4_FLASH_BUNDLE = HF / "Hy4-Flash-Vision"


def test_hy4_flash_example_matches_the_benchmark_preprocess_and_stages():
    module = load_example(HY4_FLASH_BUNDLE / "examples/inference.py")
    pixels = np.arange(43 * 57 * 3, dtype=np.uint8).reshape(43, 57, 3)
    image = Image.fromarray(pixels)
    assert torch.equal(module.preprocess(image), hy4_flash.Preprocess(512)(image))
    config = hy4_flash.Hy4FlashVisionConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=8,
        num_attention_heads=2,
        max_image_size=64,
        out_hidden_size=6,
    )
    adapter = hy4_flash.Hy4FlashAdapter.__new__(hy4_flash.Hy4FlashAdapter)
    adapter.model = hy4_flash.Hy4FlashVisionModel(config).eval()
    adapter.num_layers, adapter.resolution, adapter.device = 8, 64, "cpu"
    adapter.dtype, adapter.model_id = torch.float32, hy4_flash.MODEL_ID
    pixels = torch.randn(1, 3, 64, 64)
    expected = {
        batch.stage: batch.tokens
        for batch in adapter.extract({"pixel_values": pixels}, ["a"])
    }
    with torch.inference_mode():
        actual = module.features(adapter.model, pixels)
    for stage in ("tower", "merged", "projected"):
        torch.testing.assert_close(actual[stage], expected[stage], rtol=0, atol=0)


@pytest.mark.parametrize(
    "bundle,modeling,model,revision",
    [
        (
            "Hy4-Flash-Vision",
            "modeling_hy4_flash_vision",
            "Hy4FlashVision",
            hy4_flash.SOURCE_REVISION,
        ),
        ("Hy4-Vision", "modeling_hy4", "Hy4Vision", hy4.SOURCE_REVISION),
    ],
)
def test_hy4_bundles_declare_their_remote_code_and_attribution(
    bundle, modeling, model, revision
):
    config = json.loads((HF / bundle / "config.json").read_text())
    assert config["auto_map"] == {
        "AutoConfig": f"{modeling}.{model}Config",
        "AutoModel": f"{modeling}.{model}Model",
    }
    assert (
        (HF / bundle / "LICENSE")
        .read_text()
        .lstrip()
        .startswith("Tencent Hy License, Version 1.0")
    )
    notice = (HF / bundle / "NOTICE").read_text()
    assert revision in notice and "Section 10" in notice
