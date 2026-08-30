import json
from pathlib import Path

import pytest
from safetensors.torch import load_file
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

from scripts.export_muse_glimmer_vision import write_bundle as write_muse_bundle
from scripts.export_moonvit_k26 import BUNDLE_FILES as KIMI_BUNDLE_FILES
from scripts.export_qwen3_8_vision import write_bundle as write_qwen_bundle
from vtb.adapters.muse_glimmer import Projector


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
    preprocessor.write_text(json.dumps({
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "processor_class": "Qwen3VLProcessor",
    }))
    license_path = tmp_path / "source-license"
    license_path.write_text("Apache License\n")
    out = tmp_path / "bundle"

    write_qwen_bundle(tower, preprocessor, license_path, out)

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
    processor_config.write_text(json.dumps({
        "image_processor": {
            "image_processor_type": "MuseGlimmerImageProcessor",
            "patch_size": 2,
            "merge_size": 2,
        }
    }))
    license_path = tmp_path / "source-license"
    license_path.write_text("Apache License\n")
    usage_policy = tmp_path / "source-usage-policy"
    usage_policy.write_text("Usage policy\n")
    out = tmp_path / "bundle"

    write_muse_bundle(tower, projector, processor_config, license_path, usage_policy, out)

    loaded = AutoModel.from_pretrained(out, local_files_only=True)
    assert isinstance(loaded, MuseGlimmerVisionModel)
    assert loaded.config.model_type == "muse_glimmer_vision"
    assert set(load_file(out / "projector.safetensors")) == {
        "adapter.fc1.weight",
        "adapter.fc2.weight",
        "projection.weight",
    }
    assert json.loads((out / "preprocessor_config.json").read_text())["patch_size"] == 2
    assert json.loads((out / "preprocessor_config.json").read_text())["processor_class"] == (
        "AutoImageProcessor"
    )
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
    reason="run scripts/export_moonvit_k26.py first",
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
