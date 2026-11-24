"""Keep the report figures aligned with the result suite."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vtb_figures", ROOT / "scripts/figures.py"
)
assert SPEC and SPEC.loader
FIGURE_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURE_SCRIPT)
requires_payloads = pytest.mark.skipif(
    not (ROOT / "results/bootstrap").is_dir(),
    reason="needs the private results archive in results/",
)


@pytest.mark.parametrize("suffix", ["", "_448_pool4"])
def test_tower_name_maps_every_label_to_its_own_tower(suffix):
    for tower in FIGURE_SCRIPT.TOWERS:
        assert FIGURE_SCRIPT.tower_name(tower + suffix) == tower


def test_tower_name_rejects_unknown_labels():
    with pytest.raises(ValueError):
        FIGURE_SCRIPT.tower_name("not_a_registered_tower")


def test_projector_intervals_read_only_geometry_and_correspondence(
    tmp_path, monkeypatch
):
    overrides = FIGURE_SCRIPT.SOURCES.get("model_overrides", {})
    tower = next(
        name
        for name in FIGURE_SCRIPT.PROJECTORS
        if "projector_intervals" not in overrides.get(name, {})
    )
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    advantage = {"first_advantage": {"point_estimate": 0.1, "lower": 0.0, "upper": 0.2}}
    for axis in ("geometry-depth-matched", "correspondence-navi"):
        (bootstrap / f"{axis}-{tower}-projected-vs-tower.json").write_text(
            json.dumps(advantage)
        )
    for axis in ("recognition-mean-raw", "perturbation"):
        (bootstrap / f"{axis}-{tower}-projected-vs-tower.json").write_text(
            json.dumps({"clean": {}, "conditions": {}})
        )
    monkeypatch.setattr(FIGURE_SCRIPT, "RESULTS", tmp_path)

    assert set(FIGURE_SCRIPT.projector_intervals()) == {"depth-matched", "navi"}


def test_label_budget_series_overlays_standardized_directory(tmp_path, monkeypatch):
    bundle = tmp_path / "diagnostics/mimo26-tower-outlier-channel"
    bundle.mkdir(parents=True)
    (bundle / "label_budget_standardized.json").write_text(
        json.dumps(
            {
                "runs": {
                    "deepseek_v41_probe_attention_matched_f001": {
                        "label_fraction": 0.01,
                        "cells": [{"test_accuracy": 0.3687}],
                    }
                }
            }
        )
    )
    extra = tmp_path / "diagnostics/minimax-m31-flash"
    extra.mkdir(parents=True)
    (extra / "label_budget_standardized.json").write_text(json.dumps({"runs": {}}))
    override = extra / "m3-image-isolation"
    override.mkdir(parents=True)
    (override / "label_budget_standardized.json").write_text(json.dumps({"runs": {}}))

    std_dir = tmp_path / "diagnostics/standardized-label-budget"
    std_dir.mkdir(parents=True)
    (std_dir / "deepseek_v41_probe_attention_matched_f001.json").write_text(
        json.dumps({"label_fraction": 0.01, "cells": [{"test_accuracy": 0.3786}]})
    )
    (std_dir / "hy4_probe_attention_matched_f001.json").write_text(
        json.dumps({"label_fraction": 0.01, "cells": [{"test_accuracy": 0.4178}]})
    )
    monkeypatch.setattr(FIGURE_SCRIPT, "RESULTS", tmp_path)

    series = FIGURE_SCRIPT.label_budget_series()
    assert series["deepseek_v41"][0.01] == pytest.approx(0.3786)
    assert series["hy4"][0.01] == pytest.approx(0.4178)


def test_roster_covers_the_registered_tower_suite():
    assert FIGURE_SCRIPT.TOWERS == (
        "dinov2",
        "siglip2",
        "muse_glimmer",
        "glm5",
        "nemotron_omni",
        "kimi_k26",
        "qwen3_5",
        "deepseek_v41",
        "gemma4",
        "moonvit_v2",
        "kimi_k31_pro",
        "minimax_m3",
        "minimax_m31_flash",
        "nemotron35_super",
        "nemotron35_bolt",
        "dinov3",
        "mimo26_pro",
        "mimo26_flash",
        "qwen38_flash_next",
        "mistral_large4",
        "pe_lang_l",
        "inkling2_small",
        "inkling2_mini",
        "hy4_flash",
        "hy4",
    )
    assert set(FIGURE_SCRIPT.COLOUR) == set(FIGURE_SCRIPT.TOWERS)
    assert set(FIGURE_SCRIPT.DISPLAY) == set(FIGURE_SCRIPT.TOWERS)
    assert FIGURE_SCRIPT.DISPLAY["glm5"] == "GLM 5.3 Flash"
    assert FIGURE_SCRIPT.DISPLAY["nemotron_omni"] == "Nemotron 3 Omni"
    assert FIGURE_SCRIPT.DISPLAY["mistral_large4"] == "Mistral Large 4"
    assert FIGURE_SCRIPT.DISPLAY["deepseek_v41"] == "DeepSeek V4.1"
    assert FIGURE_SCRIPT.DISPLAY["minimax_m3"] == "MiniMax M3"
    assert len(set(FIGURE_SCRIPT.COLOUR.values())) == len(FIGURE_SCRIPT.TOWERS)
    assert (
        "dinov3" not in FIGURE_SCRIPT.PROJECTORS
        and "pe_lang_l" not in FIGURE_SCRIPT.PROJECTORS
    )
    assert set(FIGURE_SCRIPT.PROJECTORS) == set(
        FIGURE_SCRIPT.SOURCES["geometry_stage_levels"]
    ) - {"dinov2", "siglip2"}
    assert "nemotron35_bolt" not in FIGURE_SCRIPT.TOWER_STAGE
    assert not FIGURE_SCRIPT.OUTLIER_CHANNEL & set(FIGURE_SCRIPT.OCCLUSION)
    assert "qwen38_flash_next" not in FIGURE_SCRIPT.OCCLUSION


@requires_payloads
def test_default_figures_render(tmp_path):
    assert "projector-matrix" in FIGURE_SCRIPT.FIGURES
    assert "projector-forest" not in FIGURE_SCRIPT.FIGURES

    for name, draw in FIGURE_SCRIPT.FIGURES.items():
        path = tmp_path / f"{name}.png"
        draw(path)
        assert path.stat().st_size > 0
