"""Keep the report figures aligned with the result suite."""

import importlib.util
from pathlib import Path

import pytest

from vtb.roster import SLUGS

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


@pytest.mark.parametrize("model", ["minimax_m3", "minimax_m31_flash"])
@pytest.mark.parametrize("suffix", ["", "_448_pool4"])
def test_tower_name_distinguishes_minimax_siblings(model, suffix):
    assert FIGURE_SCRIPT.tower_name(model + suffix) == model


def test_roster_covers_the_registered_tower_suite():
    assert FIGURE_SCRIPT.TOWERS == SLUGS
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
    assert set(FIGURE_SCRIPT.SOURCES["geometry_stage_levels"]) <= (
        set(FIGURE_SCRIPT.PROJECTORS) | {"dinov2", "siglip2"}
    )
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
