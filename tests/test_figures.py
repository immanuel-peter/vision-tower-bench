"""Keep the report figures aligned with the committed result suite."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("vtb_figures", ROOT / "scripts/figures.py")
assert SPEC and SPEC.loader
FIGURE_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURE_SCRIPT)


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
        "minimax_m3",
        "nemotron35_super",
        "nemotron35_bolt",
        "dinov3",
        "mimo26_pro",
        "mimo26_flash",
        "qwen38_flash_next",
    )
    assert set(FIGURE_SCRIPT.COLOUR) == set(FIGURE_SCRIPT.TOWERS)
    assert set(FIGURE_SCRIPT.DISPLAY) == set(FIGURE_SCRIPT.TOWERS)
    assert FIGURE_SCRIPT.DISPLAY["glm5"] == "GLM 5.3 Flash"
    assert FIGURE_SCRIPT.DISPLAY["nemotron_omni"] == "Nemotron 3 Omni"
    assert FIGURE_SCRIPT.DISPLAY["deepseek_v41"] == "DeepSeek V4.1"
    assert FIGURE_SCRIPT.DISPLAY["minimax_m3"] == "MiniMax M3"
    assert len(set(FIGURE_SCRIPT.COLOUR.values())) == len(FIGURE_SCRIPT.TOWERS)
    assert "dinov3" not in FIGURE_SCRIPT.PROJECTORS
    assert "nemotron35_bolt" not in FIGURE_SCRIPT.TOWER_STAGE
    assert not FIGURE_SCRIPT.OUTLIER_CHANNEL & set(FIGURE_SCRIPT.OCCLUSION)
    assert "qwen38_flash_next" not in FIGURE_SCRIPT.OCCLUSION


def test_occlusion_filter_excludes_scale_and_motion_blur():
    payload = json.loads(
        (ROOT / "results/perturbation/minimax_m3_perturbation.json").read_text()
    )
    cell = next(cell for cell in payload["cells"] if cell["stage"] == "tower")
    conditions = FIGURE_SCRIPT.occlusion_conditions(cell)

    assert {condition["factor"] for condition in conditions} == {"identity", "occlusion"}
    assert [condition["value"] for condition in conditions] == [0, 0.1, 0.2, 0.35, 0.5]


def test_projector_matrix_covers_every_projector_and_panel():
    panels = FIGURE_SCRIPT.projector_intervals()

    rows = set(FIGURE_SCRIPT.PROJECTORS) - FIGURE_SCRIPT.NO_TOWER_STAGE
    assert set(FIGURE_SCRIPT.PANELS) <= set(panels)
    for panel in FIGURE_SCRIPT.PANELS:
        assert {row[0] for row in panels[panel]} == rows

    states = []
    for panel in ("depth-matched", "depth-raw", "normal-matched", "normal-raw"):
        for model, _, low, high in panels[panel]:
            if model == "qwen38_flash_next":
                continue
            states.append("projector" if low > 0 else "tower" if high < 0 else "unresolved")
    assert states.count("projector") == 30
    assert states.count("tower") == 13
    assert states.count("unresolved") == 5


def test_default_figures_render(tmp_path):
    assert "projector-matrix" in FIGURE_SCRIPT.FIGURES
    assert "projector-forest" not in FIGURE_SCRIPT.FIGURES

    for name, draw in FIGURE_SCRIPT.FIGURES.items():
        path = tmp_path / f"{name}.png"
        draw(path)
        assert path.stat().st_size > 0


def test_geometry_figures_separate_corrected_stages_from_historical_sweeps(monkeypatch, tmp_path):
    figures = []
    subplots = FIGURE_SCRIPT.plt.subplots

    def capture(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(FIGURE_SCRIPT.plt, "subplots", capture)
    FIGURE_SCRIPT.stage_levels(tmp_path / "stages.png")
    FIGURE_SCRIPT.peaks(tmp_path / "peaks.png")

    corrected = {"muse_glimmer", "kimi_k26", "qwen3_5", "moonvit_v2"}
    stages = ("tower", "merged", "projected")
    stage_lines = figures[0].axes[0].lines
    assert len(stage_lines) == len(FIGURE_SCRIPT.PROJECTORS)
    for model, line in zip(FIGURE_SCRIPT.PROJECTORS, stage_lines, strict=True):
        directory = "geometry/deterministic" if model in corrected else "geometry"
        path = ROOT / "results" / directory / f"{model}_geometry_depth_matched.json"
        cells = json.loads(path.read_text())["cells"]
        deepest = {cell["stage"]: cell["d1"] for cell in cells if cell["relative_depth"] == 1.0}
        assert list(line.get_xdata()) == [stages.index(stage) for stage in stages if stage in deepest]
        assert list(line.get_ydata()) == pytest.approx([deepest[stage] for stage in stages if stage in deepest])

    sweep_lines = {line.get_label(): line for line in figures[1].axes[0].lines}
    for model in FIGURE_SCRIPT.TOWER_STAGE:
        path = ROOT / "results/geometry" / f"{model}_geometry_depth_matched.json"
        cells = sorted(
            (cell for cell in json.loads(path.read_text())["cells"] if cell["stage"] == "tower"),
            key=lambda cell: cell["relative_depth"],
        )
        assert len(cells) == 8
        assert list(sweep_lines[FIGURE_SCRIPT.DISPLAY[model]].get_ydata()) == pytest.approx(
            [cell["d1"] for cell in cells]
        )


def test_semantic_forest_includes_the_v3_comparisons(tmp_path):
    path = tmp_path / "forest.svg"
    FIGURE_SCRIPT.semantic_forest(path)
    labels = path.read_text()

    assert "siglip2 - dinov2" in labels
    assert "dinov3 - dinov2" in labels
    for stage in ("tower", "merged", "projected"):
        assert f"attention, matched, {stage}" in labels


def test_label_budget_reads_the_standardized_runs_for_every_tower():
    series = FIGURE_SCRIPT.label_budget_series()

    assert set(series) == set(FIGURE_SCRIPT.TOWERS)
    for model in FIGURE_SCRIPT.TOWERS:
        assert sorted(series[model]) == [0.01, 0.05, 0.2, 1.0]


def test_qwen_sibling_figure_uses_paired_points_and_spair_macro_intervals(monkeypatch, tmp_path):
    figures = []
    subplots = FIGURE_SCRIPT.plt.subplots

    def capture(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(FIGURE_SCRIPT.plt, "subplots", capture)
    FIGURE_SCRIPT.qwen_sibling_intervals(tmp_path / "siblings.svg")
    panels = [
        ("semantic-qwen38_flash_next-vs-qwen3_5-{}", "difference", 100),
        ("depth-qwen38_flash_next-sibling-{}", "first_advantage", 100),
        ("normal-qwen38_flash_next-sibling-{}", "first_advantage", 1),
        ("kitti-qwen38_flash_next-sibling-{}", "first_advantage", 100),
        ("correspondence-navi-flash-next-vs-27b-{}", "difference", 100),
        ("correspondence-scannet-flash-next-vs-27b-{}", "difference", 100),
        ("correspondence-spair-flash-next-vs-27b-{}", "class_macro_difference", 100),
    ]
    for axis, (pattern, key, scale) in zip(figures[0].axes, panels):
        stages = ["tower", "projected"] if pattern.startswith("kitti") else ["tower", "merged", "projected"]
        for index, stage in enumerate(stages):
            payload = json.loads((ROOT / "results/bootstrap" / f"{pattern.format(stage)}.json").read_text())
            interval = payload[key]
            assert list(axis.lines[2 * index].get_xdata()) == pytest.approx(
                [interval["lower"] * scale, interval["upper"] * scale]
            )
            assert list(axis.lines[2 * index + 1].get_xdata()) == pytest.approx(
                [interval["point_estimate"] * scale]
            )
