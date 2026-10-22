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


def test_roster_covers_the_eleven_tower_suite():
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
    )
    assert set(FIGURE_SCRIPT.COLOUR) == set(FIGURE_SCRIPT.TOWERS)
    assert set(FIGURE_SCRIPT.DISPLAY) == set(FIGURE_SCRIPT.TOWERS)
    assert FIGURE_SCRIPT.DISPLAY["glm5"] == "GLM 5.3 Flash"
    assert FIGURE_SCRIPT.DISPLAY["nemotron_omni"] == "Nemotron Omni"
    assert FIGURE_SCRIPT.DISPLAY["deepseek_v41"] == "DeepSeek V4.1"
    assert FIGURE_SCRIPT.DISPLAY["minimax_m3"] == "MiniMax M3"


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

    assert set(FIGURE_SCRIPT.PANELS) <= set(panels)
    for panel in FIGURE_SCRIPT.PANELS:
        assert {row[0] for row in panels[panel]} == set(FIGURE_SCRIPT.PROJECTORS)

    states = []
    for panel in ("depth-matched", "depth-raw", "normal-matched", "normal-raw"):
        for _, _, low, high in panels[panel]:
            states.append("projector" if low > 0 else "tower" if high < 0 else "unresolved")
    assert states.count("projector") == 25
    assert states.count("tower") == 7
    assert states.count("unresolved") == 4


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
    stage_lines = figures[0].axes[0].lines
    assert len(stage_lines) == len(FIGURE_SCRIPT.PROJECTORS)
    for model, line in zip(FIGURE_SCRIPT.PROJECTORS, stage_lines, strict=True):
        directory = "geometry/deterministic" if model in corrected else "geometry"
        path = ROOT / "results" / directory / f"{model}_geometry_depth_matched.json"
        cells = json.loads(path.read_text())["cells"]
        deepest = {cell["stage"]: cell["d1"] for cell in cells if cell["relative_depth"] == 1.0}
        assert list(line.get_ydata()) == pytest.approx([
            deepest[stage] for stage in ("tower", "merged", "projected") if stage in deepest
        ])

    sweep_lines = {line.get_label(): line for line in figures[1].axes[1].lines}
    for model in FIGURE_SCRIPT.TOWERS:
        path = ROOT / "results/geometry" / f"{model}_geometry_depth_matched.json"
        cells = sorted(
            (cell for cell in json.loads(path.read_text())["cells"] if cell["stage"] == "tower"),
            key=lambda cell: cell["relative_depth"],
        )
        assert len(cells) == 8
        assert list(sweep_lines[model].get_ydata()) == pytest.approx([cell["d1"] for cell in cells])


def test_semantic_forest_leaves_out_the_v3_cohort(tmp_path):
    path = tmp_path / "forest.svg"
    FIGURE_SCRIPT.semantic_forest(path)
    labels = path.read_text()

    assert "siglip2 - dinov2" in labels
    for model in FIGURE_SCRIPT.LATER_COHORT:
        assert model not in labels


# ---------------------------------------------------------------- counting (v3) opt-in targets
def test_counting_targets_stay_out_of_the_default_figures():
    assert not set(FIGURE_SCRIPT.COUNT_TARGETS) & set(FIGURE_SCRIPT.FIGURES)
    assert not any(name.startswith("count") for name in FIGURE_SCRIPT.FIGURES)


def _count_fixture(root, ids, side=4, dim=6, stages=("tower", "merged", "projected"), arms=("free",)):
    """A synthetic feature cache and saved count readouts in the runner's layout (CPU only)."""
    import torch

    from vtb import counting, geometry, geometry_run
    from vtb.feature_batch import FeatureBatch
    from vtb.probe import Reducer

    torch.manual_seed(0)
    features = root / "features"
    features.mkdir()
    for stage in stages:
        FeatureBatch(tokens=torch.randn(len(ids), side * side, dim).to(torch.bfloat16), image_ids=ids[::-1],
                     model_id="toy/model", stage=stage, layer_index=3, num_layers=3, resolution=32).save(
            features / f"{stage}_L03_00000.safetensors")
    reducer = Reducer(basis=torch.randn(dim, 4), mean=torch.zeros(1, dim))
    for arm in arms:
        for stage in stages if arm == "free" else ("tower",):
            width = 4 + int(arm == "exemplar")
            meta = {"protocol": counting.PROTOCOL, "arm": arm, "input_dim": width, "head": "linear",
                    "hidden_dim": 8, "grid": counting.TARGET_GRID, "density_scale": counting.DENSITY_SCALE,
                    "grid_extent": 1.0, "model_id": "toy/model", "stage": stage, "layer": 3, "resolution": 32,
                    "tokens": side * side, "identity": "toy"}
            for seed in (0, 1):
                torch.manual_seed(seed)
                head = geometry.CountHead([width], head="linear", hidden_dim=8)
                geometry_run.save_readout(root / "readouts" / "toy" / f"{arm}_matched" / f"{stage}_L03", seed,
                                          head, reducer, meta)
    return features, root / "readouts"


def test_fsc_density_panel_from_saved_readouts_on_cpu(tmp_path):
    import numpy as np

    from vtb import cache, counting, geometry_run

    ids = [f"fsc147_{k}" for k in range(3)]
    features, readouts = _count_fixture(tmp_path, ids, arms=("free", "exemplar"))
    prep = tmp_path / "prep"
    prep.mkdir()
    images = {i: {"split": "test", "count": 3 + k, "boxes": [[0.1, 0.1, 0.3, 0.3]], "box_valid": [True]}
              for k, i in enumerate(ids)}
    (prep / "fsc147_manifest.json").write_text(json.dumps({"images": images}))
    rng = np.random.default_rng(0)
    np.savez(prep / "fsc147_targets.npz",
             **{i: counting.density_map(rng.random((3 + k, 2)) * 0.9) for k, i in enumerate(ids)})

    data = FIGURE_SCRIPT.fsc_density_data("toy", features, readouts, prep, ids, seeds=(0, 1), device="cpu")
    assert data["stages"] == ["tower", "merged", "projected"]
    assert data["truth"].shape == (3, 64, 64) and data["density"]["merged"].shape == (3, 64, 64)
    assert data["truth"].sum(axis=(1, 2)) == pytest.approx([3, 4, 5], abs=1e-3)
    # Seed-mean density, read through the same load_readout path the runner saves for.
    tokens, order, _ = cache.load(features, "projected", 3)
    rows = [order.index(i) for i in ids]
    directory = readouts / "toy" / "free_matched" / "projected_L03"
    expected = sum(geometry_run.load_readout(directory, s, "cpu").density(tokens[rows])[:, 0] for s in (0, 1)) / 2
    assert data["density"]["projected"] == pytest.approx(expected.numpy(), rel=1e-5, abs=1e-7)
    assert data["counts"]["projected"] == pytest.approx(expected.flatten(1).sum(1).double().numpy(), rel=1e-5)
    out = tmp_path / "fsc.png"
    FIGURE_SCRIPT.fsc_density_panel(out, data)
    assert out.stat().st_size > 0
    with pytest.raises(ValueError, match="not FSC-147 test"):
        FIGURE_SCRIPT.fsc_density_data("toy", features, readouts, prep, ["fsc147_99"], device="cpu")
    # The conditioned Tower reads the release boxes from the manifest.
    conditioned = FIGURE_SCRIPT.fsc_density_data("toy", features, readouts, prep, ids, arm="exemplar",
                                                 seeds=(0,), device="cpu")
    assert conditioned["stages"] == ["tower"] and conditioned["density"]["tower"].shape == (3, 64, 64)
    # A readout trained on another cache (here: the Tower slice) is refused.
    directory = readouts / "toy" / "free_matched" / "merged_L03"
    wrong = geometry_run.load_readout(directory, 0, "cpu")
    geometry_run.save_readout(directory, 0, wrong.head, wrong.reducer, wrong.meta | {"stage": "tower"})
    with pytest.raises(ValueError, match="does not match cache"):
        FIGURE_SCRIPT.fsc_density_data("toy", features, readouts, prep, ids, seeds=(0,), device="cpu")


def test_fsc_selection_writes_rule_and_ids(tmp_path):
    from tests.test_count_bootstrap import make_truth, write_result

    ids, true = make_truth()
    write_result(tmp_path, "toy", "free", "matched", ids, true, ("tower", "merged", "projected"))
    out = tmp_path / "qualitative_ids.json"
    chosen = FIGURE_SCRIPT.select_fsc_qualitative(tmp_path, out, k=3)
    saved = json.loads(out.read_text())
    assert saved == chosen and saved["status"] == "selected" and len(saved["image_ids"]) == 3
    assert "median" in saved["rule"] and saved["sources"]
    committed = json.loads(FIGURE_SCRIPT.QUALITATIVE_IDS.read_text())
    assert committed["rule"] == FIGURE_SCRIPT._counting_modules()[1].select_dense([], k=committed["k"])["rule"]


def test_countqa_panel_selects_successes_and_failures_and_attributes(tmp_path):
    import numpy as np
    from PIL import Image

    from tests.test_count_bootstrap import write_countqa
    from vtb import counting

    ids = [f"countqa_{k:04d}_abcdef01" for k in range(6)]
    true = np.array([3.0, 5.0, 8.0, 2.0, 12.0, 7.0])
    features, readouts = _count_fixture(tmp_path, ids, stages=("tower",), arms=("free",))
    data_dir = tmp_path / "countqa"
    (data_dir / "images").mkdir(parents=True)
    images = {}
    for k, image_id in enumerate(ids):
        width, height = 40 + 4 * k, 30
        frame = counting.letterbox_frame(width, height)
        Image.new("RGB", (frame.side, frame.side), (124, 116, 104)).save(data_dir / "images" / f"{image_id}.png")
        images[image_id] = {"width": width, "height": height, "frame": frame.as_dict(), "answer": int(true[k]),
                            "png": f"images/{image_id}.png", "question_id": f"{image_id}_q0"}
    manifest = data_dir / "manifest.json"
    manifest.write_text(json.dumps({"images": images, "eligibility_sha256": "e", "source": {
        "repo": "Jayant-Sravan/CountQA", "revision": "f92cc6fe46542c61e2916e3d2ae9a911e2216b1a",
        "citation": "Tamarapalli et al. CountQA (2025). CC-BY-4.0."}}))
    result = write_countqa(tmp_path, "toy", "free", "matched", ids, true, ("tower",))

    chosen = FIGURE_SCRIPT.countqa_selection(result, k=2)
    assert [label for _, label in chosen] == ["success", "success", "failure", "failure"]
    assert len({i for i, _ in chosen}) == 4
    data = FIGURE_SCRIPT.countqa_panel_data("toy", features, readouts, manifest, result, arm="free",
                                            k=2, seeds=(0, 1), device="cpu", image_side=32)
    assert data["boxes"] is None and data["density"].shape == (4, 64, 64)
    assert data["images"][0].shape == (32, 32, 3)
    # Padding never counts: the predicted count only sums the valid area.
    k = ids.index(data["image_ids"][0])
    valid = counting.coverage(counting.letterbox_frame(40 + 4 * k, 30).valid, 64)
    assert data["predicted"][0] == pytest.approx((data["density"][0] * valid).sum(), rel=1e-4)
    out = tmp_path / "countqa.svg"
    FIGURE_SCRIPT.countqa_panel(out, data)
    text = out.read_text()
    assert "CC-BY-4.0" in text and "error-selected" in text
