"""Check that result navigation resolves to the intended experiment records."""

import hashlib
import json
from pathlib import Path


RESULTS = Path(__file__).parents[1] / "results"
INDEX = json.loads((RESULTS / "index.json").read_text())


def test_index_links_and_original_completion_records():
    for experiment in INDEX["experiments"].values():
        assert (RESULTS / experiment["path"]).is_dir()
        if "cohort" in experiment:
            assert experiment["cohort"] in INDEX["cohorts"]
    assert (RESULTS / INDEX["history"]).is_file()

    for manifest in INDEX["cell_manifests"]:
        for line in (RESULTS / manifest["path"]).read_text().splitlines():
            name, expected = line.split("\t")
            payload = json.loads((RESULTS / manifest["results_directory"] / name).read_text())
            assert len(payload["cells"]) == int(expected), name


def test_correction_scope_preserves_full_sweeps_and_cohort_membership():
    v1 = {"dinov2", "siglip2", "muse_glimmer", "kimi_k26", "qwen3_5", "moonvit_v2"}
    additions = {"glm5", "nemotron_omni", "deepseek_v41", "gemma4", "minimax_m3"}
    assert set(INDEX["cohorts"]["v1_six"]["models"]) == v1
    assert set(INDEX["cohorts"]["v2_eleven"]["models"]) == v1 | additions

    correction = INDEX["experiments"]["geometry_deterministic"]
    assert set(correction["supersedes"]["models"]) == v1
    assert correction["supersedes"]["relative_depth"] == 1.0
    for model in v1:
        for task in correction["supersedes"]["tasks"]:
            path = RESULTS / correction["path"] / f"{model}_geometry_{task}_matched.json"
            payload = json.loads(path.read_text())
            assert {cell["relative_depth"] for cell in payload["cells"]} == {1.0}

    sources = INDEX["figure_sources"]
    v3_projectors = {"nemotron35_super", "nemotron35_bolt", "mimo26_pro", "mimo26_flash"}
    assert set(sources["geometry_stage_levels"]) == v1 | additions | v3_projectors | {"qwen38_flash_next"}
    for model in v1 | additions:
        sweep = RESULTS / sources["geometry_layer_sweep"].format(model=model)
        cells = json.loads(sweep.read_text())["cells"]
        assert len({cell["relative_depth"] for cell in cells if cell["stage"] == "tower"}) == 8
        stage_path = RESULTS / sources["geometry_stage_levels"][model]
        expected = RESULTS / "geometry"
        if model in v1:
            expected /= "deterministic"
        assert stage_path == expected / f"{model}_geometry_depth_matched.json"
        assert stage_path.is_file()

        for readout in ("attention", "mean"):
            for arm in ("raw", "matched"):
                assert (RESULTS / sources["recognition"].format(
                    model=model, readout=readout, arm=arm,
                )).is_file()


def test_v3_cohort_uses_the_standard_layout():
    models = INDEX["cohorts"]["v3"]["models"]
    assert set(models) == {"nemotron35_super", "nemotron35_bolt", "dinov3", "mimo26_pro", "mimo26_flash"}
    stage_levels = INDEX["figure_sources"]["geometry_stage_levels"]
    assert set(models) & set(stage_levels) == set(models) - {"dinov3"}
    for model in set(models) - {"dinov3"}:
        assert stage_levels[model] == f"geometry/{model}_geometry_depth_matched.json"
    for model in models:
        paths = [f"perturbation/{model}_perturbation.json"]
        paths += [f"recognition/{model}_probe_{r}_{a}.json" for r in ("attention", "mean") for a in ("raw", "matched")]
        paths += [f"geometry/{model}_geometry_{t}_{a}.json" for t in ("depth", "normal") for a in ("raw", "matched")]
        paths += [f"correspondence/{model}_correspondence_{d}.json" for d in ("navi", "scannet", "spair")]
        paths += [
            f"label-budget/{model}_probe_{r}_matched_{f}.json"
            for r in ("attention", "mean") for f in ("f001", "f005", "f020", "f10")
        ]
        for path in paths:
            assert (RESULTS / path).is_file(), path
    assert not (RESULTS / "v3").exists()


def test_flash_next_geometry_contains_every_slice_with_native_grids_and_recorded_splits():
    evidence = json.loads((RESULTS / "evidence/qwen38-flash-next/splits.json").read_text())
    split = evidence["datasets"]["diode"]
    slices = {("tower", layer) for layer in [3, 7, 10, 14, 17, 20, 24, 27]}
    slices |= {("merged", 27), ("projected", 27)}
    widths = {"tower": 1152, "merged": 4608, "projected": 2560}
    for task in ["depth", "normal"]:
        for arm in ["raw", "matched"]:
            path = RESULTS / "geometry" / f"qwen38_flash_next_geometry_{task}_{arm}.json"
            payload = json.loads(path.read_text())
            json.dumps(payload, allow_nan=False)
            assert payload["images"] == len(split["all_ids"])
            assert payload["train_images"] == len(split["train_ids"])
            assert payload["epochs"] == 10 and payload["seeds"] == 3
            assert set(payload["learning_rates"]) == {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}
            assert len(payload["cells"]) == 10
            assert {(c["stage"], c["layer_index"]) for c in payload["cells"]} == slices
            for cell in payload["cells"]:
                assert cell["model_id"] == "immanuelpeter/Qwen3.8-Flash-Next-Vision"
                assert cell["grid"] == (28 if cell["stage"] == "tower" else 14)
                assert cell["token_width"] == (512 if arm == "matched" else widths[cell["stage"]])
                assert cell["images"] == len(split["test_ids"])
                assert len(cell["per_seed"]) == 3
                assert all(row["images"] == len(split["test_ids"]) for row in cell["per_seed"])


def test_dinov3_intervals_point_at_committed_sources():
    paths = sorted(RESULTS.glob("bootstrap/*-dinov3-vs-dinov2.json"))
    assert len(paths) == 7
    for path in paths:
        protocol = json.loads(path.read_text())["protocol"]
        for key in ("first_source", "second_source"):
            if key in protocol:
                assert (RESULTS.parent / protocol[key]).is_file(), (path.name, key)


def test_mimo26_intervals_point_at_committed_sources():
    paths = sorted(RESULTS.glob("bootstrap/*-mimo26-pro-vs-flash*.json"))
    assert len(paths) == 13
    for path in paths:
        for source, digest in json.loads(path.read_text()).get("sources", {}).items():
            assert hashlib.sha256((RESULTS.parent / source).read_bytes()).hexdigest() == digest, (path.name, source)
