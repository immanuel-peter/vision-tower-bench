"""Check that result navigation resolves to the intended experiment records."""

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
    assert set(sources["geometry_stage_levels"]) == v1 | additions
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
