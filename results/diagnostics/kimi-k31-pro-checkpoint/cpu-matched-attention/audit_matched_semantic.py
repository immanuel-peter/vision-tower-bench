"""Independently audit full matched matrices, thread matching and exact CI replay."""

import hashlib
import json
from pathlib import Path

import torch

from scripts.validate_kimi_results import validate_probe
from vtb.semantic_bootstrap import paired_image_bootstrap

root = Path("/tmp/vtb-delivery/issues/24")
ids = json.loads((root / "ids-imagenet100.json").read_text())["ids"]
splits = json.loads((root / "splits-imagenet100.json").read_text())
phase = json.loads((root / "cpu-matched-recognition-progress.json").read_text())
assert phase["complete"]
matrix = {}
reports = []
for model, model_id in (("kimi_k31_pro", "immanuelpeter/MoonViT-K3.1-Pro"), ("moonvit_v2", "immanuelpeter/MoonViT-V2")):
    path = root / "measurements/recognition" / f"{model}_probe_attention_matched.json"
    payload = json.loads(path.read_text())
    report = validate_probe(payload, model_id, ids, "recognition", "matched", "attention", expected_splits=splits)
    matrix[model] = {(row["stage"], row["layer_index"]): row for row in payload["cells"]}
    for cell in payload["cells"]:
        source = Path(cell["source_fit"]["file"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == cell["source_fit"]["sha256"]
        assert cell["fit_threads"] == cell["source_fit"]["threads"] in (4, 8)
    report.update(model=model, file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  actual_thread_counts=payload["execution_provenance"]["per_cell_threads"])
    reports.append(report)
assert matrix["kimi_k31_pro"].keys() == matrix["moonvit_v2"].keys()
for key in matrix["kimi_k31_pro"]:
    assert matrix["kimi_k31_pro"][key]["fit_threads"] == matrix["moonvit_v2"][key]["fit_threads"]
book = json.loads((root / "completed-inventory.json").read_text())
replays = []
for record in book["completed_paired"]:
    if record["cell"]["task"] != "recognition":
        continue
    path = Path(record["source"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    payload = json.loads(path.read_text())
    rows = payload["predictions"]
    assert [row["image_id"] for row in rows] == splits["test"]
    labels = torch.tensor([row["label"] for row in rows])
    names = [payload[key]["name"] for key in ("first", "second")]
    correctness = [torch.tensor([row[name] for row in rows]).T == labels for name in names]
    threads = payload["protocol"].get("bootstrap_threads", payload["protocol"]["threads"])
    torch.set_num_threads(threads)
    interval = paired_image_bootstrap(*correctness)
    assert interval == payload["difference"], f"bootstrap replay mismatch: {path}"
    first, second = record["cell"]["first_stage"], record["cell"]["second_stage"]
    first_model = "kimi_k31_pro"
    second_model = "moonvit_v2" if record["cell"]["comparison"] == "kimi_k31_pro-minus-moonvit_v2" else "kimi_k31_pro"
    assert matrix[first_model][first, record["cell"]["first_layer"]]["fit_threads"] == matrix[second_model][second, record["cell"]["second_layer"]]["fit_threads"]
    replays.append(dict(file=str(path), exact_replay=True, resamples=10000, bootstrap_seed=0, bootstrap_threads=threads, paired_test_images=1950))
assert len(replays) == 11
report = dict(passed=True, complete_matched_cells=20, source_full_budget_cells_reused=2,
              phase_new_fits=sum(not row["resumed"] for row in phase["results"]),
              phase_verified_reuses=sum(row["resumed"] for row in phase["results"]),
              total_remaining_cells_fitted_across_cpu4_cpu8_phases=18,
              sibling_thread_matching=True, matrices=reports, exact_semantic_replays=replays)
(root / "matched-semantic-complete-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key not in ("matrices", "exact_semantic_replays")}, indent=2))
