"""Audit attention budgets, append Pro figure inputs and reuse identical full-budget fits."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch

from scripts.validate_kimi_results import validate_probe
from vtb.semantic_bootstrap import paired_image_bootstrap

root = Path("/tmp/vtb-delivery/issues/24")
destination = Path("results/diagnostics/kimi-k31-pro-checkpoint")
phase = json.loads((root / "cpu-attention-budget-progress.json").read_text())
assert phase["complete"] and phase["cells"] == 16
ids = json.loads((root / "ids-imagenet100.json").read_text())["ids"]
full_splits = json.loads((root / "splits-imagenet100.json").read_text())
labels = json.loads(Path("/tmp/vtb-delivery/data/imagenet100/validation_labels.json").read_text())["labels"]
completed, full, pro_standardized = [], {}, {}
proofs = destination / "cpu-attention-budgets"
proofs.mkdir(exist_ok=True)
for row in phase["results"]:
    model, fraction, kind = row["model"], row["fraction"], row["kind"]
    private = root / "measurements" / ("diagnostics" if kind == "standardized" else "label-budget") / f"{model}_probe_attention_matched_f{round(fraction * 100):03d}.json"
    payload = json.loads(private.read_text())
    assert hashlib.sha256(private.read_bytes()).hexdigest() == row["payload_sha256"]
    splits = json.loads((root / f"splits-imagenet100-f{fraction}.json").read_text())
    model_id = "immanuelpeter/MoonViT-K3.1-Pro" if model == "kimi_k31_pro" else "immanuelpeter/MoonViT-V2"
    validate_probe(payload, model_id, ids, kind, "matched", "attention", fraction, splits)
    name = private.name.replace("_f100.json", "_f10.json")
    if kind == "label-budget":
        public = Path("results/label-budget") if model == "kimi_k31_pro" else destination / "sibling-control/label-budget"
    else:
        public = destination / "standardized-label-budget" if model == "kimi_k31_pro" else destination / "sibling-control/standardized-label-budget"
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(private, public / name)
    folder = proofs / kind
    folder.mkdir(exist_ok=True)
    shutil.copy2(private.with_suffix(".validation.json"), folder / private.with_suffix(".validation.json").name)
    shutil.copy2(private.with_suffix(".log"), folder / private.with_suffix(".txt").name)
    task = "standardized_label_budget" if kind == "standardized" else "label_budget"
    completed.append(dict(cell=dict(task=task, model=model, stage="tower", layer=27, arm="matched", readout="attention", fraction=fraction),
                          source=str(private), sha256=row["payload_sha256"]))
    if kind == "standardized" and model == "kimi_k31_pro":
        pro_standardized[Path(name).stem] = payload
    if kind == "label-budget" and fraction == 1.:
        assert splits == full_splits and payload["protocol"]["device"] == "cpu"
        cell = payload["cells"][0]
        assert cell["fit_threads"] == 4 and cell["test_image_ids"] == full_splits["test"]
        assert cell["test_labels"] == [labels[image] for image in cell["test_image_ids"]]
        scoped = copy.deepcopy(payload)
        scoped["scope"] = "one verified recognition cell; not a complete ten-cell matrix"
        scoped["reuse_provenance"] = dict(source=str(private), source_sha256=row["payload_sha256"],
            compatibility="Same full13000imagecache, exact9100/1950/1950split, full label fraction, train-only PCA seed0,11LRseed0search,3selected-rate retrainings,20epochs,CPU4; per-cell PCA and training seed resets make selection scope irrelevant.")
        path = root / "measurements/recognition-cells" / f"{model}_probe_attention_matched_tower_L27.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(scoped, indent=2) + "\n")
        public = destination / "recognition-cells"
        public.mkdir(exist_ok=True)
        shutil.copy2(path, public / path.name)
        completed.append(dict(cell=dict(task="recognition", model=model, stage="tower", layer=27, arm="matched", readout="attention"),
                              source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), reuse=scoped["reuse_provenance"]))
        full[model] = scoped
aggregate_path = Path("results/diagnostics/mimo26-tower-outlier-channel/label_budget_standardized.json")
aggregate = json.loads(aggregate_path.read_text())
old = copy.deepcopy(aggregate["runs"])
assert not set(old) & set(pro_standardized)
aggregate["runs"].update(pro_standardized)
assert all(aggregate["runs"][key] == value for key, value in old.items())
aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n")
first, second = full["kimi_k31_pro"], full["moonvit_v2"]
assert first["protocol"] == second["protocol"]
a, b = first["cells"][0], second["cells"][0]
assert a["test_image_ids"] == b["test_image_ids"] and a["test_labels"] == b["test_labels"]
pred_a, pred_b, target = torch.tensor(a["test_predictions_by_seed"]), torch.tensor(b["test_predictions_by_seed"]), torch.tensor(a["test_labels"])
correct_a, correct_b = pred_a == target, pred_b == target
interval = paired_image_bootstrap(correct_a, correct_b)
assert abs(interval["point_estimate"] - (correct_a.double().mean() - correct_b.double().mean()).item()) < 1e-6
plan = json.loads((root / "fullsuite-plan.json").read_text())
def option(argv, name):
    return argv[argv.index(name) + 1]
job = next(job for job in plan["jobs"] if "scripts.semantic_bootstrap" in job.get("argv", [])
           and option(job["argv"], "--first-name") == "kimi_k31_pro" and option(job["argv"], "--second-name") == "moonvit_v2"
           and option(job["argv"], "--first-stage") == "tower" and option(job["argv"], "--first-layer") == "27")
out = Path(option(job["argv"], "--out"))
def summary(cell, model):
    return dict(name=model, model_id=cell["model_id"], stage="tower", layer_index=27, relative_depth=1.,
                learning_rate=cell["learning_rate"], learning_rate_search=cell["learning_rate_search_unrounded"],
                validation_accuracy=cell["validation_accuracy_unrounded"], seed_accuracies=cell["test_accuracy_by_seed"],
                mean_accuracy=sum(cell["test_accuracy_by_seed"]) / 3)
paired = dict(protocol=dict(readout="attention", arm="matched", epochs=20, seeds=3, test_images=1950,
              pairing="seed-mean correctness, paired over test images", device="cpu", threads=4),
              first=summary(a, "kimi_k31_pro"), second=summary(b, "moonvit_v2"), difference=interval,
              predictions=[dict(image_id=image, label=int(target[index]), kimi_k31_pro=pred_a[:, index].tolist(),
                                moonvit_v2=pred_b[:, index].tolist()) for index, image in enumerate(a["test_image_ids"])],
              reuse_provenance=dict(first=first["reuse_provenance"], second=second["reuse_provenance"],
                                    compatibility="Same complete selected fits and actual1950testpredictions, pairedCPU4settings; no head refit or point substitution."))
out.write_text(json.dumps(paired, indent=2) + "\n")
shutil.copy2(out, Path("results/bootstrap") / out.name)
comparison = dict(comparison="kimi_k31_pro-minus-moonvit_v2", task="recognition", first_stage="tower", second_stage="tower", first_layer=27, second_layer=27, arm="matched", readout="attention")
book = json.loads((root / "completed-inventory.json").read_text())
book["completed"].extend(completed)
book["completed_paired"].append(dict(cell=comparison, source=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(), reused_same_fit_predictions=True))
expected = json.loads((root / "expected-inventory.json").read_text())
actual = {json.dumps(row["cell"], sort_keys=True) for row in book["completed"]}
paired_keys = {json.dumps(row["cell"], sort_keys=True) for row in book["completed_paired"]}
assert len(actual) == len(book["completed"]) == 144 and actual <= {json.dumps(row, sort_keys=True) for row in expected["cells"]}
assert len(paired_keys) == 53 and paired_keys <= {json.dumps(row, sort_keys=True) for row in expected["paired_comparison_cells"]}
book.update(completed_primary_cells=144, completed_paired_cells=53,
            pending=[row for row in expected["cells"] if json.dumps(row, sort_keys=True) not in actual])
(root / "completed-inventory.json").write_text(json.dumps(book, indent=2) + "\n")
(destination / "completed-inventory.json").write_text(json.dumps(book, indent=2) + "\n")
shutil.copy2(root / "cpu-attention-budget-progress.json", proofs / "validation-summary.json")
shutil.copy2(root / "cpu-attention-budget-pair.log", proofs / "execution-log.txt")
reuse = dict(passed=True, source_cells=2, recognition_cells_reused=2, paired_comparisons_reused=1,
             actual_paired_test_ids=1950, seed_ids=[0, 1, 2], resamples=10000,
             bootstrap_output=str(out), bootstrap_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
             sources=[payload["reuse_provenance"] for payload in full.values()])
(root / "attention-full-budget-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
(proofs / "full-budget-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
print(json.dumps(dict(primary_cells=144, paired_cells=53, standardized_runs=len(aggregate["runs"]), interval=interval), indent=2))
