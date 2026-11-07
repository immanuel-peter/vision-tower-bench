"""Resume full-data matched attention cells and pair their actual saved predictions."""

import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import torch

from scripts.validate_kimi_results import validate_probe
from vtb.semantic_bootstrap import paired_image_bootstrap

root = Path("/tmp/vtb-delivery/issues/24")
public = Path("results/diagnostics/kimi-k31-pro-checkpoint")
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and torch.get_num_threads() == 4
ids = json.loads((root / "ids-imagenet100.json").read_text())["ids"]
splits = json.loads((root / "splits-imagenet100.json").read_text())
labels_by_id = json.loads(Path("/tmp/vtb-delivery/data/imagenet100/validation_labels.json").read_text())["labels"]
models = {"kimi_k31_pro": "immanuelpeter/MoonViT-K3.1-Pro", "moonvit_v2": "immanuelpeter/MoonViT-V2"}
plan = json.loads((root / "fullsuite-plan.json").read_text())
inventory = json.loads((root / "expected-inventory.json").read_text())
book = json.loads((root / "completed-inventory.json").read_text())
parts = {model: {} for model in models}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upsert(collection, record):
    found = [row for row in collection if row["cell"] == record["cell"]]
    if found:
        assert len(found) == 1 and found[0]["sha256"] == record["sha256"]
    else:
        collection.append(record)


def save_book():
    primary = {json.dumps(row["cell"], sort_keys=True) for row in book["completed"]}
    paired = {json.dumps(row["cell"], sort_keys=True) for row in book["completed_paired"]}
    assert len(primary) == len(book["completed"]) and primary <= {json.dumps(row, sort_keys=True) for row in inventory["cells"]}
    assert len(paired) == len(book["completed_paired"]) and paired <= {json.dumps(row, sort_keys=True) for row in inventory["paired_comparison_cells"]}
    book.update(completed_primary_cells=len(primary), completed_paired_cells=len(paired),
                pending=[row for row in inventory["cells"] if json.dumps(row, sort_keys=True) not in primary])
    text = json.dumps(book, indent=2) + "\n"
    (root / "completed-inventory.json").write_text(text)
    (public / "completed-inventory.json").write_text(text)


def cell_path(model, stage, layer):
    return root / "measurements/recognition-cells" / f"{model}_probe_attention_matched_{stage}_L{layer:02d}.json"


def audit_cell(model, stage, layer, path):
    payload = json.loads(path.read_text())
    audit = validate_probe(payload, models[model], ids, "recognition", "matched", "attention",
                           expected_splits=splits, expected_slices={(stage, layer)})
    cell = payload["cells"][0]
    assert cell["fit_device"] == "cpu" and cell["fit_threads"] == 4
    assert cell["test_labels"] == [labels_by_id[image] for image in cell["test_image_ids"]]
    audit.update(model=model, stage=stage, layer=layer, scope="one complete full-data cell; not a complete matrix",
                 payload_sha256=digest(path), fit_device="cpu", fit_threads=4)
    path.with_suffix(".validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    folder = public / "recognition-cells"
    folder.mkdir(exist_ok=True)
    shutil.copy2(path, folder / path.name)
    shutil.copy2(path.with_suffix(".validation.json"), folder / path.with_suffix(".validation.json").name)
    if path.with_suffix(".log").exists():
        shutil.copy2(path.with_suffix(".log"), folder / path.with_suffix(".txt").name)
    parts[model][stage, layer] = (payload, path)
    upsert(book["completed"], dict(cell=dict(task="recognition", model=model, stage=stage, layer=layer,
                   arm="matched", readout="attention"), source=str(path), sha256=digest(path)))
    save_book()
    return audit


def option(argv, key):
    return argv[argv.index(key) + 1]


def pair_available():
    for job in plan["jobs"]:
        argv = job.get("argv", [])
        if "scripts.semantic_bootstrap" not in argv:
            continue
        first_name, second_name = option(argv, "--first-name"), option(argv, "--second-name")
        first_model = "kimi_k31_pro" if first_name.startswith("kimi_k31_pro") else "moonvit_v2"
        second_model = "kimi_k31_pro" if second_name.startswith("kimi_k31_pro") else "moonvit_v2"
        akey, bkey = (option(argv, "--first-stage"), int(option(argv, "--first-layer"))), (option(argv, "--second-stage"), int(option(argv, "--second-layer")))
        if akey not in parts[first_model] or bkey not in parts[second_model]:
            continue
        a, apath = parts[first_model][akey]
        b, bpath = parts[second_model][bkey]
        assert a["protocol"] == b["protocol"] and a["feature_provenance"]["details"]["physical_gpu"] == b["feature_provenance"]["details"]["physical_gpu"] == 1
        ac, bc = a["cells"][0], b["cells"][0]
        assert ac["test_image_ids"] == bc["test_image_ids"] == splits["test"] and ac["test_labels"] == bc["test_labels"]
        at, bt, target = torch.tensor(ac["test_predictions_by_seed"]), torch.tensor(bc["test_predictions_by_seed"]), torch.tensor(ac["test_labels"])
        out = Path(option(argv, "--out"))
        if out.exists():
            output = json.loads(out.read_text())
            assert [row["image_id"] for row in output["predictions"]] == splits["test"]
            assert all(row[first_name] == at[:, i].tolist() and row[second_name] == bt[:, i].tolist() and row["label"] == int(target[i]) for i, row in enumerate(output["predictions"]))
        else:
            def summary(cell, name):
                return dict(name=name, model_id=cell["model_id"], stage=cell["stage"], layer_index=cell["layer_index"],
                            relative_depth=cell["relative_depth"], learning_rate=cell["learning_rate"],
                            learning_rate_search=cell["learning_rate_search_unrounded"], validation_accuracy=cell["validation_accuracy_unrounded"],
                            seed_accuracies=cell["test_accuracy_by_seed"], mean_accuracy=sum(cell["test_accuracy_by_seed"]) / 3)
            output = dict(protocol=dict(readout="attention", arm="matched", epochs=20, seeds=3, test_images=1950,
                pairing="seed-mean correctness, paired over test images", device="cpu", threads=4),
                first=summary(ac, first_name), second=summary(bc, second_name),
                difference=paired_image_bootstrap(at == target, bt == target),
                predictions=[dict(image_id=image, label=int(target[i]), **{first_name: at[:, i].tolist(), second_name: bt[:, i].tolist()}) for i, image in enumerate(splits["test"])],
                source_fit_provenance=dict(first=dict(file=str(apath), sha256=digest(apath)), second=dict(file=str(bpath), sha256=digest(bpath)),
                                          same_device="cpu", same_threads=4, source_producer="GPU1 BF16 batch1 pool4"))
            out.write_text(json.dumps(output, indent=2) + "\n")
        interval = output["difference"]
        point = ((at == target).double().mean() - (bt == target).double().mean()).item()
        assert abs(interval["point_estimate"] - point) < 1e-6 and interval["resamples"] == 10000 and interval["bootstrap_seed"] == 0 and interval["confidence"] == .95
        comparison = "kimi_k31_pro-minus-moonvit_v2" if first_model != second_model else "kimi_k31_pro-projected-minus-tower"
        key = dict(comparison=comparison, task="recognition", first_stage=akey[0], second_stage=bkey[0], first_layer=akey[1], second_layer=bkey[1], arm="matched", readout="attention")
        upsert(book["completed_paired"], dict(cell=key, source=str(out), sha256=digest(out), reused_same_fit_predictions=True))
        shutil.copy2(out, Path("results/bootstrap") / out.name)
        job.update(status="complete_content_verified_cpu", result_sha256=digest(out))
    save_book()


for model in models:
    audit_cell(model, "tower", 27, cell_path(model, "tower", 27))
specs = [("projected", 27), ("merged", 27)] + [("tower", layer) for layer in (24, 20, 17, 14, 10, 7, 3)]
done = []
for stage, layer in specs:
    for model in models:
        if (root / "cpu-boundary-request.json").exists():
            (root / "cpu-matched-recognition-progress.json").write_text(json.dumps(dict(complete=False, results=done,
                stopped_at_cell_boundary=True, next_cell=[model, stage, layer]), indent=2) + "\n")
            raise SystemExit(0)
        out = cell_path(model, stage, layer)
        began, resumed = time.monotonic(), out.exists()
        if not resumed:
            argv = [".venv/bin/python", "-m", "vtb.probe_run", "--run", str(root / "features/gpu1-bf16-bs1/semantic" / f"{model}_448_pool4"),
                "--labels", "/tmp/vtb-delivery/data/imagenet100/validation_labels.json", "--out", str(out), "--readout", "attention",
                "--match-capacity", "--stages", stage, "--depth-points", str(layer), "--device", "cpu", "--save-predictions"]
            with out.with_suffix(".log").open("w") as log:
                process = subprocess.run(["bash", "-ic", shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise SystemExit(process.returncode)
        audit = audit_cell(model, stage, layer, out)
        audit.update(seconds=time.monotonic() - began, resumed=resumed)
        done.append(audit)
        pair_available()
        (root / "cpu-matched-recognition-progress.json").write_text(json.dumps(dict(complete=False, results=done), indent=2) + "\n")
        print(json.dumps(audit), flush=True)
for model in models:
    assert len(parts[model]) == 10
    first, _ = parts[model]["tower", 27]
    combined = {key: copy.deepcopy(value) for key, value in first.items() if key not in ("cells", "scope", "reuse_provenance")}
    combined["cells"] = []
    for key in sorted(parts[model]):
        payload, source = parts[model][key]
        assert payload["protocol"] == first["protocol"] and payload["feature_provenance"] == first["feature_provenance"]
        cell = copy.deepcopy(payload["cells"][0])
        cell["source_fit"] = dict(file=str(source), sha256=digest(source), device="cpu", threads=4)
        combined["cells"].append(cell)
    validate_probe(combined, models[model], ids, "recognition", "matched", "attention", expected_splits=splits)
    path = root / "measurements/recognition" / f"{model}_probe_attention_matched.json"
    path.write_text(json.dumps(combined, indent=2) + "\n")
    folder = Path("results/recognition") if model == "kimi_k31_pro" else public / "sibling-control/recognition"
    shutil.copy2(path, folder / path.name)
(root / "fullsuite-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
(root / "cpu-matched-recognition-progress.json").write_text(json.dumps(dict(complete=True, results=done, newly_fitted_cells=18,
    complete_matched_recognition_cells=20, paired_semantic_cells=11), indent=2) + "\n")
