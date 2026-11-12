"""Resume complete raw-attention sibling pairs on the explicitly granted GPU; preserve CPU fits."""

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

root = Path("/tmp/vtb-delivery/issues/24")
public = Path("results/diagnostics/kimi-k31-pro-checkpoint")
gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
grant = json.loads((root / "gpu-grant.json").read_text())
assert gpu == "0" and grant["physical_gpu"] == 0 and grant["active"] and grant["phase"] == "remaining-suite"
assert torch.get_num_threads() == 8
prior = [json.loads(line) for line in (root / "execution.jsonl").read_text().splitlines() if json.loads(line).get("phase") == "cpu8-raw-recognition"]
assert prior[-1]["event"] == "end" and prior[-1]["exit_code"] == 0
import fcntl
with Path("/tmp/vtb-delivery/locks/gpu0.lock").open("a") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        fcntl.flock(lock, fcntl.LOCK_UN)
        raise RuntimeError("Run under run_logged.py --gpu 0 full-process flock")
hardware = subprocess.check_output(["bash", "-ic", "nvidia-smi -i 0 --query-gpu=uuid,name --format=csv,noheader"], text=True).strip()
ids = json.loads((root / "ids-imagenet100.json").read_text())["ids"]
splits = json.loads((root / "splits-imagenet100.json").read_text())
labels = json.loads(Path("/tmp/vtb-delivery/data/imagenet100/validation_labels.json").read_text())["labels"]
models = {"kimi_k31_pro": "immanuelpeter/MoonViT-K3.1-Pro", "moonvit_v2": "immanuelpeter/MoonViT-V2"}
book = json.loads((root / "completed-inventory.json").read_text())
expected = json.loads((root / "expected-inventory.json").read_text())
parts = {model: {} for model in models}
records = []
specs = [("tower", layer) for layer in (3, 7, 10, 14, 17, 20, 24, 27)] + [("merged", 27), ("projected", 27)]
for stage, layer in specs:
    for model, model_id in models.items():
        out = root / "measurements/recognition-cells" / f"{model}_probe_attention_raw_{stage}_L{layer:02d}.json"
        argv = [".venv/bin/python", "-m", "vtb.probe_run", "--run", str(root / "features/gpu1-bf16-bs1/semantic" / f"{model}_448_pool4"),
                "--labels", "/tmp/vtb-delivery/data/imagenet100/validation_labels.json", "--out", str(out), "--readout", "attention",
                "--stages", stage, "--depth-points", str(layer), "--device", "cuda:0", "--save-predictions"]
        began, resumed = time.monotonic(), out.exists()
        if not resumed:
            with out.with_suffix(".log").open("w") as log:
                result = subprocess.run(["bash", "-ic", shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise SystemExit(result.returncode)
        payload = json.loads(out.read_text())
        audit = validate_probe(payload, model_id, ids, "recognition", "raw", "attention",
                               expected_splits=splits, expected_slices={(stage, layer)})
        cell = payload["cells"][0]
        assert cell["fit_device"] in {"cpu", "cuda:0"} and cell["fit_threads"] == 8
        if not resumed:
            assert cell["fit_device"] == "cuda:0"
        assert cell["test_labels"] == [labels[image] for image in cell["test_image_ids"]]
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        audit.update(model=model, stage=stage, layer=layer, arm="raw", fit_threads=8, fit_device=cell["fit_device"], physical_gpu=0 if cell["fit_device"] == "cuda:0" else None, gpu_identity=hardware if cell["fit_device"] == "cuda:0" else None,
                     scope="complete full-data single cell; not a full matrix", source_sha256=digest,
                     seconds=time.monotonic() - began, resumed=resumed, command=shlex.join(argv))
        if not resumed:
            out.with_suffix(".validation.json").write_text(json.dumps(audit, indent=2) + "\n")
        else:
            original = json.loads(out.with_suffix(".validation.json").read_text())
            assert original["source_sha256"] == digest and original["fit_device"] == cell["fit_device"]
            audit["requested_resume_command_not_executed"] = audit["command"]
            audit["command"] = original["command"]
            audit["verified_original_validation_sha256"] = hashlib.sha256(out.with_suffix(".validation.json").read_bytes()).hexdigest()
        folder = public / "recognition-cells"
        shutil.copy2(out, folder / out.name)
        shutil.copy2(out.with_suffix(".validation.json"), folder / out.with_suffix(".validation.json").name)
        if out.with_suffix(".log").exists():
            shutil.copy2(out.with_suffix(".log"), folder / out.with_suffix(".txt").name)
        parts[model][stage, layer] = (payload, out)
        key = dict(task="recognition", model=model, stage=stage, layer=layer, arm="raw", readout="attention")
        existing = [row for row in book["completed"] if row["cell"] == key]
        if existing:
            assert len(existing) == 1 and existing[0]["sha256"] == digest
        else:
            book["completed"].append(dict(cell=key, source=str(out), sha256=digest))
        keys = {json.dumps(row["cell"], sort_keys=True) for row in book["completed"]}
        assert len(keys) == len(book["completed"]) and keys <= {json.dumps(row, sort_keys=True) for row in expected["cells"]}
        book.update(completed_primary_cells=len(keys), pending=[row for row in expected["cells"] if json.dumps(row, sort_keys=True) not in keys])
        for path in (root / "completed-inventory.json", public / "completed-inventory.json"):
            path.write_text(json.dumps(book, indent=2) + "\n")
        records.append(audit)
        (root / "gpu-raw-recognition-progress.json").write_text(json.dumps(dict(complete=False, results=records), indent=2) + "\n")
        print(json.dumps(audit), flush=True)
    first, second = [parts[model][stage, layer][0]["cells"][0] for model in models]
    assert first["fit_device"] == second["fit_device"] and first["fit_threads"] == second["fit_threads"]
for model, model_id in models.items():
    assert len(parts[model]) == 10
    template = next(iter(parts[model].values()))[0]
    output = {key: copy.deepcopy(value) for key, value in template.items() if key != "cells"}
    output["protocol"]["device"] = "per-cell"
    output["cells"] = []
    for key in sorted(parts[model]):
        payload, path = parts[model][key]
        assert {key: value for key, value in payload["protocol"].items() if key != "device"} == {key: value for key, value in template["protocol"].items() if key != "device"} and payload["feature_provenance"] == template["feature_provenance"]
        cell = copy.deepcopy(payload["cells"][0])
        cell["source_fit"] = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), device=cell["fit_device"], threads=8, protocol=copy.deepcopy(payload["protocol"]), physical_gpu=0 if cell["fit_device"] == "cuda:0" else None)
        output["cells"].append(cell)
    validate_probe(output, model_id, ids, "recognition", "raw", "attention", expected_splits=splits)
    path = root / "measurements/recognition" / f"{model}_probe_attention_raw.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    folder = Path("results/recognition") if model == "kimi_k31_pro" else public / "sibling-control/recognition"
    shutil.copy2(path, folder / path.name)
(root / "gpu-raw-recognition-progress.json").write_text(json.dumps(dict(complete=True, results=records,
    complete_raw_attention_cells=20), indent=2) + "\n")
