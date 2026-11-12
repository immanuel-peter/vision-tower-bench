"""Run complete CPU mean label budgets and audit identical 100% cell reuse."""

import copy
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from scripts.validate_kimi_results import validate_probe

root = Path("/tmp/vtb-delivery/issues/24")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("OMP_NUM_THREADS") != "4":
    raise SystemExit("CPU budget phase requires hidden CUDA and four threads")
recognition = json.loads((root / "cpu-mean-primary-progress.json").read_text())
if not recognition["complete"] or recognition["cells"] != 40:
    raise SystemExit("complete audited mean recognition required before budget reuse")
ids = json.loads((root / "ids-imagenet100.json").read_text())["ids"]
results = []
for model, model_id in (("kimi_k31_pro", "immanuelpeter/MoonViT-K3.1-Pro"), ("moonvit_v2", "immanuelpeter/MoonViT-V2")):
    run = root / "features/gpu1-bf16-bs1/semantic" / f"{model}_448_pool4"
    for fraction in (.01, .05, .2, 1.):
        started = time.monotonic()
        out = root / "measurements/label-budget" / f"{model}_probe_mean_matched_f{round(fraction * 100):03d}.json"
        reused = None
        if not out.exists():
            if fraction == 1.:
                source = root / "measurements/recognition" / f"{model}_probe_mean_matched.json"
                payload = copy.deepcopy(json.loads(source.read_text()))
                payload["cells"] = [cell for cell in payload["cells"] if cell["stage"] == "tower" and cell["layer_index"] == 27]
                if len(payload["cells"]) != 1 or payload["protocol"]["device"] != "cpu":
                    raise ValueError("100% reuse requires the same actual CPU fitted Tower cell")
                reused = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                              compatibility="Same features, full training/validation/test IDs, PCA seed0, LR search seed0, selected-LR3seeds,20epochs,CPU4; each cell seeds independently.")
                payload["reuse_provenance"] = reused
                out.write_text(json.dumps(payload, indent=2) + "\n")
            else:
                argv = [".venv/bin/python", "-m", "vtb.probe_run", "--run", str(run),
                        "--labels", "/tmp/vtb-delivery/data/imagenet100/validation_labels.json",
                        "--out", str(out), "--readout", "mean", "--match-capacity", "--device", "cpu",
                        "--stages", "tower", "--depth-points", "27", "--label-fraction", str(fraction)]
                with out.with_suffix(".log").open("w") as log:
                    result = subprocess.run(["bash", "-ic", shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise SystemExit(result.returncode)
        payload = json.loads(out.read_text())
        splits = json.loads((root / f"splits-imagenet100-f{fraction}.json").read_text())
        audit = validate_probe(payload, model_id, ids, "label-budget", "matched", readout="mean",
                               label_fraction=fraction, expected_splits=splits)
        audit.update(model=model, fraction=fraction, fit_device="cpu", fit_threads=4,
                     seconds=time.monotonic() - started,
                     reuse=payload.get("reuse_provenance"),
                     payload_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
        out.with_suffix(".validation.json").write_text(json.dumps(audit, indent=2) + "\n")
        results.append(audit)
        (root / "cpu-mean-label-budget-progress.json").write_text(json.dumps(dict(complete=False, results=results), indent=2) + "\n")
        print(json.dumps(audit), flush=True)
(root / "cpu-mean-label-budget-progress.json").write_text(json.dumps(dict(complete=True, results=results, cells=8,
    new_fitted_cells=6, identical_full_budget_reused_cells=2), indent=2) + "\n")
