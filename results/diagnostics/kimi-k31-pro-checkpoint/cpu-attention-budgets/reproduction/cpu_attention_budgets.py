"""Run primary and prescribed standardized attention budgets sequentially on CPU4."""

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
    raise SystemExit("CPU4 only")
ids = json.loads((root / "ids-imagenet100.json").read_text())["ids"]
label_map = json.loads(Path("/tmp/vtb-delivery/data/imagenet100/validation_labels.json").read_text())["labels"]
results = []
for model, model_id in (("kimi_k31_pro", "immanuelpeter/MoonViT-K3.1-Pro"), ("moonvit_v2", "immanuelpeter/MoonViT-V2")):
    run = root / "features/gpu1-bf16-bs1/semantic" / f"{model}_448_pool4"
    assert json.loads((run / "progress.json").read_text())["complete"]
    for standardized in (False, True):
        kind = "standardized" if standardized else "label-budget"
        directory = root / "measurements" / ("diagnostics" if standardized else "label-budget")
        for fraction in (.01, .05, .2, 1.):
            out = directory / f"{model}_probe_attention_matched_f{round(fraction * 100):03d}.json"
            argv = [".venv/bin/python", "-m", "vtb.probe_run", "--run", str(run),
                    "--labels", "/tmp/vtb-delivery/data/imagenet100/validation_labels.json", "--out", str(out),
                    "--readout", "attention", "--match-capacity", "--device", "cpu", "--stages", "tower",
                    "--depth-points", "27", "--label-fraction", str(fraction)]
            if standardized:
                argv.append("--standardize-channels")
            elif fraction == 1.:
                argv.append("--save-predictions")
            resumed = out.exists()
            began = time.monotonic()
            if not resumed:
                with out.with_suffix(".log").open("w") as log:
                    result = subprocess.run(["bash", "-ic", shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise SystemExit(result.returncode)
            payload = json.loads(out.read_text())
            splits = json.loads((root / f"splits-imagenet100-f{fraction}.json").read_text())
            audit = validate_probe(payload, model_id, ids, kind, "matched", "attention", fraction, splits)
            if not standardized and fraction == 1.:
                cell = payload["cells"][0]
                if cell["test_labels"] != [label_map[image] for image in cell["test_image_ids"]]:
                    raise ValueError("retained labels differ from the audited ImageNet label source")
                if cell["fit_device"] != "cpu" or cell["fit_threads"] != 4:
                    raise ValueError("retained predictions have different fit provenance")
            audit.update(model=model, kind=kind, fraction=fraction, fit_device="cpu", fit_threads=4,
                         seconds=time.monotonic() - began, resumed=resumed,
                         payload_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), command=shlex.join(argv))
            out.with_suffix(".validation.json").write_text(json.dumps(audit, indent=2) + "\n")
            results.append(audit)
            (root / "cpu-attention-budget-progress.json").write_text(json.dumps(dict(complete=False, results=results), indent=2) + "\n")
            print(json.dumps(audit), flush=True)
(root / "cpu-attention-budget-progress.json").write_text(json.dumps(dict(complete=True, results=results, cells=16), indent=2) + "\n")
