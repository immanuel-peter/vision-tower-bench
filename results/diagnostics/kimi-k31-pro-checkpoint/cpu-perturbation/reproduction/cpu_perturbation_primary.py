"""Fit unchanged primary perturbation readouts on verified sibling GPU caches."""

import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from scripts.perturbation_run import condition_details
from scripts.validate_kimi_results import validate_perturbation

root = Path("/tmp/vtb-delivery/issues/24")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("OMP_NUM_THREADS") != "4":
    raise SystemExit("CPU perturbation requires hidden CUDA and four threads")
if not json.loads((root / "gpu-extraction-release.json").read_text())["released"]:
    raise SystemExit("GPU extraction release required")
manifest_path = Path("/tmp/vtb-delivery/data/perturbation/transforms.manifest.json")
manifest = json.loads(manifest_path.read_text())
details = condition_details(manifest)
ids = json.loads((root / "ids-perturbation.json").read_text())["ids"]
splits = json.loads((root / "splits-perturbation.json").read_text())
results = []
for model, model_id, revision in (("kimi_k31_pro", "immanuelpeter/MoonViT-K3.1-Pro", "6ab95e2daf86e8100be2e712a7a71c0fba26dac4"),
                                  ("moonvit_v2", "immanuelpeter/MoonViT-V2", "bcaedfccd1f07d23ab20de8a0a168668c27cacb3")):
    directory = f"{model}_448_pool4"
    features = root / "features/gpu1-bf16-bs1/perturbation"
    out = root / "measurements/perturbation" / f"{model}_perturbation.json"
    argv = [".venv/bin/python", "-m", "scripts.perturbation_run", "--clean-run", str(features / "identity" / directory),
            "--features-root", str(features), "--model-dir", directory,
            "--labels", "/tmp/vtb-delivery/data/imagenet100/validation_labels.json",
            "--transform-manifest", str(manifest_path), "--out", str(out), "--device", "cpu"]
    started = time.monotonic()
    resumed = out.exists()
    if not resumed:
        with out.with_suffix(".log").open("w") as log:
            process = subprocess.run(["bash", "-ic", shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise SystemExit(process.returncode)
    payload = json.loads(out.read_text())
    audit = validate_perturbation(payload, model_id, ids, manifest["conditions"], splits, revision,
                                  hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    for cell in payload["cells"]:
        for row in cell["conditions"]:
            if any(row[key] != details[row["condition"]][key] for key in ("factor", "level", "value")):
                raise ValueError("actual condition factor/level/value differs from source manifest")
    audit.update(model=model, seconds=time.monotonic() - started, resumed=resumed,
                 command=shlex.join(argv), payload_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    results.append(audit)
    (root / "cpu-perturbation-primary-progress.json").write_text(json.dumps(dict(complete=False, results=results), indent=2) + "\n")
    print(json.dumps(audit), flush=True)
(root / "cpu-perturbation-primary-progress.json").write_text(json.dumps(dict(complete=True, results=results, cells=78), indent=2) + "\n")
