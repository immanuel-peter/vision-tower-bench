"""Run full corrected mean recognition cells sequentially after GPU release."""

import datetime
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
    raise SystemExit("CPU mean phase requires hidden CUDA and four CPU threads")
release = json.loads((root / "gpu-extraction-release.json").read_text())
if not release["released"]:
    raise SystemExit("GPU extraction release is required before CPU probe training")
pair = json.loads((root / "gpu-extraction-pair-audit.json").read_text())
if not pair["passed"] or not pair["complete"]:
    raise SystemExit("complete content-verified sibling extraction is required")
ids = json.loads((root / "ids-imagenet100.json").read_text())["ids"]
splits = json.loads((root / "splits-imagenet100.json").read_text())
results = []
for model, model_id in (("kimi_k31_pro", "immanuelpeter/MoonViT-K3.1-Pro"), ("moonvit_v2", "immanuelpeter/MoonViT-V2")):
    run = root / "features/gpu1-bf16-bs1/semantic" / f"{model}_448_pool4"
    progress = json.loads((run / "progress.json").read_text())
    if not progress["complete"] or progress["images"] != 13000:
        raise SystemExit("complete full-size semantic cache required")
    for arm in ("raw", "matched"):
        out = root / "measurements/recognition" / f"{model}_probe_mean_{arm}.json"
        argv = [".venv/bin/python", "-m", "vtb.probe_run", "--run", str(run),
                "--labels", "/tmp/vtb-delivery/data/imagenet100/validation_labels.json",
                "--out", str(out), "--readout", "mean", "--device", "cpu"]
        if arm == "matched":
            argv.append("--match-capacity")
        resumed = out.is_file()
        began = time.monotonic()
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if not resumed:
            with out.with_suffix(".log").open("w") as log:
                process = subprocess.run(["bash", "-ic", shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise SystemExit(process.returncode)
        payload = json.loads(out.read_text())
        if payload["protocol"]["device"] != "cpu":
            raise ValueError("CPU phase cannot reuse a different training device")
        audit = validate_probe(payload, model_id, ids, "recognition", arm,
                               readout="mean", expected_splits=splits)
        audit.update(model=model, arm=arm, readout="mean", fit_device="cpu", fit_threads=4,
                     feature_device="cuda:0", resumed=resumed, started=started,
                     seconds=time.monotonic() - began, command=shlex.join(argv),
                     payload_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
        out.with_suffix(".validation.json").write_text(json.dumps(audit, indent=2) + "\n")
        results.append(audit)
        (root / "cpu-mean-primary-progress.json").write_text(json.dumps(dict(complete=False, results=results), indent=2) + "\n")
        print(json.dumps(audit), flush=True)
(root / "cpu-mean-primary-progress.json").write_text(json.dumps(dict(complete=True, results=results, cells=sum(row["cells"] for row in results)), indent=2) + "\n")
