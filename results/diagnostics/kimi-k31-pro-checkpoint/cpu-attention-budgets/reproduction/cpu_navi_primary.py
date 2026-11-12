"""Evaluate the full NAVI pair manifest on CPU4 for both pinned checkpoints."""

import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from scripts.correspondence_run import DATASETS
from scripts.validate_kimi_results import validate_correspondence

root = Path("/tmp/vtb-delivery/issues/24")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("OMP_NUM_THREADS") != "4":
    raise SystemExit("CPU4 only; no CUDA reservation is held")
data = Path("/tmp/vtb-delivery/data/correspondence/navi/navi_v1")
expected = json.loads((root / "pairs-navi.json").read_text())["ids"]
dataset = DATASETS["navi"](data, 448)
assert len(dataset) == len(expected) == 555
selected = [dataset[index]["id"] for index in range(len(dataset))]
if selected != expected:
    raise ValueError("current actual NAVI selection differs from archived manifest")
(root / "navi-cpu-source-selection.json").write_text(json.dumps(dict(passed=True, root=str(data), selected_pairs=555,
    ids=selected, ordered_ids_sha256=hashlib.sha256(json.dumps(selected).encode()).hexdigest()), indent=2) + "\n")
results = []
for model, model_id in (("kimi_k31_pro", "immanuelpeter/MoonViT-K3.1-Pro"), ("moonvit_v2", "immanuelpeter/MoonViT-V2")):
    if (root / "cpu-boundary-request.json").exists():
        (root / "cpu-navi-primary-progress.json").write_text(json.dumps(dict(complete=False, results=results,
            stopped_at_model_boundary=True, next_model=model), indent=2) + "\n")
        print("CPU boundary request honored before the next model; completed verified outputs retained", flush=True)
        raise SystemExit(0)
    out = root / "measurements/correspondence" / f"{model}_correspondence_navi.json"
    argv = [".venv/bin/python", str(root / "pinned_kimi_cli.py"), "scripts.correspondence_run", "--model", model,
            "--dataset", "navi", "--root", str(data), "--out", str(out), "--resolution", "448",
            "--evaluation-side", "64", "--num-correspondences", "1000", "--device", "cpu"]
    began = time.monotonic()
    resumed = out.exists()
    if not resumed:
        with out.with_suffix(".log").open("w") as log:
            process = subprocess.run(["bash", "-ic", shlex.join(argv)], stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise SystemExit(process.returncode)
    payload = json.loads(out.read_text())
    audit = validate_correspondence(payload, expected)
    runtime = json.loads(out.with_suffix(".runtime.json").read_text())
    assert len(runtime["adapters"]) == 1
    adapter = runtime["adapters"][0]
    assert payload["model_id"] == adapter["checkpoint"]["model_id"] == model_id
    assert adapter["device"] == "cpu" and adapter["cuda_visible_devices"] == "" and adapter["threads"] == 4
    assert adapter["exact_immutable_identity"] and adapter["identity_tensors"] == 168 and adapter["dtype"] == "torch.bfloat16"
    assert payload["evaluation_side"] == 64 and payload["num_correspondences"] == 1000
    audit.update(model=model, cells=3, device="cpu", torch_threads=4, batch_size=1, dtype="bfloat16",
                 source_manifest="navi-cpu-source-selection.json", runtime=runtime, seconds=time.monotonic() - began,
                 resumed=resumed, command=shlex.join(argv), payload_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    results.append(audit)
    (root / "cpu-navi-primary-progress.json").write_text(json.dumps(dict(complete=False, results=results), indent=2) + "\n")
    print(json.dumps(audit), flush=True)
(root / "cpu-navi-primary-progress.json").write_text(json.dumps(dict(complete=True, results=results, cells=6), indent=2) + "\n")
