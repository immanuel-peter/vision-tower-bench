"""Run prescribed paired perturbation comparisons and audit their actual rows."""

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from scripts.validate_kimi_results import finite_numbers

root = Path("/tmp/vtb-delivery/issues/24")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("OMP_NUM_THREADS") != "4":
    raise SystemExit("CPU4 only")
phase = json.loads((root / "cpu-perturbation-primary-progress.json").read_text())
assert phase["complete"] and phase["cells"] == 78
for row in phase["results"]:
    source = root / "measurements/perturbation" / f"{row['model']}_perturbation.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == row["payload_sha256"]
plan = json.loads((root / "fullsuite-plan.json").read_text())
jobs = [job for job in plan["cpu_post_jobs"] if job["kind"] == "perturbation"]
assert len(jobs) == 4
test_ids = sorted(json.loads((root / "splits-perturbation.json").read_text())["test"])
conditions = json.loads(Path("/tmp/vtb-delivery/data/perturbation/transforms.manifest.json").read_text())["conditions"]
completed = []
for job in jobs:
    out = Path(job["out"])
    if not out.exists():
        with out.with_suffix(".log").open("w") as log:
            result = subprocess.run(["bash", "-ic", shlex.join(job["argv"])], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(result.returncode)
    payload = json.loads(out.read_text())
    assert finite_numbers(payload)
    assert set(payload["conditions"]) == set(conditions) - {"identity"}
    clean = None
    for condition, block in payload["conditions"].items():
        rows = block["measurements"]
        assert [row["image_id"] for row in rows] == test_ids and block["test_images"] == 300
        first, second = job["first_name"], job["second_name"]
        for row in rows:
            assert all(len(row[key]) == 3 and all(v in (0, 1) for v in row[key]) for key in (first, second, first + "_clean", second + "_clean"))
        point = sum(sum(row[first]) - sum(row[second]) for row in rows) / 900
        current_clean = [[row[first + "_clean"], row[second + "_clean"]] for row in rows]
        assert clean is None or clean == current_clean
        clean = current_clean
        interval = block["difference"]
        assert abs(point - interval["point_estimate"]) < 1e-6
        assert interval["resamples"] == 10000 and interval["bootstrap_seed"] == 0 and interval["confidence"] == .95
        assert interval["lower"] <= interval["upper"]
    clean_point = sum(sum(a) - sum(b) for a, b in clean) / 900
    assert payload["clean"]["test_images"] == 300 and abs(payload["clean"]["difference"]["point_estimate"] - clean_point) < 1e-6
    comparison = "kimi_k31_pro-minus-moonvit_v2" if job["first"] != job["second"] else "kimi_k31_pro-projected-minus-tower"
    record = dict(passed=True, source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (job["first"], job["second"])},
                  payload_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), file=str(out),
                  comparisons=[dict(comparison=comparison, task="perturbation", condition=condition,
                                    first_stage=job["stage"], second_stage=job["second_stage"]) for condition in conditions],
                  actual_paired_test_images=300, seeds=3, resamples=10000, fit_device="cpu", fit_threads=4)
    out.with_suffix(".validation.json").write_text(json.dumps(record, indent=2) + "\n")
    completed.append(record)
    print(json.dumps({key: value for key, value in record.items() if key != "comparisons"}), flush=True)
expected = json.loads((root / "expected-inventory.json").read_text())["paired_comparison_cells"]
actual = [cell for record in completed for cell in record["comparisons"]]
assert len(actual) == 52 and {json.dumps(cell, sort_keys=True) for cell in actual} == {json.dumps(cell, sort_keys=True) for cell in expected if cell["task"] == "perturbation"}
(root / "cpu-perturbation-bootstrap-progress.json").write_text(json.dumps(dict(complete=True, passed=True, cells=52, results=completed), indent=2) + "\n")
