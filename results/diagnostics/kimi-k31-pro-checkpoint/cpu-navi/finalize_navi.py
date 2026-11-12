"""Validate full CPU NAVI results, run prescribed bootstraps and archive them."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from scripts.validate_kimi_results import finite_numbers, validate_correspondence, validate_pairing

root = Path("/tmp/vtb-delivery/issues/24")
destination = Path("results/diagnostics/kimi-k31-pro-checkpoint")
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("OMP_NUM_THREADS") == "4"
phase = json.loads((root / "cpu-navi-primary-progress.json").read_text())
assert phase["complete"] and phase["cells"] == 6
ids = json.loads((root / "pairs-navi.json").read_text())["ids"]
proofs = destination / "cpu-navi"
completed, paired = [], []
for record in phase["results"]:
    source = root / "measurements/correspondence" / f"{record['model']}_correspondence_navi.json"
    payload = json.loads(source.read_text())
    validate_correspondence(payload, ids)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record["payload_sha256"]
    adapter = record["runtime"]["adapters"][0]
    assert adapter["validated_image_forwards"] == 1110 and all(adapter["native_integrity"].values())
    assert adapter["device"] == "cpu" and adapter["threads"] == 4 and adapter["cuda_visible_devices"] == ""
    public = Path("results/correspondence") if record["model"] == "kimi_k31_pro" else destination / "sibling-control/correspondence"
    public.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, public / source.name)
    for suffix in (".validation.json", ".runtime.json", ".log"):
        path = source.with_suffix(suffix)
        shutil.copy2(path, proofs / path.name.replace(".log", ".txt"))
    for cell in payload["cells"]:
        completed.append(dict(cell=dict(task="navi", model=record["model"], stage=cell["stage"], layer=27),
                              source=str(source), sha256=record["payload_sha256"]))
plan = json.loads((root / "fullsuite-plan.json").read_text())
jobs = [job for job in plan["cpu_post_jobs"] if job["kind"] == "correspondence" and "navi" in job["first"]]
assert len(jobs) == 4
for job in jobs:
    first, second = [json.loads(Path(job[key]).read_text()) for key in ("first", "second")]
    audit = validate_pairing(first, second, f"{job['stage']}:27", f"{job['second_stage']}:27")
    assert len(audit["finite_ids"]) == 555 and not audit["exclusion_union"] and audit["same_excluded_ids"]
    out = Path(job["out"])
    if not out.exists():
        with out.with_suffix(".log").open("w") as log:
            process = subprocess.run(["bash", "-ic", shlex.join(job["argv"])], stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise SystemExit(process.returncode)
    payload = json.loads(out.read_text())
    assert finite_numbers(payload)
    a = {row["pair_id"]: row["cells"][f"{job['stage']}:27"]["recall_2cm"] for row in first["measurements"]}
    b = {row["pair_id"]: row["cells"][f"{job['second_stage']}:27"]["recall_2cm"] for row in second["measurements"]}
    point = sum(a[identity] - b[identity] for identity in ids) / 555
    sibling = job["first"] != job["second"]
    interval = payload["difference"] if sibling else payload["first_advantage"]
    assert abs(point - interval["point_estimate"]) < 1e-6
    assert interval["resamples"] == 10000 and interval["confidence"] == .95 and interval["bootstrap_seed"] == 0
    if sibling:
        assert payload["finite_pairs"] == 555 and not payload["excluded_nonfinite_pair_ids"]
        assert {row["pair_id"] for row in payload["measurements"]} == set(ids)
    else:
        assert interval["pairs"] == 555
    key = dict(comparison="kimi_k31_pro-minus-moonvit_v2" if sibling else "kimi_k31_pro-projected-minus-tower",
               task="navi", first_stage=job["stage"], second_stage=job["second_stage"])
    record = dict(cell=key, source=str(out), sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  pairing_audit=audit, actual_point_from_same_rows=point, fit_device="cpu", threads=4)
    out.with_suffix(".validation.json").write_text(json.dumps(record, indent=2) + "\n")
    paired.append(record)
    shutil.copy2(out, Path("results/bootstrap") / out.name)
    shutil.copy2(out.with_suffix(".validation.json"), proofs / out.with_suffix(".validation.json").name)
    job.update(status="complete_content_verified_cpu", result_sha256=record["sha256"])
book = json.loads((root / "completed-inventory.json").read_text())
book["completed"].extend(completed)
book["completed_paired"].extend(paired)
expected = json.loads((root / "expected-inventory.json").read_text())
actual = {json.dumps(row["cell"], sort_keys=True) for row in book["completed"]}
actual_paired = {json.dumps(row["cell"], sort_keys=True) for row in book["completed_paired"]}
assert len(actual) == len(book["completed"]) == 150 and actual <= {json.dumps(row, sort_keys=True) for row in expected["cells"]}
assert len(actual_paired) == 57 and actual_paired <= {json.dumps(row, sort_keys=True) for row in expected["paired_comparison_cells"]}
book.update(completed_primary_cells=150, completed_paired_cells=57,
            pending=[row for row in expected["cells"] if json.dumps(row, sort_keys=True) not in actual])
for path in (root / "completed-inventory.json", destination / "completed-inventory.json"):
    path.write_text(json.dumps(book, indent=2) + "\n")
for job in plan["jobs"]:
    if "--dataset" in job.get("argv", []) and job["argv"][job["argv"].index("--dataset") + 1] == "navi":
        job.update(status="complete_content_verified_cpu", actual_device="cpu", actual_threads=4)
(root / "fullsuite-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
shutil.copy2(root / "fullsuite-plan.json", destination / "fullsuite-plan.json")
shutil.copy2(root / "cpu-navi-primary-progress.json", proofs / "validation-summary.json")
shutil.copy2(root / "cpu-navi-primary-native-validated.log", proofs / "execution-log.txt")
(root / "cpu-navi-bootstrap-progress.json").write_text(json.dumps(dict(complete=True, cells=4, results=paired), indent=2) + "\n")
print(json.dumps(dict(primary_cells=150, paired_cells=57, navi_primary=6, navi_paired=4), indent=2))
