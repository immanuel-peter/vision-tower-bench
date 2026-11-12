"""Assemble final reports only from complete, independently audited results."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
PUBLIC = Path("results/diagnostics/kimi-k31-pro-checkpoint")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inventory = read(ROOT / "completed-inventory.json")
    primary = read(ROOT / "completed-primary-independent-content-audit.json")
    paired = read(ROOT / "all-paired-intervals-independent-replay.json")
    hashes = read(ROOT / "full-experiment-inventory-hash-audit.json")
    tests = read(ROOT / "final-regression-tests.json")
    assert inventory["completed_primary_cells"] == primary["completed_primary_cells"] == 286
    assert inventory["completed_paired_cells"] == paired["comparison_cells"] == 102
    assert primary["passed"] and paired["passed"] and hashes["passed"] and tests["passed"]
    assert primary["complete_primary_inventory"] and paired["all_intervals_replayed_exactly"]
    for row in inventory["completed"] + inventory["completed_paired"]:
        assert digest(row["source"]) == row["sha256"]
    for gpu in (0, 1):
        release = read(ROOT / f"scheduler/gpu{gpu}-release.json")
        assert release["passed"] and release["memory_mib"] == 0 and not release["cuda_processes"]
        assert release["exclusive_lock_reacquired"] and release["exclusive_lock_released"]
        shutil.copy2(ROOT / f"scheduler/gpu{gpu}-release.json", PUBLIC / "two-gpu-scheduler")
    audits = PUBLIC / "independent-content-audits"
    for name in ("completed-primary-independent-content-audit.json", "all-paired-intervals-independent-replay.json",
                 "audit_completed_primary_content.py", "audit_all_paired_intervals.py"):
        shutil.copy2(ROOT / name, audits / name)
    for name in ("completed-inventory.json", "full-experiment-inventory-hash-audit.json",
                 "remaining-gpu-release.json", "remaining-gpu-progress.json", "final-regression-tests.json"):
        shutil.copy2(ROOT / name, PUBLIC / name)
    shutil.copy2(ROOT / "final-regression-tests.log", PUBLIC / "final-regression-tests.txt")
    shutil.copy2(ROOT / "cpu8-final-correspondence-pairs.log", audits / "cpu8-final-correspondence-pairs.txt")
    template = (ROOT / "final-readme.md.pending").read_text()
    for old, new in (("All168", "All 168"), ("fixed448", "fixed 448"), ("to0.93", "to 0.93"),
                     ("conditions x2", "conditions x 2"), ("entries,1,660", "entries, 1,660"),
                     ("shards,48.18", "shards, 48.18")):
        template = template.replace(old, new)
    template = template.replace("FINAL_INVENTORY_STATUS", "The full inventory contains **286/286 primary cells and 102/102 paired comparisons**.\nAll primary payloads passed independent content checks, and every interval was\nreplayed exactly from its actual rows. Both original GPU phases are released.\n[PR #29](https://github.com/immanuel-peter/vision-tower-bench/pull/29) contains\nthe implementation, measurements, audits and reproducibility records.")
    lines = ["| Task | Stage | Fresh K3 | Pro |", "| --- | --- | ---: | ---: |"]
    for task, metric, scale in (("depth", "d1", 100), ("normal", "mean_deg", 1)):
        models = {model: read(ROOT / f"measurements/geometry/{model}_geometry_{task}_matched.json")
                  for model in ("moonvit_v2", "kimi_k31_pro")}
        for stage in ("tower", "merged", "projected"):
            points = [next(c[metric] for c in models[m]["cells"] if c["stage"] == stage and c["layer_index"] == 27) * scale
                      for m in ("moonvit_v2", "kimi_k31_pro")]
            lines.append(f"| Matched {task} | {stage} | {points[0]:.3f} | {points[1]:.3f} |")
    template = template.replace("GEOMETRY_MATRIX_TABLE", "\n".join(lines))
    lines = ["| Dataset / metric | Stage | Fresh K3 | Pro |", "| --- | --- | ---: | ---: |"]
    for dataset, metric in (("navi", "recall_2cm"), ("scannet", "recall_10px"), ("spair", "pck_0.1")):
        models = {m: read(ROOT / f"measurements/correspondence/{m}_correspondence_{dataset}.json")
                  for m in ("moonvit_v2", "kimi_k31_pro")}
        for stage in ("tower", "merged", "projected"):
            points = [100 * next(c[metric] for c in models[m]["cells"] if c["stage"] == stage)
                      for m in ("moonvit_v2", "kimi_k31_pro")]
            lines.append(f"| {dataset} / {metric}, % | {stage} | {points[0]:.3f} | {points[1]:.3f} |")
    template = template.replace("CORRESPONDENCE_TABLE", "\n".join(lines))
    template = template.replace("REGRESSION_STATUS", f"The final CPU regression suite passed {tests['passed_tests']} tests and skipped\n{tests['skipped_tests']} weight-dependent cases under `VTB_SKIP_WEIGHTS=1`.\n[The test record](final-regression-tests.json) and [full output](final-regression-tests.txt)\nretain the command, runtime and source hashes. Independent real-weight identity\nand reference-forward evidence remain separate from the offline suite.")
    corrected = ROOT / "inference-smoke/corrected-publication/smoke/summary.json"
    if corrected.exists() and read(corrected)["passed"]:
        gate = read(corrected)
        assert all(row["exit_code"] == 0 for row in gate["runs"].values())
        template = template.replace("CORRECTED_INFERENCE_STATUS", f"The corrected published example passed from a new clean HF download at\nimmutable `{gate['revision']}`. Standalone inference\nexited zero in {gate['runs']['standalone']['seconds']:.6f} s; a separate all-Stage\ncomparison exited zero in {gate['runs']['numerical']['seconds']:.6f} s. Tower, merged\nand projected features are bit-exact with the bench adapter in CUDA BF16\nbatch 1 on physical GPU0, logical cuda:0, with 2 CPU threads.\n[The corrected published proof](inference-smoke/corrected-publication/smoke/summary.json)\nrecords both commands, the real image, actual shapes and immutable revision.\n[The GPU release](inference-smoke/corrected-publication/smoke/gpu0-release.json)\nverifies idle memory, no CUDA PIDs and lock reacquisition. A post-success CPU\nwrapper descriptor-shutdown error is retained separately in\n[the cleanup record](inference-smoke/corrected-publication/smoke/wrapper-cleanup.json);\nboth model checks passed before that cleanup. No numerical gate was loosened.\nThe original approximate-GELU functional proof remains historical evidence.")
    else:
        template = template.replace("CORRECTED_INFERENCE_STATUS", "Corrected example publication and a new fresh-download numerical CUDA proof\nremain pending external owner coordination and an explicit smoke reservation.\nThe original published functional pass is retained. Local corrected-output\nagreement is not described as a corrected published-snapshot pass.")
    assert not any(token in template for token in ("FINAL_INVENTORY_STATUS", "_TABLE", "REGRESSION_STATUS", "CORRECTED_INFERENCE_STATUS"))
    shutil.copy2(PUBLIC / "README.md", PUBLIC / "history-through-primary-completion.md")
    (PUBLIC / "README.md").write_text(template)
    index_path = Path("results/index.json")
    index = read(index_path)
    index["cohorts"]["kimi_k31_pro_checkpoint"]["description"] = "Matched vision-architecture checkpoint comparison against fresh Kimi K3 controls. Complete 286 primary cells and 102 paired comparisons, independent content checks, exact interval replays and final figures; production BF16 backend limitation retained."
    section = index["experiments"]["kimi_k31_pro_checkpoint"]
    section["protocol"] = "Matched vision-architecture checkpoint comparison at 448px; all ten recognition and geometry slices, both capacities and readouts, full label budgets, isolated standardized diagnostics, 13 perturbations, all three correspondence datasets, KITTI and throughput. Seed-0 LR search followed by selected-rate three-seed retraining; paired controls share actual devices and settings."
    section["status"] = "Complete benchmark inventory of 286 primary cells and 102 paired comparisons; actual content and exact interval replay verified. Production official TORCH_SDPA BF16 remains failed; accepted mathematical-reference conversion parity unchanged. Original 5311 published functional gate passed; see evidence for the separately tracked exact-GELU example correction."
    section["measurements"].update(
        full_primary_content_audit="diagnostics/kimi-k31-pro-checkpoint/independent-content-audits/completed-primary-independent-content-audit.json",
        all_paired_replay="diagnostics/kimi-k31-pro-checkpoint/independent-content-audits/all-paired-intervals-independent-replay.json",
        paired_comparison_report="diagnostics/kimi-k31-pro-checkpoint/paired-comparisons.md",
        paired_relative_depth_figure="diagnostics/kimi-k31-pro-checkpoint/matched-sibling-relative-depth-intervals.png",
        final_regression="diagnostics/kimi-k31-pro-checkpoint/final-regression-tests.json")
    section["measurements"]["paired_comparison_report_generator"] = "scripts/summarize_kimi_checkpoint_pairs.py"
    section["measurements"]["corrected_published_example"] = "diagnostics/kimi-k31-pro-checkpoint/inference-smoke/corrected-publication/smoke/summary.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print("Final benchmark documents assembled from audited286/102 and actual passing test results")


if __name__ == "__main__":
    main()
