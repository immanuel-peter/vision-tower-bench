"""Render all required intervals after exact independent numerical replay."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path,
                        default=Path("results/diagnostics/kimi-k31-pro-checkpoint"))
    args = parser.parse_args()
    inventory_path = args.evidence / "completed-inventory.json"
    inventory = read(inventory_path)
    expected = read(args.evidence / "expected-inventory.json")
    audit_path = args.evidence / "independent-content-audits/all-paired-intervals-independent-replay.json"
    audit = read(audit_path)
    assert inventory["completed_primary_cells"] == 286
    assert inventory["completed_paired_cells"] == audit["comparison_cells"] == 102
    assert audit["passed"] and audit["all_intervals_replayed_exactly"]
    entries = {json.dumps(row["cell"], sort_keys=True): row for row in inventory["completed_paired"]}
    cells = expected["paired_comparison_cells"]
    assert len(entries) == len(cells) == 102
    assert set(entries) == {json.dumps(cell, sort_keys=True) for cell in cells}
    replayed = {row["sha256"] for row in audit["results"] if row["exact_numerical_replay"]}
    visibility_path = args.evidence / "spair-visibility-audit.json"
    visibility = read(visibility_path)["measurements"]
    selected_spair = len(visibility)
    empty_spair = sum(row["valid_keypoints"] == 0 for row in visibility)
    finite_spair = selected_spair - empty_spair
    rows = []

    def values(interval, scale):
        assert interval["confidence"] == .95 and interval["resamples"] == 10000
        assert interval["bootstrap_seed"] == 0
        result = [scale * interval[key] for key in ("point_estimate", "lower", "upper")]
        assert all(math.isfinite(value) for value in result)
        assert result[1] <= result[2]
        return result

    def formatted(numbers):
        point, low, high = numbers
        return f"{point:+.3f} [{low:+.3f}, {high:+.3f}]"

    for position, cell in enumerate(cells):
        entry = entries[json.dumps(cell, sort_keys=True)]
        assert entry["sha256"] in replayed
        source = Path("results/bootstrap") / Path(entry["source"]).name
        assert source.is_file() and sha(source) == entry["sha256"]
        payload = read(source)
        task = cell["task"]
        scale = 1 if task == "normal" else 100
        units = "degrees reduced" if task == "normal" else "percentage points"
        if task == "perturbation":
            condition = cell["condition"]
            selected = payload["clean"] if condition == "identity" else payload["conditions"][condition]
            interval, count = selected["difference"], selected["test_images"]
        elif task in {"depth", "normal", "kitti"}:
            interval, count = payload["first_advantage"], payload["protocol"]["test_images"]
        elif task == "recognition":
            interval, count = payload["difference"], payload["protocol"]["test_images"]
        elif "difference" in payload:
            interval, count = payload["difference"], payload["finite_pairs"]
        else:
            interval = payload["first_advantage"]
            count = interval["pairs"]
        assert count > 0
        if task == "spair":
            assert count == finite_spair
        numbers = values(interval, scale)
        macro = payload.get("class_macro_difference")
        rows.append(dict(index=position, cell=cell, point=numbers[0], lower=numbers[1], upper=numbers[2],
                         units=units, samples=count, source=str(source), sha256=entry["sha256"],
                         macro_interval=values(macro, 100) if macro else None,
                         sign="second minus first error" if task == "normal" else "first minus second score"))
    lines = ["# All 102 paired comparisons", "",
             "Every interval below was replayed exactly from its actual paired rows. "
             "Each source retains its own point estimate. No matrix point is substituted.", "",
             "Positive values favor the first side. Normal-error comparisons report second-minus-first "
             "error in degrees; all other rows report first-minus-second score in percentage points. "
             "Intervals are individual 95% intervals with 10,000 resamples and seed 0, without multiplicity adjustment.", "",
             f"SPair retains all {selected_spair:,} source selection IDs; only the {finite_spair:,} nonempty pairs enter its paired intervals. "
             f"The source-backed exclusion union lists all {empty_spair} empty-pair IDs and reasons. "
             "Class-macro intervals appear separately in the same three sibling-comparison rows.", "",
             "Production official vLLM TORCH_SDPA BF16 agreement remains failed. "
             "Accepted conversion parity uses the independent mathematical reference.", "",
             "| ID | Task / arm / condition | Comparison | Stages | N | First-side advantage [95% interval] | Class-macro interval | Source |",
             "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        cell = row["cell"]
        description = "/".join(str(cell[key]) for key in ("task", "arm", "readout", "condition") if cell.get(key))
        stages = f"{cell['first_stage']}:{cell.get('first_layer', 27)} vs {cell['second_stage']}:{cell.get('second_layer', 27)}"
        comparison = "Pro vs K3" if cell["comparison"] == "kimi_k31_pro-minus-moonvit_v2" else "Pro projected vs Tower"
        target = os.path.relpath(row["source"], args.evidence)
        lines.append(f"| {row['index']} | {description} | {comparison} | {stages} | {row['samples']} | "
                     f"{formatted([row['point'], row['lower'], row['upper']])} | "
                     f"{formatted(row['macro_interval']) if row['macro_interval'] else ''} | [JSON]({target}) |")
    output = args.evidence / "paired-comparisons.md"
    output.write_text("\n".join(lines) + "\n")
    output.with_suffix(".json").write_text(json.dumps(dict(
        comparison_cells=102, complete_expected_inventory=True,
        inventory_sha256=sha(inventory_path), numerical_replay_audit_sha256=sha(audit_path),
        source_visibility_audit_sha256=sha(visibility_path),
        generator_sha256=sha(Path(__file__)), rows=rows,
    ), indent=2) + "\n")
    print(f"Wrote {len(rows)} source-backed comparison rows to {output}")


if __name__ == "__main__":
    main()
