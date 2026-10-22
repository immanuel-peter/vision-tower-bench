#!/usr/bin/env python3
"""Validate the AI-authored CountQA exemplar export and write results/counting/countqa/exemplars_ai.json.

This is a separate input protocol from the human one in ``countqa_annotate.py finalize``, which
writes ``box_source="human"`` and ``status="approved"`` and must never see this export. Here every
box keeps ``annotation_source="ai"``, ``box_source="ai"``, ``human_reviewed=false`` and
``approved_by=null``. The file status is ``ai_unreviewed``, which the human loader refuses.

Rows keep their export status:

``boxed``              one AI-selected exemplar; included in the AI-exemplar evaluation
``no_clean_instance``  excluded: the annotator found no complete, unoccluded instance
``needs_review``       excluded: unresolved target identity or instance boundary; never boxed

Results that use this file are an experimental, supplementary FSC-147-trained counting transfer
on a CountQA-derived subset, not official CountQA benchmark scores.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import countqa_annotate as annotate
from vtb import counting

AI_FORMAT = "vtb-countqa-exemplars-ai-v1"
AI_STATUS = "ai_unreviewed"
AI_PROTOCOL = "countqa-ai-exemplars-v1"
ANNOTATION_PROTOCOL = "user-authorized-ai-visual-exemplars-v1"
STATUSES = ("boxed", "no_clean_instance", "needs_review")
EXCLUSION = {"no_clean_instance": "no_clean_instance", "needs_review": "needs_review_unresolved"}
EVALUATION_LABEL = ("experimental, supplementary: FSC-147-trained frozen-feature counting transfer on a "
                    "CountQA-derived subset with AI-authored exemplar boxes; not official CountQA "
                    "benchmark scores and not comparable to published CountQA MLLM results")
LIMITATIONS = (
    ("FSC-147 to CountQA transfer: heads trained on centre crops read letterboxed whole images with padding "
     "and smaller objects."),
    ("Exemplars are AI-selected (gpt-6-astra) and not human-reviewed; box errors and target-identity errors "
     "are possible."),
    "Only questions with a clean exemplar are included; this may systematically exclude hard images.",
    ("Sensitivity to the choice of exemplar and stability of model rankings under alternative exemplars or "
     "protocol choices are not established."),
)


def _provenance_ok(record: dict, where: str) -> None:
    if (record.get("annotation_source") != "ai" or record.get("human_reviewed") is not False
            or record.get("approved_by") is not None):
        raise ValueError(f"{where}: needs annotation_source='ai', human_reviewed=false, approved_by=null")


def package_hash(package: Path, name: str) -> str | None:
    for line in (package / "SHA256SUMS.txt").read_text().splitlines():
        digest, _, path = line.partition("  ")
        if path.strip() == name:
            return digest
    return None


def convert(export: dict, export_sha: str, audit: dict, eligibility: dict, eligibility_sha: str,
            sheet_sha: str, manifest: dict, parent_qa: dict | None = None) -> dict:
    """Validate the AI export against the frozen subset and return the exemplars_ai payload."""
    if export.get("format") != annotate.EXPORT_FORMAT:
        raise ValueError(f"not a {annotate.EXPORT_FORMAT} export")
    if export.get("annotation_protocol") != ANNOTATION_PROTOCOL:
        raise ValueError(f"annotation protocol {export.get('annotation_protocol')!r} is not {ANNOTATION_PROTOCOL}")
    _provenance_ok(export, "export")
    if export.get("sheet_sha256") != sheet_sha or export.get("eligibility_sha256") != eligibility_sha:
        raise ValueError("export was made from a different sheet or eligibility")
    if manifest.get("eligibility_sha256") != eligibility_sha:
        raise ValueError("the prepared CountQA manifest was built from a different eligibility")
    if audit.get("export_sha256") != export_sha:
        raise ValueError("the audit describes a different export")
    _provenance_ok(audit, "audit")
    if audit.get("answers_used_for_decisions") is not False:
        raise ValueError("the audit does not state that answers were withheld from box decisions")

    kept = {entry["question_id"]: (image_id, entry) for image_id, entry in eligibility["kept"].items()}
    seen = [item["question_id"] for item in export["items"]]
    if len(seen) != len(set(seen)) or set(seen) != set(kept):
        raise ValueError("export items do not match the kept questions one to one")
    audited = {item["question_id"]: item for item in audit["items"]}
    if set(audited) != set(kept):
        raise ValueError("audit items do not match the kept questions one to one")

    items = {}
    for item in sorted(export["items"], key=lambda item: item["question_id"]):
        question_id = item["question_id"]
        image_id, entry = kept[question_id]
        prepared, record_audit = manifest["images"].get(image_id), audited[question_id]
        if item.get("image_id") != image_id or record_audit.get("image_id") != image_id:
            raise ValueError(f"{question_id}: image ID mismatch")
        if prepared is None or prepared["question_id"] != question_id:
            raise ValueError(f"{question_id}: not in the prepared CountQA manifest")
        if record_audit.get("image_sha256") != prepared["source_sha256"]:
            raise ValueError(f"{question_id}: annotated image bytes differ from the prepared source")
        size = (item.get("image_width"), item.get("image_height"))
        if size != (entry["width"], entry["height"]) or size != (prepared["width"], prepared["height"]):
            raise ValueError(f"{question_id}: annotated size {size} differs from the source image")
        _provenance_ok(item, question_id)
        status = item.get("status")
        if status not in STATUSES or record_audit.get("status") != status:
            raise ValueError(f"{question_id}: status {status!r} (audit {record_audit.get('status')!r})")
        record = {
            "image_id": image_id, "status": status, "box_px": None, "box_source_unit": None,
            "box_letterbox_unit": None, "box_source": None, "excluded_reason": EXCLUSION.get(status),
            "annotation_source": "ai", "annotation_model": item.get("annotation_model"),
            "human_reviewed": False, "approved_by": None, "confidence": item.get("confidence"),
            "annotated_at": item.get("annotated_at"), "note": item.get("note", ""),
            "review_status": record_audit.get("review_status"),
            "independent_ai_adjudication": "independent_adjudication" in record_audit,
        }
        if status == "boxed":
            if item.get("box_source") != "ai":
                raise ValueError(f"{question_id}: boxed row with box_source {item.get('box_source')!r}")
            box = annotate.validate_box(item["box_px"], entry["width"], entry["height"])
            unit = box / np.array([entry["width"], entry["height"]] * 2, dtype=np.float64)
            if not np.allclose(unit, item["box_source_unit"], atol=1e-9):
                raise ValueError(f"{question_id}: box_source_unit does not match box_px")
            frame = counting.letterbox_frame(entry["width"], entry["height"])
            letterbox, kept_fraction = counting.clip_box(frame.boxes(box)[0])
            if kept_fraction < 1.0 - 1e-9:
                raise AssertionError(f"{question_id}: a source box must lie inside the letterbox canvas")
            record |= {"box_px": [float(v) for v in box], "box_source_unit": [float(v) for v in unit],
                       "box_letterbox_unit": [round(float(v), 6) for v in letterbox], "box_source": "ai"}
        elif item.get("box_px") is not None or item.get("box_source") is not None:
            raise ValueError(f"{question_id}: {status} row carries a box")
        items[question_id] = record

    counts = Counter(record["status"] for record in items.values())
    declared = {k: export["counts"][k] for k in STATUSES}
    if declared != {k: counts[k] for k in STATUSES}:
        raise ValueError(f"export counts {declared} do not reconcile with its rows {dict(counts)}")
    models = sorted({record["annotation_model"] for record in items.values()})
    included = sorted(record["image_id"] for record in items.values() if record["status"] == "boxed")
    return {
        "format": AI_FORMAT, "status": AI_STATUS, "protocol": AI_PROTOCOL,
        "evaluation_label": EVALUATION_LABEL, "experimental": True, "supplementary": True,
        "official_countqa_benchmark": False,
        "annotation_protocol": export["annotation_protocol"], "annotation_source": "ai",
        "annotation_model": models[0] if len(models) == 1 else models, "annotator": export.get("annotator"),
        "human_reviewed": False, "approved_by": None, "annotation_revision": export.get("annotation_revision"),
        "coordinates": "box_px: continuous source pixels (x0, y0, x1, y1) of the released image; "
                       "box_source_unit: box_px / (w, h, w, h); box_letterbox_unit: "
                       "counting.letterbox_frame(w, h).boxes, rounded to 6 decimals",
        "source": {
            "export_sha256": export_sha, "exported_at": export.get("exported_at"), "tool": export.get("tool"),
            "first_pass_export_sha256": export.get("first_pass_export_sha256"),
            "adjudication_sha256": export.get("adjudication_sha256"),
            "task_data_sha256": audit.get("task_data_sha256"),
        },
        "review_coverage": {
            "inspected": audit.get("visual_inspection_count"),
            "boxed_overlay_rechecked_by_ai": audit.get("boxed_overlay_review_count"),
            "independent_ai_adjudication": audit.get("independent_adjudication_count"),
            "parent_ai_spot_check": len(parent_qa["items"]) if parent_qa else None,
            "human_reviewed": 0, "answers_used_for_decisions": False,
            "statement": audit.get("coverage_statement"),
        },
        "eligibility_sha256": eligibility_sha, "sheet_sha256": sheet_sha,
        "counts": {"questions": len(items), **{k: counts[k] for k in STATUSES}},
        "included_image_ids": len(included),
        "limitations": list(LIMITATIONS),
        "items": items,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", type=Path, required=True,
                    help="AI annotation package directory, e.g. "
                         "$VTB_WORK/data/countqa/annotations/ai-exemplars-20261022-v2")
    ap.add_argument("--manifest", type=Path, required=True, help="$VTB_WORK/data/countqa/manifest.json")
    ap.add_argument("--results", type=Path, default=annotate.RESULTS)
    ap.add_argument("--out", type=Path, default=None, help="default <results>/exemplars_ai.json")
    args = ap.parse_args(argv)
    export_path, audit_path = args.package / "countqa_exemplars_ai_export.json", args.package / "annotation_audit.json"
    data = export_path.read_bytes()
    export_sha = annotate.sha256_bytes(data)
    for path, digest in ((export_path, export_sha), (audit_path, annotate.sha256_file(audit_path))):
        if package_hash(args.package, path.name) != digest:
            raise SystemExit(f"{path.name} does not match the package SHA256SUMS.txt")
    eligibility, eligibility_sha = annotate.load_eligibility(args.results)
    sheet_sha = annotate.sha256_file(args.results / "annotation" / "sheet.csv")
    parent_path = args.package / "parent_qa_report.json"
    payload = convert(json.loads(data), export_sha, json.loads(audit_path.read_text()), eligibility,
                      eligibility_sha, sheet_sha, json.loads(args.manifest.read_text()),
                      json.loads(parent_path.read_text()) if parent_path.exists() else None)
    payload["source"] |= {"package": args.package.name,
                          "audit_sha256": annotate.sha256_file(audit_path),
                          "package_sha256sums_sha256": annotate.sha256_file(args.package / "SHA256SUMS.txt")}
    out = args.out or args.results / "exemplars_ai.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out}: {json.dumps(payload['counts'])}; status {payload['status']}, human_reviewed false")


if __name__ == "__main__":
    main()
