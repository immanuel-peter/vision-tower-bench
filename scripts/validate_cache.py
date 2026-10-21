#!/usr/bin/env python3
"""Validate a finished feature cache and write its complete.json.

The readouts refuse caches without this record, so a crashed or partial
extraction is never mistaken for a finished one.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from vtb import cache


def code_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "vtb", "scripts"], cwd=root,
                           capture_output=True, text=True).stdout.strip()
    return {"commit": head, "dirty": bool(dirty)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="cache directory written by vtb.extract")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="prep manifest whose 'images' keys are the expected image IDs")
    ap.add_argument("--model", required=True, help="adapter key, e.g. qwen3_5")
    ap.add_argument("--stages", nargs="+", required=True, choices=("tower", "merged", "projected"))
    ap.add_argument("--layer", type=int, required=True, help="final Tower layer of every expected slice")
    ap.add_argument("--source", default="", help="weight source and revision, from the roster manifest")
    ap.add_argument("--log", type=Path, default=None, help="extraction log to reference")
    args = ap.parse_args()

    expected = sorted(json.loads(args.manifest.read_text())["images"])
    report = cache.validate(args.run, expected, [(stage, args.layer) for stage in args.stages])
    summary = json.loads((args.run / "summary.json").read_text()) if (args.run / "summary.json").exists() else {}
    info = {
        "model": args.model,
        "model_id": next(iter(report.values()))["model_id"],
        "source": args.source,
        "resolution": int(next(iter(report.values()))["resolution"]),
        "slices": report,
        "images": len(expected),
        "image_ids_sha256": hashlib.sha256("\n".join(expected).encode()).hexdigest(),
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "extract_summary": summary,
        "log": str(args.log) if args.log else None,
        "code": code_state(),
        "torch": torch.__version__,
        "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = cache.write_completion(args.run, info)
    print(f"{args.model}: {len(report)} slices, {len(expected)} images -> {path}")


if __name__ == "__main__":
    main()
