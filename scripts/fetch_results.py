"""Download the measurement payloads from the results dataset at the revision pinned in results/index.json."""
import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MARKER = ROOT / "results/payload-revision.txt"


def pinned():
    return json.loads((ROOT / "results/index.json").read_text())["payloads"]


def fetched():
    return MARKER.is_file() and MARKER.read_text().strip() == pinned()["revision"]


def fetch(prefixes=(), jobs=8):
    payloads = pinned()
    prefixes = [p.removeprefix("./") for p in prefixes] or ["results/", "docs/measurements/"]
    full_results = any(p.rstrip("/") == "results" for p in prefixes)
    if full_results and fetched():
        prefixes = [p for p in prefixes if not p.startswith("results")]
    # The dataset stores docs/measurements/ at measurements/.
    groups = {ROOT: [p + "*" for p in prefixes if not p.startswith("docs/")],
              ROOT / "docs": [p.removeprefix("docs/") + "*" for p in prefixes if p.startswith("docs/")]}
    for local_dir, patterns in groups.items():
        if patterns:
            snapshot_download(payloads["dataset"], repo_type="dataset", revision=payloads["revision"],
                              local_dir=local_dir, allow_patterns=patterns, max_workers=jobs)
    if full_results:
        MARKER.write_text(payloads["revision"] + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", nargs="*", help="only paths starting with these prefixes, e.g. results/recognition/")
    parser.add_argument("--check", action="store_true", help="exit nonzero unless results/ was fetched at the pinned revision")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if fetched() else 1)
    fetch(args.prefixes, args.jobs)
    print(f"payloads at {pinned()['dataset']}@{pinned()['revision']}", file=sys.stderr)


if __name__ == "__main__":
    main()
