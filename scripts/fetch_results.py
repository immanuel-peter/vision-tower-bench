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


def snapshots(prefixes=(), **kwargs):
    payloads = pinned()
    prefixes = [p.removeprefix("./") for p in prefixes] or ["results/", "docs/measurements/"]
    # The dataset stores docs/measurements/ at measurements/.
    groups = {ROOT: [p + "*" for p in prefixes if not p.startswith("docs/")],
              ROOT / "docs": [p.removeprefix("docs/") + "*" for p in prefixes if p.startswith("docs/")]}
    return [snapshot_download(payloads["dataset"], repo_type="dataset", revision=payloads["revision"],
                              local_dir=local_dir, allow_patterns=patterns, **kwargs)
            for local_dir, patterns in groups.items() if patterns], prefixes


def fetch(prefixes=(), jobs=8):
    _, prefixes = snapshots(prefixes, max_workers=jobs)
    if any(p.rstrip("/") == "results" for p in prefixes):
        MARKER.write_text(pinned()["revision"] + "\n")


def stale(prefixes=()):
    results, _ = snapshots(prefixes, dry_run=True)
    return sorted(info.filename for infos in results for info in infos if info.will_download)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", nargs="*", help="only paths starting with these prefixes, e.g. results/recognition/")
    parser.add_argument("--check", action="store_true", help="list missing or out-of-date payloads without downloading")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.check:
        todo = stale(args.prefixes)
        print("\n".join(todo + [f"{len(todo)} payloads missing or out of date"]), file=sys.stderr)
        sys.exit(1 if todo else 0)
    fetch(args.prefixes, args.jobs)
    print(f"payloads at {pinned()['dataset']}@{pinned()['revision']}", file=sys.stderr)


if __name__ == "__main__":
    main()
