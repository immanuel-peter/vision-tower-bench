"""Download measurement payloads listed in results/index.json at the pinned dataset revision."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payloads(prefixes=()):
    files = json.loads((ROOT / "results/index.json").read_text())["payloads"]["files"]
    prefixes = [p.removeprefix("./") for p in prefixes]
    return {path: entry for path, entry in files.items() if not prefixes or any(path.startswith(p) for p in prefixes)}


def absent(prefixes=()):
    return sorted(path for path in payloads(prefixes) if not (ROOT / path).is_file())


def missing(prefixes=()):
    return sorted(path for path, entry in payloads(prefixes).items() if not (ROOT / path).is_file() or sha256(ROOT / path) != entry["sha256"])


def download(url, handle, attempts=8):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url) as response:
                handle.seek(0)
                handle.truncate()
                for block in iter(lambda: response.read(1 << 20), b""):
                    handle.write(block)
            return
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
            time.sleep(float(error.headers.get("Retry-After") or 2 ** attempt))


def fetch(path, entry):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        try:
            download(entry["url"], handle)
        except BaseException:
            os.unlink(handle.name)
            raise
    if sha256(Path(handle.name)) != entry["sha256"]:
        os.unlink(handle.name)
        raise ValueError(f"{path}: SHA256 mismatch")
    os.replace(handle.name, target)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefixes", nargs="*", help="only paths starting with these prefixes, e.g. results/recognition/")
    parser.add_argument("--check", action="store_true", help="report missing or modified files without downloading")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    files = payloads(args.prefixes)
    if not files:
        sys.exit(f"no payloads match {args.prefixes}")
    todo = missing(args.prefixes)
    if args.check:
        for path in todo:
            print(path)
        print(f"{len(files) - len(todo)}/{len(files)} payloads present and verified", file=sys.stderr)
        sys.exit(1 if todo else 0)
    with ThreadPoolExecutor(args.jobs) as pool:
        for count, path in enumerate(pool.map(lambda p: fetch(p, files[p]), todo), 1):
            print(f"[{count}/{len(todo)}] {path}", file=sys.stderr)
    print(f"{len(files)} payloads present and verified", file=sys.stderr)


if __name__ == "__main__":
    main()
