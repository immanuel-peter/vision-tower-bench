"""Build a standalone Tower and Projector repository from its pinned parent checkpoint.

Each adapter module that publishes a release defines ``export(out)``, or
``export(out, checkpoint)`` when one module serves several checkpoints.
"""

import argparse
from importlib import import_module
from pathlib import Path

from vtb.bundle import ROOT, print_sizes
from vtb.roster import BY_SLUG, Tower


def exporter(tower: Tower):
    adapter = tower.adapter()
    module = import_module(adapter.__module__)
    if not hasattr(module, "export"):
        raise SystemExit(f"{tower.slug} is loaded from its upstream release and has no export")
    checkpoint = getattr(adapter, "checkpoint", None)
    return (lambda out: module.export(out, checkpoint)) if checkpoint else module.export


def export(slug: str, out: Path | None = None) -> Path:
    tower = BY_SLUG[slug]
    out = out or ROOT / "hf" / tower.model_id.split("/")[-1]
    exporter(tower)(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tower", choices=sorted(BY_SLUG))
    parser.add_argument("--out", type=Path, help="defaults to the tracked hf/ release directory")
    args = parser.parse_args()
    out = export(args.tower, args.out)
    print_sizes(out)
    print(f"\nUpload with:\n  hf upload {BY_SLUG[args.tower].model_id} {out}")


if __name__ == "__main__":
    main()
