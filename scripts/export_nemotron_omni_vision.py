"""Build a standalone C-RADIOv4-H vision repository."""

import argparse
from pathlib import Path

from vtb.adapters.nemotron_omni import (
    PROJECTOR_PREFIX,
    SOURCE_REPO,
    SOURCE_REVISION,
    TOWER_PREFIX,
    VISION_SHARD,
    load_source_parts,
)
from vtb.bundle import ROOT, expect_tensors, print_sizes, write_bundle
from vtb.shards import load_prefixed

TOWER_TENSORS = 390
PROJECTOR_TENSORS = 3


def export(out: Path) -> None:
    expect_tensors(load_prefixed(SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION), TOWER_TENSORS, "Tower")
    expect_tensors(load_prefixed(SOURCE_REPO, [VISION_SHARD], PROJECTOR_PREFIX, revision=SOURCE_REVISION), PROJECTOR_TENSORS, "Projector")

    radio, projector = load_source_parts()
    write_bundle(
        out,
        weights={"model.safetensors": radio.state_dict(), "projector.safetensors": projector.state_dict()},
        configs={
            "config.json": {"architectures": ["RADIOModel"], "source": SOURCE_REPO, "revision": SOURCE_REVISION},
            "projector_config.json": {"type": "internvl_mlp1", "vit_hidden": 1280, "projector_hidden": 20480, "llm_hidden": 2688},
        },
        license=ROOT / "hf/C-RADIOv4-H-3-Omni/LICENSE",
    )
    print_sizes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "hf/C-RADIOv4-H-3-Omni")
    args = parser.parse_args()
    export(args.out)
    print(f"\nUpload with:\n  hf upload immanuelpeter/C-RADIOv4-H-3-Omni {args.out}")


if __name__ == "__main__":
    main()
