"""Build a standalone Nemotron Omni C-RADIOv4-H vision repository."""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from vtb.adapters.nemotron_omni import (
    PROJECTOR_PREFIX,
    SOURCE_REPO,
    SOURCE_REVISION,
    TOWER_PREFIX,
    VISION_SHARD,
    load_source_parts,
)
from vtb.shards import load_prefixed

TOWER_TENSORS = 390
PROJECTOR_TENSORS = 3


def export(out: Path) -> None:
    tower_weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION
    )
    if len(tower_weights) != TOWER_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} Tower tensors, read {len(tower_weights)}")
    projector_weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], PROJECTOR_PREFIX, revision=SOURCE_REVISION
    )
    if len(projector_weights) != PROJECTOR_TENSORS:
        raise SystemExit(f"expected {PROJECTOR_TENSORS} Projector tensors, read {len(projector_weights)}")

    radio, projector = load_source_parts()
    out.mkdir(parents=True, exist_ok=True)
    save_file(radio.state_dict(), out / "model.safetensors", metadata={"format": "pt"})
    save_file(projector.state_dict(), out / "projector.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(
        json.dumps({"architectures": ["RADIOModel"], "source": SOURCE_REPO, "revision": SOURCE_REVISION}, indent=2)
        + "\n"
    )
    (out / "projector_config.json").write_text(
        json.dumps({"type": "internvl_mlp1", "vit_hidden": 1280, "projector_hidden": 20480, "llm_hidden": 2688}, indent=2)
        + "\n"
    )
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/Nemotron-Omni-Vision"))
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
