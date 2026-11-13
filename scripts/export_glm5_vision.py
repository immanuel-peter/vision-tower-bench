"""Build a standalone GLM-5.3-Flash vision repository."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from vtb.adapters.glm5 import (
    SOURCE_REPO,
    SOURCE_REVISION,
    VISION_PREFIX,
    VISION_SHARD,
    load_source_tower,
)
from vtb.bundle import ROOT, expect_tensors, print_sizes, write_bundle
from vtb.shards import load_prefixed

TOWER_TENSORS = 347


def export(out: Path) -> None:
    expect_tensors(load_prefixed(SOURCE_REPO, [VISION_SHARD], VISION_PREFIX, revision=SOURCE_REVISION), TOWER_TENSORS, "Tower")
    processor = json.loads(Path(hf_hub_download(SOURCE_REPO, "processor_config.json", revision=SOURCE_REVISION)).read_text())
    write_bundle(
        out,
        tower=load_source_tower(),
        architecture="Glm5NextVisionModel",
        preprocessor=processor["image_processor"],
        license=Path(hf_hub_download(SOURCE_REPO, "LICENSE", revision=SOURCE_REVISION)),
    )
    print_sizes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "hf/GLM-5.3-Flash-Vision")
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
