"""Build a standalone GLM-5.3-Flash vision repository."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import Glm5NextVisionModel

from vtb.adapters.glm5 import (
    SOURCE_REPO,
    SOURCE_REVISION,
    VISION_PREFIX,
    VISION_SHARD,
    load_source_tower,
)
from vtb.shards import load_prefixed

TOWER_TENSORS = 347


def write_bundle(tower: Glm5NextVisionModel, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tower.config.architectures = ["Glm5NextVisionModel"]
    tower.save_pretrained(out, safe_serialization=True)

    processor = json.loads(
        Path(hf_hub_download(SOURCE_REPO, "processor_config.json", revision=SOURCE_REVISION)).read_text()
    )
    image_processor = processor["image_processor"]
    image_processor["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(json.dumps(image_processor, indent=2) + "\n")

    license_path = Path(hf_hub_download(SOURCE_REPO, "LICENSE", revision=SOURCE_REVISION))
    shutil.copyfile(license_path, out / "LICENSE")


def export(out: Path) -> None:
    weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], VISION_PREFIX, revision=SOURCE_REVISION
    )
    if len(weights) != TOWER_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} Tower tensors, read {len(weights)}")
    tower = load_source_tower()
    write_bundle(tower, out)
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/GLM-5.3-Flash-Vision"))
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
