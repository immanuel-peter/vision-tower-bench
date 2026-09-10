"""Build a standalone Qwen3.8-27B vision repository."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, Qwen3_5VisionModel

from vtb.adapters.qwen3_5 import SOURCE_REPO, VISION_PREFIX, VISION_SHARD
from vtb.shards import load_prefixed

SOURCE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOWER_TENSORS = 333


def load_source_tower(dtype: torch.dtype = torch.bfloat16) -> Qwen3_5VisionModel:
    config = AutoConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION).vision_config
    tower = Qwen3_5VisionModel._from_config(config, dtype=dtype)
    weights = load_prefixed(SOURCE_REPO, [VISION_SHARD], VISION_PREFIX, revision=SOURCE_REVISION)
    if len(weights) != TOWER_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} Tower tensors, read {len(weights)}")
    tower.load_state_dict(weights)
    return tower.eval()


def write_bundle(
    tower: Qwen3_5VisionModel,
    preprocessor: Path,
    license_path: Path,
    out: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tower.config.architectures = ["Qwen3_5VisionModel"]
    tower.save_pretrained(out, safe_serialization=True)
    processor_config = json.loads(preprocessor.read_text())
    processor_config["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(
        json.dumps(processor_config, indent=2) + "\n"
    )
    shutil.copyfile(license_path, out / "LICENSE")


def export(out: Path) -> None:
    tower = load_source_tower()

    preprocessor = Path(
        hf_hub_download(SOURCE_REPO, "preprocessor_config.json", revision=SOURCE_REVISION)
    )
    license_path = Path(hf_hub_download(SOURCE_REPO, "LICENSE", revision=SOURCE_REVISION))
    write_bundle(tower, preprocessor, license_path, out)

    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/Qwen3.8-27B-Vision"))
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
