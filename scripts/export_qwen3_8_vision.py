"""Build a standalone pinned Qwen3.8 vision repository."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5VisionModel
from safetensors import safe_open

from vtb.adapters.qwen3_5 import CHECKPOINTS, QWEN27B, Checkpoint, load_source_tower as _load_source_tower, source_json, VISION_PREFIX

SOURCE_REVISION = QWEN27B.revision
TOWER_TENSORS = 333


def load_source_tower(dtype: torch.dtype = torch.bfloat16, checkpoint: Checkpoint = QWEN27B) -> Qwen3_5VisionModel:
    return _load_source_tower(checkpoint, dtype)


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


def export(out: Path, checkpoint: Checkpoint = QWEN27B) -> None:
    tower = load_source_tower(checkpoint=checkpoint)

    preprocessor = Path(
        hf_hub_download(checkpoint.repo, "preprocessor_config.json", revision=checkpoint.revision)
    )
    license_path = Path(hf_hub_download(checkpoint.repo, "LICENSE", revision=checkpoint.revision))
    write_bundle(tower, preprocessor, license_path, out)

    assets = Path(__file__).resolve().parents[1] / "hf" / checkpoint.model_id.split("/")[-1]
    for name in ("README.md", ".gitattributes", "examples/inference.py"):
        source, destination = assets / name, out / name
        if source.is_file() and source.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    write_export_manifest(out, checkpoint)
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def write_export_manifest(out: Path, checkpoint: Checkpoint) -> None:
    from vtb.bundle_manifest import write_manifest
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    shards = {v for k,v in index.items() if k.startswith(VISION_PREFIX)}
    write_manifest(out, checkpoint.repo, checkpoint.revision, shards, merger_in_model=True,
                   extra={"out_hidden_size": checkpoint.out_hidden_size, "license": checkpoint.license})



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default=QWEN27B.name)
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.checkpoint]
    export(args.out or checkpoint.bundle, checkpoint)


if __name__ == "__main__":
    main()
