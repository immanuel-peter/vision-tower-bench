"""Build a standalone pinned Qwen3.8 vision repository."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5VisionModel

from vtb.adapters.qwen3_5 import CHECKPOINTS, QWEN27B, Checkpoint, load_source_tower as _load_source_tower, source_json, VISION_PREFIX
from vtb.bundle import ROOT, print_sizes, write_bundle

SOURCE_REVISION = QWEN27B.revision
TOWER_TENSORS = 333


def load_source_tower(dtype: torch.dtype = torch.bfloat16, checkpoint: Checkpoint = QWEN27B) -> Qwen3_5VisionModel:
    return _load_source_tower(checkpoint, dtype)


def export(out: Path, checkpoint: Checkpoint = QWEN27B) -> None:
    tower = load_source_tower(checkpoint=checkpoint)

    preprocessor = Path(hf_hub_download(checkpoint.repo, "preprocessor_config.json", revision=checkpoint.revision))
    write_bundle(
        out,
        tower=tower,
        architecture="Qwen3_5VisionModel",
        preprocessor=json.loads(preprocessor.read_text()),
        license=Path(hf_hub_download(checkpoint.repo, "LICENSE", revision=checkpoint.revision)),
    )

    assets = ROOT / "hf" / checkpoint.model_id.split("/")[-1]
    for name in ("README.md", ".gitattributes", "examples/inference.py"):
        source, destination = assets / name, out / name
        if source.is_file() and source.resolve() != destination.resolve():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    write_export_manifest(out, checkpoint)
    print_sizes(out)


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
