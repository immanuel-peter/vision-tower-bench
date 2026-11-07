"""Build a standalone pinned Qwen3.8 vision repository."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5VisionModel

from vtb.adapters.qwen3_5 import CHECKPOINTS, QWEN27B, Checkpoint, load_source_tower as _load_source_tower

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

    manifest = {"source_repo": checkpoint.repo, "source_revision": checkpoint.revision,
                "tensor_count": len(tower.state_dict()), "out_hidden_size": checkpoint.out_hidden_size,
                "license": checkpoint.license}
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default=QWEN27B.name)
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.checkpoint]
    export(args.out or checkpoint.bundle, checkpoint)


if __name__ == "__main__":
    main()
