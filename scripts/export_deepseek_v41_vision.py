"""Build a standalone DeepSeek V4.1 Flash vision repository."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from vtb.adapters.deepseek_v41 import (
    ALIGNER_PREFIX,
    SOURCE_REPO,
    SOURCE_REVISION,
    TOWER_PREFIX,
    VISION_SHARD,
    load_source_parts,
)
from vtb.shards import load_prefixed

TOWER_TENSORS = 259
PROJECTOR_TENSORS = 4


def export(out: Path) -> None:
    tower_weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION
    )
    if len(tower_weights) != TOWER_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} Tower tensors, read {len(tower_weights)}")
    projector_weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], ALIGNER_PREFIX, revision=SOURCE_REVISION
    )
    if len(projector_weights) != PROJECTOR_TENSORS:
        raise SystemExit(f"expected {PROJECTOR_TENSORS} Aligner tensors, read {len(projector_weights)}")

    tower, aligner = load_source_parts()
    out.mkdir(parents=True, exist_ok=True)
    save_file(tower.state_dict(), out / "model.safetensors", metadata={"format": "pt"})
    save_file(aligner.state_dict(), out / "projector.safetensors", metadata={"format": "pt"})
    (out / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepSeekV41ViT"],
                "vision_n_layers": 32,
                "vision_dim": 1024,
                "vision_patch_size": 14,
                "vision_downsample_ratio": 3,
                "aligner_dim": 5120,
            },
            indent=2,
        )
        + "\n"
    )
    try:
        shutil.copyfile(
            hf_hub_download(SOURCE_REPO, "LICENSE", revision=SOURCE_REVISION),
            out / "LICENSE",
        )
    except Exception:
        pass
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/DeepSeek-V4.1-Vision"))
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
