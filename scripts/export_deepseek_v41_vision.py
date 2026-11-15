"""Build a standalone DeepSeek-ViT repository."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from vtb.adapters.deepseek_v41 import (
    ALIGNER_PREFIX,
    SOURCE_REPO,
    SOURCE_REVISION,
    TOWER_PREFIX,
    VISION_SHARD,
    load_source_parts,
)
from vtb.bundle import ROOT, expect_tensors, print_sizes, write_bundle
from vtb.shards import load_prefixed

TOWER_TENSORS = 259
PROJECTOR_TENSORS = 4


def export(out: Path) -> None:
    expect_tensors(load_prefixed(SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION), TOWER_TENSORS, "Tower")
    expect_tensors(load_prefixed(SOURCE_REPO, [VISION_SHARD], ALIGNER_PREFIX, revision=SOURCE_REVISION), PROJECTOR_TENSORS, "Aligner")

    tower, aligner = load_source_parts()
    write_bundle(
        out,
        weights={"model.safetensors": tower.state_dict(), "projector.safetensors": aligner.state_dict()},
        configs={
            "config.json": {
                "architectures": ["DeepSeekV41ViT"],
                "vision_n_layers": 32,
                "vision_dim": 1024,
                "vision_patch_size": 14,
                "vision_downsample_ratio": 3,
                "aligner_dim": 5120,
            }
        },
        license=Path(hf_hub_download(SOURCE_REPO, "LICENSE", revision=SOURCE_REVISION)),
    )
    print_sizes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "hf/DeepSeek-ViT")
    args = parser.parse_args()
    export(args.out)
    print(f"\nUpload with:\n  hf upload immanuelpeter/DeepSeek-ViT {args.out}")


if __name__ == "__main__":
    main()
