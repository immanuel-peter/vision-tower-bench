"""Build a standalone MiniMax-M3 vision repository."""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from vtb.adapters.minimax_m3 import (
    CHECKPOINTS,
    TOWER_PREFIX, PROJECTOR_PREFIX, MERGE_MLP_PREFIX,
    M3,
    load_source_parts,
    source_weights,
    source_json,
)
from vtb.bundle import expect_tensors, print_sizes, write_bundle

TOWER_TENSORS = 515
PROJECTOR_TENSORS = 8


def write_manifest(out: Path, checkpoint=M3) -> dict:
    from vtb.bundle_manifest import write_manifest as common_manifest
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    shards = {v for k,v in index.items() if k.startswith((TOWER_PREFIX, PROJECTOR_PREFIX, MERGE_MLP_PREFIX))}
    return common_manifest(out, checkpoint.repo, checkpoint.revision, shards, allow_mixed_source=True)



def export(out: Path, checkpoint=M3) -> None:
    tower_weights, projector_weights = source_weights(checkpoint)
    expect_tensors(tower_weights, TOWER_TENSORS, "Tower")
    expect_tensors(projector_weights, PROJECTOR_TENSORS, "Projector")

    tower, projector = load_source_parts(checkpoint=checkpoint)
    preprocessor = json.loads(
        Path(hf_hub_download(checkpoint.repo, "preprocessor_config.json", revision=checkpoint.revision)).read_text()
    )
    write_bundle(
        out,
        tower=tower,
        architecture="MiniMaxM3VLVisionModel",
        weights={"model.safetensors": tower_weights, "projector.safetensors": projector_weights},
        configs={
            "projector_config.json": {
                "input_size": projector.linear_1.in_features,
                "hidden_size": projector.linear_1.out_features,
                "output_size": projector.merge_linear_2.out_features,
                "merged_hidden_size": projector.merge_linear_1.in_features,
            }
        },
        preprocessor=preprocessor,
        license=Path(hf_hub_download(checkpoint.repo, "LICENSE", revision=checkpoint.revision)),
    )
    from vtb.bundle_manifest import copy_assets
    copy_assets(checkpoint.release_dir, out)
    write_manifest(out, checkpoint)
    print_sizes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default=M3.name)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.checkpoint]
    export(args.out or checkpoint.release_dir, checkpoint)


if __name__ == "__main__":
    main()
