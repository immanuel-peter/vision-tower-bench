"""Build a standalone MiniMax-M3 vision repository."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from vtb.adapters.minimax_m3 import (
    MERGE_MLP_PREFIX,
    PROJECTOR_PREFIX,
    PROJECTOR_SHARDS,
    SOURCE_REPO,
    SOURCE_REVISION,
    TOWER_PREFIX,
    VISION_SHARD,
    load_source_parts,
    remap_projector,
    remap_tower,
)
from vtb.shards import load_prefixed

TOWER_TENSORS = 515
PROJECTOR_TENSORS = 8


def export(out: Path) -> None:
    tower_weights = remap_tower(
        load_prefixed(SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION)
    )
    if len(tower_weights) != TOWER_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} Tower tensors, read {len(tower_weights)}")
    projector_weights = remap_projector(
        load_prefixed(SOURCE_REPO, list(PROJECTOR_SHARDS), PROJECTOR_PREFIX, revision=SOURCE_REVISION),
        load_prefixed(SOURCE_REPO, list(PROJECTOR_SHARDS), MERGE_MLP_PREFIX, revision=SOURCE_REVISION),
    )
    if len(projector_weights) != PROJECTOR_TENSORS:
        raise SystemExit(f"expected {PROJECTOR_TENSORS} Projector tensors, read {len(projector_weights)}")

    tower, projector = load_source_parts()
    out.mkdir(parents=True, exist_ok=True)
    tower.config.architectures = ["MiniMaxM3VLVisionModel"]
    tower.save_pretrained(out, safe_serialization=True)
    save_file(projector.state_dict(), out / "projector.safetensors", metadata={"format": "pt"})
    (out / "projector_config.json").write_text(
        json.dumps(
            {
                "input_size": projector.linear_1.in_features,
                "hidden_size": projector.linear_1.out_features,
                "output_size": projector.merge_linear_2.out_features,
                "merged_hidden_size": projector.merge_linear_1.in_features,
            },
            indent=2,
        )
        + "\n"
    )
    preprocessor = json.loads(
        Path(hf_hub_download(SOURCE_REPO, "preprocessor_config.json", revision=SOURCE_REVISION)).read_text()
    )
    preprocessor["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(json.dumps(preprocessor, indent=2) + "\n")
    shutil.copyfile(
        hf_hub_download(SOURCE_REPO, "LICENSE", revision=SOURCE_REVISION),
        out / "LICENSE",
    )
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/MiniMax-M3-Vision"))
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
