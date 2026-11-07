"""Build a standalone MiniMax-M3 vision repository."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from vtb.adapters.minimax_m3 import (
    CHECKPOINTS,
    M3,
    load_source_parts,
    source_weights,
    source_json,
)

TOWER_TENSORS = 515
PROJECTOR_TENSORS = 8


def write_manifest(out: Path, checkpoint=M3) -> dict:
    index = source_json(checkpoint, "model.safetensors.index.json")["weight_map"]
    prefixes = ("vision_tower.vision_model.", "multi_modal_projector.", "patch_merge_mlp.")
    shards = sorted({shard for name, shard in index.items() if name.startswith(prefixes)})
    files = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "export_manifest.json"
        and "__pycache__" not in path.relative_to(out).parts
        and ".cache" not in path.relative_to(out).parts
        and path.suffix != ".pyc"
    }
    manifest = {
        "source_repo": checkpoint.repo, "source_revision": checkpoint.revision,
        "source_shards": shards, "tower_tensors": TOWER_TENSORS,
        "projector_tensors": PROJECTOR_TENSORS, "dtype": "bfloat16",
        "source_dtypes_preserved": True, "hash_excludes": ["export_manifest.json"],
        "files_sha256": files, "files": files,
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export(out: Path, checkpoint=M3) -> None:
    tower_weights, projector_weights = source_weights(checkpoint)
    if len(tower_weights) != TOWER_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} Tower tensors, read {len(tower_weights)}")
    if len(projector_weights) != PROJECTOR_TENSORS:
        raise SystemExit(f"expected {PROJECTOR_TENSORS} Projector tensors, read {len(projector_weights)}")

    tower, projector = load_source_parts(checkpoint=checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    tower.config.architectures = ["MiniMaxM3VLVisionModel"]
    tower.save_pretrained(out, safe_serialization=True)
    save_file(tower_weights, out / "model.safetensors", metadata={"format": "pt"})
    save_file(projector_weights, out / "projector.safetensors", metadata={"format": "pt"})
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
        Path(hf_hub_download(checkpoint.repo, "preprocessor_config.json", revision=checkpoint.revision)).read_text()
    )
    preprocessor["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(json.dumps(preprocessor, indent=2) + "\n")
    shutil.copyfile(
        hf_hub_download(checkpoint.repo, "LICENSE", revision=checkpoint.revision),
        out / "LICENSE",
    )
    write_manifest(out, checkpoint)
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default=M3.name)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.checkpoint]
    export(args.out or checkpoint.release_dir, checkpoint)


if __name__ == "__main__":
    main()
