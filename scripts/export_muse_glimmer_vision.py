"""Build a standalone Muse Glimmer Tower and Projector repository."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoConfig, MuseGlimmerVisionModel

from vtb.adapters.muse_glimmer import (
    ADAPTER_PREFIX,
    PROJECTION_PREFIX,
    SHARDS,
    SOURCE_REPO,
    TOWER_PREFIX,
    Projector,
)
from vtb.shards import load_prefixed

SOURCE_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
TOWER_TENSORS = 806
PROJECTOR_TENSORS = 3


def load_source_parts(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[MuseGlimmerVisionModel, Projector]:
    """Build both halves from the pinned parent checkpoint, which the release must match."""
    config = AutoConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION)
    tower = MuseGlimmerVisionModel._from_config(config.vision_config, dtype=dtype)
    tower_weights = load_prefixed(SOURCE_REPO, SHARDS, TOWER_PREFIX, revision=SOURCE_REVISION)
    if len(tower_weights) != TOWER_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} Tower tensors, read {len(tower_weights)}")
    tower.load_state_dict(tower_weights)

    projector = Projector(config)
    projector.adapter.load_state_dict(
        load_prefixed(SOURCE_REPO, SHARDS, ADAPTER_PREFIX, revision=SOURCE_REVISION)
    )
    projector.projection.load_state_dict(
        load_prefixed(SOURCE_REPO, SHARDS, PROJECTION_PREFIX, revision=SOURCE_REVISION)
    )
    if len(projector.state_dict()) != PROJECTOR_TENSORS:
        raise SystemExit(f"expected {PROJECTOR_TENSORS} Projector tensors")
    return tower.eval(), projector.to(dtype).eval()


def projector_config(projector: Projector) -> dict:
    return {
        "input_size": projector.adapter.fc1.in_features,
        "hidden_size": projector.adapter.fc1.out_features,
        "output_size": projector.projection.out_features,
        "linear_bias": False,
        "activation_func": "gelu",
        "norm_type": "rmsnorm_without_scale",
        "norm_eps": projector.norm.eps,
        "torch_dtype": str(projector.projection.weight.dtype).removeprefix("torch."),
    }


def write_bundle(
    tower: MuseGlimmerVisionModel,
    projector: Projector,
    processor_config: Path,
    license_path: Path,
    usage_policy: Path,
    out: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tower.config.architectures = ["MuseGlimmerVisionModel"]
    tower.save_pretrained(out, safe_serialization=True)
    save_file(projector.state_dict(), out / "projector.safetensors", metadata={"format": "pt"})
    (out / "projector_config.json").write_text(json.dumps(projector_config(projector), indent=2) + "\n")

    parent_processor = json.loads(processor_config.read_text())
    parent_processor["image_processor"]["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(
        json.dumps(parent_processor["image_processor"], indent=2) + "\n"
    )
    shutil.copyfile(license_path, out / "LICENSE")
    shutil.copyfile(usage_policy, out / "USAGE_POLICY.md")


def export(out: Path) -> None:
    tower, projector = load_source_parts()

    files = {
        name: Path(hf_hub_download(SOURCE_REPO, name, revision=SOURCE_REVISION))
        for name in ("processor_config.json", "LICENSE", "USAGE_POLICY.md")
    }
    write_bundle(
        tower,
        projector,
        files["processor_config.json"],
        files["LICENSE"],
        files["USAGE_POLICY.md"],
        out,
    )

    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/Muse-Glimmer-Vision"))
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
