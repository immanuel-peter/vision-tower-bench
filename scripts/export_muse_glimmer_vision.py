"""Build a standalone Muse Glimmer Tower and Projector repository."""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, MuseGlimmerVisionModel

from vtb.adapters.muse_glimmer import (
    ADAPTER_PREFIX,
    PROJECTION_PREFIX,
    SHARDS,
    SOURCE_REPO,
    TOWER_PREFIX,
    Projector,
)
from vtb.bundle import ROOT, expect_tensors, print_sizes, write_bundle
from vtb.shards import load_prefixed

SOURCE_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
TOWER_TENSORS = 806
PROJECTOR_TENSORS = 3


def load_source_parts(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[MuseGlimmerVisionModel, Projector]:
    config = AutoConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION)
    tower = MuseGlimmerVisionModel._from_config(config.vision_config, dtype=dtype)
    tower_weights = expect_tensors(load_prefixed(SOURCE_REPO, SHARDS, TOWER_PREFIX, revision=SOURCE_REVISION), TOWER_TENSORS, "Tower")
    tower.load_state_dict(tower_weights)

    projector = Projector(config)
    projector.adapter.load_state_dict(
        load_prefixed(SOURCE_REPO, SHARDS, ADAPTER_PREFIX, revision=SOURCE_REVISION)
    )
    projector.projection.load_state_dict(
        load_prefixed(SOURCE_REPO, SHARDS, PROJECTION_PREFIX, revision=SOURCE_REVISION)
    )
    expect_tensors(projector.state_dict(), PROJECTOR_TENSORS, "Projector")
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


def export(out: Path) -> None:
    tower, projector = load_source_parts()
    files = {
        name: Path(hf_hub_download(SOURCE_REPO, name, revision=SOURCE_REVISION))
        for name in ("processor_config.json", "LICENSE", "USAGE_POLICY.md")
    }
    write_bundle(
        out,
        tower=tower,
        architecture="MuseGlimmerVisionModel",
        weights={"projector.safetensors": projector.state_dict()},
        configs={"projector_config.json": projector_config(projector)},
        preprocessor=json.loads(files["processor_config.json"].read_text())["image_processor"],
        license=files["LICENSE"],
        files={"USAGE_POLICY.md": files["USAGE_POLICY.md"]},
    )
    print_sizes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "hf/Muse-Glimmer-Vision")
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
