"""Build a standalone Gemma 4 31B vision repository."""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, Gemma4VisionModel

from vtb.adapters.gemma4 import (
    PROJECTOR_PREFIX,
    SOURCE_REPO,
    SOURCE_REVISION,
    TOWER_PREFIX,
    VISION_SHARD,
    Projector,
    projector_settings,
)
from vtb.bundle import ROOT, expect_tensors, print_sizes, write_bundle
from vtb.shards import load_prefixed

TOWER_TENSORS = 355
PROJECTOR_TENSORS = 1


def load_source_parts(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Gemma4VisionModel, Projector]:
    config = AutoConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION)
    tower = Gemma4VisionModel._from_config(config.vision_config, dtype=dtype)
    tower_weights = expect_tensors(load_prefixed(SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION), TOWER_TENSORS, "Tower")
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in tower_weights.items()})

    projector = Projector(projector_settings(config.vision_config, config.text_config.hidden_size))
    projector_weights = expect_tensors(
        load_prefixed(SOURCE_REPO, [VISION_SHARD], PROJECTOR_PREFIX, revision=SOURCE_REVISION), PROJECTOR_TENSORS, "Projector"
    )
    projector.load_state_dict({name: tensor.to(dtype) for name, tensor in projector_weights.items()})
    return tower.eval(), projector.to(dtype).eval()


def projector_config(projector: Projector) -> dict:
    return {
        "input_size": projector.embedding_projection.in_features,
        "output_size": projector.embedding_projection.out_features,
        "norm_eps": projector.embedding_pre_projection_norm.eps,
        "torch_dtype": str(projector.embedding_projection.weight.dtype).removeprefix("torch."),
    }


def export(out: Path) -> None:
    tower, projector = load_source_parts()
    processor = json.loads(Path(hf_hub_download(SOURCE_REPO, "processor_config.json", revision=SOURCE_REVISION)).read_text())
    write_bundle(
        out,
        tower=tower,
        architecture="Gemma4VisionModel",
        weights={"projector.safetensors": projector.state_dict()},
        configs={"projector_config.json": projector_config(projector)},
        preprocessor=processor["image_processor"],
        license=ROOT / "hf/Gemma4-31B-Vision/LICENSE",
    )
    print_sizes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "hf/Gemma4-31B-Vision")
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
