"""Build a standalone Gemma 4 31B vision repository."""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
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
from vtb.shards import load_prefixed

TOWER_TENSORS = 355
PROJECTOR_TENSORS = 1


def load_source_parts(
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Gemma4VisionModel, Projector]:
    config = AutoConfig.from_pretrained(SOURCE_REPO, revision=SOURCE_REVISION)
    tower = Gemma4VisionModel._from_config(config.vision_config, dtype=dtype)
    tower_weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], TOWER_PREFIX, revision=SOURCE_REVISION
    )
    if len(tower_weights) != TOWER_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} Tower tensors, read {len(tower_weights)}")
    tower.load_state_dict({name: tensor.to(dtype) for name, tensor in tower_weights.items()})

    projector = Projector(projector_settings(config.vision_config, config.text_config.hidden_size))
    projector_weights = load_prefixed(
        SOURCE_REPO, [VISION_SHARD], PROJECTOR_PREFIX, revision=SOURCE_REVISION
    )
    if len(projector_weights) != PROJECTOR_TENSORS:
        raise SystemExit(f"expected {PROJECTOR_TENSORS} Projector tensors, read {len(projector_weights)}")
    projector.load_state_dict({name: tensor.to(dtype) for name, tensor in projector_weights.items()})
    return tower.eval(), projector.to(dtype).eval()


def projector_config(projector: Projector) -> dict:
    return {
        "input_size": projector.embedding_projection.in_features,
        "output_size": projector.embedding_projection.out_features,
        "norm_eps": projector.embedding_pre_projection_norm.eps,
        "torch_dtype": str(projector.embedding_projection.weight.dtype).removeprefix("torch."),
    }


def write_bundle(tower: Gemma4VisionModel, projector: Projector, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tower.config.architectures = ["Gemma4VisionModel"]
    tower.save_pretrained(out, safe_serialization=True)
    save_file(projector.state_dict(), out / "projector.safetensors", metadata={"format": "pt"})
    (out / "projector_config.json").write_text(json.dumps(projector_config(projector), indent=2) + "\n")

    processor = json.loads(
        Path(hf_hub_download(SOURCE_REPO, "processor_config.json", revision=SOURCE_REVISION)).read_text()
    )
    image_processor = processor["image_processor"]
    image_processor["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(json.dumps(image_processor, indent=2) + "\n")


def export(out: Path) -> None:
    tower, projector = load_source_parts()
    write_bundle(tower, projector, out)
    for path in sorted(out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/Gemma4-31B-Vision"))
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
