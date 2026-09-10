"""Build a publishable MoonViT-V2 repository from the Kimi K3 shards."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from vtb.adapters.moonvit_v2 import (
    PROJECTOR_PREFIX,
    PROJECTOR_SHARD,
    SOURCE_REPO,
    TOWER_PREFIX,
    TOWER_SHARD,
)

TOWER_REPO = "AI4Industry/MoonViT-V2"
TOWER_TENSORS = 165
PROJECTOR_TENSORS = 3

# Loading the Tower needs AI4Industry's remote code, so the published repository carries it.
VENDORED = (
    "config.json",
    "configuration_moonvit_v2.py",
    "modeling_moonvit_v2.py",
    "image_processing_moonvit_v2.py",
    "preprocessor_config.json",
)


def read_prefix(shard: str, prefix: str) -> dict[str, torch.Tensor]:
    weights = load_file(hf_hub_download(SOURCE_REPO, shard))
    return {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}


def check_against_standalone(tower: dict[str, torch.Tensor]) -> None:
    standalone = load_file(hf_hub_download(TOWER_REPO, "model.safetensors"))
    if set(standalone) != set(tower):
        missing = set(standalone) ^ set(tower)
        raise SystemExit(f"tensor names differ from {TOWER_REPO}: {sorted(missing)[:5]}")
    for name, weight in standalone.items():
        if not torch.equal(weight, tower[name]):
            raise SystemExit(f"{name} differs from {TOWER_REPO}")


def projector_config(projector: dict[str, torch.Tensor]) -> dict:
    width, merged_width = projector["proj.2.weight"].shape
    return {
        "merge_kernel_size": [2, 2],
        "merge_type": "sd2_tpool",
        "input_size": merged_width,
        "hidden_size": merged_width,
        "output_size": width,
        "linear_bias": False,
        "activation_func": "gelu",
        "norm_type": "rmsnorm",
        "norm_eps": 1e-5,
        "torch_dtype": str(projector["proj.2.weight"].dtype).removeprefix("torch."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("hf/MoonViT-V2"))
    parser.add_argument("--skip-parity", action="store_true", help="skip the bit-exact check against " + TOWER_REPO)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tower = read_prefix(TOWER_SHARD, TOWER_PREFIX)
    projector = read_prefix(PROJECTOR_SHARD, PROJECTOR_PREFIX)
    if len(tower) != TOWER_TENSORS or len(projector) != PROJECTOR_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} and {PROJECTOR_TENSORS} tensors, read {len(tower)} and {len(projector)}")

    if not args.skip_parity:
        check_against_standalone(tower)

    save_file(tower, args.out / "model.safetensors", metadata={"format": "pt"})
    save_file(projector, args.out / "projector.safetensors", metadata={"format": "pt"})
    (args.out / "projector_config.json").write_text(json.dumps(projector_config(projector), indent=2) + "\n")

    for name in VENDORED:
        shutil.copyfile(hf_hub_download(TOWER_REPO, name), args.out / name)

    for path in sorted(args.out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")
    print(f"\nUpload with:\n  hf upload immanuelpeter/MoonViT-V2 {args.out}")


if __name__ == "__main__":
    main()
