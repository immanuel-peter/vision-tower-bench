"""Build a publishable MoonViT-V2 repository from the Kimi K3 shards."""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vtb.adapters.moonvit_v2 import (
    CHECKPOINTS,
    K3,
    PROJECTOR_PREFIX,
    TOWER_PREFIX,
)
from vtb.bundle import ROOT, expect_tensors, print_sizes, write_bundle
from vtb.shards import load_prefixed

TOWER_REPO = "AI4Industry/MoonViT-V2"
CODE_REVISION = "c5acb88baf02a94e68ca8a225f7eebee152c1fc6"
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


def discover_shards(checkpoint) -> dict[str, list[str]]:
    path = hf_hub_download(checkpoint.repo, "model.safetensors.index.json", revision=checkpoint.revision)
    mapping = json.loads(Path(path).read_text())["weight_map"]
    result = {prefix: sorted({shard for name, shard in mapping.items() if name.startswith(prefix)})
              for prefix in (TOWER_PREFIX, PROJECTOR_PREFIX)}
    if not all(result.values()):
        raise ValueError("source index lacks Tower or Projector tensors")
    selected = {shard for group in result.values() for shard in group}
    if any(shard in selected and not name.startswith((TOWER_PREFIX, PROJECTOR_PREFIX))
           for name, shard in mapping.items()):
        raise ValueError("vision shards contain non-vision tensors; refusing a full LM download")
    return result


def read_parent(checkpoint, shards: list[str], prefix: str) -> dict[str, torch.Tensor]:
    combined = {}
    for shard in shards:
        weights = load_prefixed(checkpoint.repo, [shard], prefix, revision=checkpoint.revision)
        if combined.keys() & weights.keys():
            raise ValueError("duplicate source tensor")
        combined.update(weights)
    return combined


def check_against_standalone(tower: dict[str, torch.Tensor]) -> None:
    standalone = load_file(hf_hub_download(TOWER_REPO, "model.safetensors", revision=CODE_REVISION))
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
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="moonvit_v2")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    checkpoint = CHECKPOINTS[args.checkpoint]
    args.out = args.out or ROOT / "hf" / checkpoint.bundle

    shards = discover_shards(checkpoint)
    tower = expect_tensors(read_parent(checkpoint, shards[TOWER_PREFIX], TOWER_PREFIX), TOWER_TENSORS, "Tower")
    projector = expect_tensors(read_parent(checkpoint, shards[PROJECTOR_PREFIX], PROJECTOR_PREFIX), PROJECTOR_TENSORS, "Projector")

    if any(t.dtype != torch.bfloat16 for t in (*tower.values(), *projector.values())):
        raise ValueError("expected BF16 source tensors")
    if checkpoint == K3:
        check_against_standalone(tower)

    write_bundle(
        args.out,
        weights={"model.safetensors": tower, "projector.safetensors": projector},
        configs={"projector_config.json": projector_config(projector)},
        license=Path(hf_hub_download(checkpoint.repo, "LICENSE", revision=checkpoint.revision)),
        files={
            **{name: Path(hf_hub_download(TOWER_REPO, name, revision=CODE_REVISION)) for name in VENDORED},
            "LICENSE-CODE": Path(hf_hub_download(TOWER_REPO, "LICENSE", revision=CODE_REVISION)),
        },
    )
    for filename, prefix in (("model.safetensors", TOWER_PREFIX), ("projector.safetensors", PROJECTOR_PREFIX)):
        exported = load_file(args.out / filename)
        parent = read_parent(checkpoint, shards[prefix], prefix)
        if exported.keys() != parent.keys() or any(not torch.equal(exported[k], parent[k]) for k in parent):
            raise ValueError(f"export differs from pinned parent: {filename}")
    from vtb.bundle_manifest import copy_assets, write_manifest
    assets = ROOT / "hf" / checkpoint.bundle
    copy_assets(assets, args.out)
    write_manifest(args.out, checkpoint.repo, checkpoint.revision,
                   [name for group in shards.values() for name in group],
                   extra={"exact_parent_identity": True, "code_repo": TOWER_REPO, "code_revision": CODE_REVISION,
                          "forward_parity": "Independent math-reference BF16 exact; production official SDPA BF16 disagreement retained."})

    print_sizes(args.out)
    print(f"\nBundle: {checkpoint.model_id}. Publish only after independent forward parity.")


if __name__ == "__main__":
    main()
