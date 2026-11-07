"""Build a publishable MoonViT-V2 repository from the Kimi K3 shards."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from vtb.adapters.moonvit_v2 import (
    CHECKPOINTS,
    K3,
    PROJECTOR_PREFIX,
    PROJECTOR_SHARD,
    SOURCE_REPO,
    TOWER_PREFIX,
    TOWER_SHARD,
)

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


def read_prefix(shard: str, prefix: str, checkpoint=K3) -> dict[str, torch.Tensor]:
    weights = load_file(hf_hub_download(checkpoint.repo, shard, revision=checkpoint.revision))
    return {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}


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
    args.out = args.out or Path("hf") / checkpoint.bundle
    args.out.mkdir(parents=True, exist_ok=True)

    shards = discover_shards(checkpoint)
    def read(prefix):
        combined = {}
        for shard in shards[prefix]:
            weights = read_prefix(shard, prefix, checkpoint)
            if combined.keys() & weights.keys():
                raise ValueError("duplicate source tensor")
            combined.update(weights)
        return combined
    tower, projector = read(TOWER_PREFIX), read(PROJECTOR_PREFIX)
    if len(tower) != TOWER_TENSORS or len(projector) != PROJECTOR_TENSORS:
        raise SystemExit(f"expected {TOWER_TENSORS} and {PROJECTOR_TENSORS} tensors, read {len(tower)} and {len(projector)}")

    if any(t.dtype != torch.bfloat16 for t in (*tower.values(), *projector.values())):
        raise ValueError("expected BF16 source tensors")
    if checkpoint == K3:
        check_against_standalone(tower)

    save_file(tower, args.out / "model.safetensors", metadata={"format": "pt"})
    save_file(projector, args.out / "projector.safetensors", metadata={"format": "pt"})
    (args.out / "projector_config.json").write_text(json.dumps(projector_config(projector), indent=2) + "\n")

    for name in VENDORED:
        shutil.copyfile(hf_hub_download(TOWER_REPO, name, revision=CODE_REVISION), args.out / name)
    shutil.copyfile(hf_hub_download(checkpoint.repo, "LICENSE", revision=checkpoint.revision), args.out / "LICENSE")
    shutil.copyfile(hf_hub_download(TOWER_REPO, "LICENSE", revision=CODE_REVISION), args.out / "CODE_LICENSE")
    for filename, prefix in (("model.safetensors", TOWER_PREFIX), ("projector.safetensors", PROJECTOR_PREFIX)):
        exported = load_file(args.out / filename)
        parent = read(prefix)
        if exported.keys() != parent.keys() or any(not torch.equal(exported[k], parent[k]) for k in parent):
            raise ValueError(f"export differs from pinned parent: {filename}")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "source_repo": checkpoint.repo, "source_revision": checkpoint.revision,
        "shards": shards, "tower_tensors": len(tower), "projector_tensors": len(projector),
        "dtype": "bfloat16", "exact_parent_identity": True,
        "code_repo": TOWER_REPO, "code_revision": CODE_REVISION,
        "files": {p.name: sha(p) for p in args.out.iterdir() if p.is_file() and p.name != "export_manifest.json"},
        "forward_parity": "pending independent parent validation; tensor identity is not forward parity",
    }
    (args.out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for path in sorted(args.out.iterdir()):
        print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.name}")
    print(f"\nBundle: {checkpoint.model_id}. Publish only after independent forward parity.")


if __name__ == "__main__":
    main()
