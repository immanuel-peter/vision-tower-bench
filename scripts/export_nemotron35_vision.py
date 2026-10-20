"""Build standalone Nemotron Super and Nemotron Bolt vision repositories."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from vtb.adapters.nemotron35 import (
    CHECKPOINTS, CODE_CONFIG_REPO, CODE_CONFIG_REVISION, RADIO_CODE_REVISION,
    source_index, source_weights,
)


def export(name: str, out: Path | None = None) -> Path:
    checkpoint = CHECKPOINTS[name]
    source = checkpoint.release_dir
    out = out or source
    license_hash = hashlib.sha256((source / "LICENSE").read_bytes()).hexdigest()
    if license_hash != "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df":
        raise ValueError("bundle must include the unmodified official OpenMDW-1.1 license")
    weights = source_weights(checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() != source.resolve():
        shutil.copytree(source, out, ignore=shutil.ignore_patterns("*.safetensors"), dirs_exist_ok=True)
    for part, filename in (("tower", "model.safetensors"), ("projector", "projector.safetensors"), ("final_norm", "final_norm.safetensors")):
        if weights[part]:
            save_file(weights[part], out / filename, metadata={"format": "pt"})
    config = {
        "source": checkpoint.repo, "revision": checkpoint.revision,
        "code_config_source": CODE_CONFIG_REPO, "code_config_revision": CODE_CONFIG_REVISION,
        "radio_code_revision": RADIO_CODE_REVISION,
        "vision_final_layernorm": checkpoint.final_layernorm,
        "patch_size": 16, "num_layers": 32, "num_prefix_tokens": 10,
        "summary_idxs": [0, 1], "vit_hidden": 1280, "downsample_ratio": 0.5,
        "projector_hidden": 20480, "llm_hidden": 4096,
        "projector_activation": "relu2", "projector_norm_eps": 1e-5,
        "vision_final_layernorm_eps": 1e-6,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = source_index(checkpoint)
    inventory = {
        part: {
            key: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "sha256": hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
            for key, value in sorted(part_weights.items())
        } for part, part_weights in weights.items()
    }
    inventory["source_shards"] = sorted({shard for key, shard in index.items() if key.startswith(("vision_model.", "mlp1.", "vision_projector.vision_final_layernorm."))})
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"model": name, "out": str(out), "tensor_counts": {key: len(value) for key, value in weights.items()}}, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, required=True)
    parser.add_argument("--out", type=Path, default=None, help="defaults to the tracked hf/ release directory")
    args = parser.parse_args()
    out = export(args.model, args.out)
    print(f"\nUpload with:\n  hf upload {CHECKPOINTS[args.model].model_id} {out}")


if __name__ == "__main__":
    main()
