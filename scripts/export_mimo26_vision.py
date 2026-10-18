"""Build standalone MiMo 2.6 Pro and Flash vision repositories from the pinned parents."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from vtb.adapters.mimo26 import CHECKPOINTS, ROOT, load_model, source_json, source_weights

MODELING_SOURCE = ROOT / "vtb/models/mimo26.py"


def export(name: str, out: Path | None = None) -> Path:
    checkpoint = CHECKPOINTS[name]
    source = checkpoint.release_dir
    out = out or source
    model = load_model(checkpoint, torch.bfloat16, bundle=Path("/nonexistent-mimo-bundle"))
    original = source_weights(checkpoint)
    for key, tensor in original.items():
        if not torch.equal(model.state_dict()["visual." + key], tensor):
            raise ValueError(f"export changed {key}")
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() != source.resolve():
        shutil.copytree(source, out, ignore=shutil.ignore_patterns("*.safetensors"), dirs_exist_ok=True)
    model.config.architectures = ["MiMoVisionModel"]
    model.config.dtype = torch.bfloat16
    model.config.auto_map = {
        "AutoConfig": "modeling_mimo_vision.MiMoVisionConfig",
        "AutoModel": "modeling_mimo_vision.MiMoVisionModel",
    }
    model.config.save_pretrained(out)
    save_file(model.state_dict(), out / "model.safetensors", metadata={"format": "pt"})
    shutil.copyfile(MODELING_SOURCE, out / "modeling_mimo_vision.py")
    processor = source_json(checkpoint, "preprocessor_config.json")
    processor["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(json.dumps(processor, indent=2) + "\n")
    manifest = {
        "source_repo": checkpoint.repo, "source_revision": checkpoint.revision,
        "tensors": len(original), "bytes": sum(t.nbytes for t in original.values()),
        "tensor_sha256": {
            key: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for key, t in original.items()
        },
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{name}: {manifest['tensors']} tensors, {manifest['bytes']} bytes -> {out}")
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
