"""Build the standalone PixtralViT-Large-4 repository from the pinned Mistral Large 4 parent.

Only the 276 vision tensors are read, by byte range, from the shard that the
pinned consolidated index names. Tensors are renamed to HF Pixtral and Mistral3
names; ``wq`` and ``wk`` rows are reordered per head from Mistral's interleaved
rotary layout to HF's rotate-half layout. Values are never cast or rounded.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file

from vtb.adapters import mistral_large4 as ml4

BUNDLE_FILES = (
    "README.md",
    "NOTICE",
    "projector.py",
    "examples/inference.py",
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def projector_settings(source: dict) -> dict:
    """The Mistral3Config fields Mistral3MultiModalProjector reads."""
    text = source["text_config"]
    return {
        "vision_config": ml4.vision_config(source).to_dict(),
        "text_config": {"model_type": "mistral", "hidden_size": text["hidden_size"], "rms_norm_eps": text["rms_norm_eps"]},
        "spatial_merge_size": source["spatial_merge_size"],
        "projector_hidden_act": source["projector_hidden_act"],
        "multimodal_projector_bias": source["multimodal_projector_bias"],
        "vision_feature_layer": source["vision_feature_layer"],
    }


def manifest(native: dict[str, torch.Tensor], shards: list[str]) -> dict:
    info = HfApi().model_info(ml4.SOURCE_REPO, revision=ml4.SOURCE_REVISION, files_metadata=True)
    lfs = {item.rfilename: item.lfs.sha256 for item in info.siblings if item.lfs is not None}
    renamed = {name: ml4.remap_key(name) for name in sorted(native)}
    return {
        "source_repo": ml4.SOURCE_REPO,
        "source_revision": ml4.SOURCE_REVISION,
        "source_index": ml4.SOURCE_INDEX,
        "source_shards": {shard: lfs[shard] for shard in shards},
        "tensors": len(native),
        "tower_tensors": sum(name.startswith(ml4.TOWER_PREFIX) for name in native),
        "projector_tensors": sum(not name.startswith(ml4.TOWER_PREFIX) for name in native),
        "bytes": sum(t.nbytes for t in native.values()),
        "dtype": "bfloat16",
        "rope_permutation": {
            "tensors": sorted(name for name, target in renamed.items() if ml4.is_rotary(target)),
            "description": "per head, rows (2i, 2i + 1) move to (i, i + head_dim / 2); "
            "w.view(16, 64, 2, 2048).transpose(1, 2).reshape(2048, 2048)",
        },
        "renamed": renamed,
        "tensor_sha256": {name: tensor_sha256(native[name]) for name in sorted(native)},
    }


def write_bundle(native: dict[str, torch.Tensor], source: dict, processor: dict, license_path: Path, out: Path,
                 shards: list[str]) -> None:
    tower_state, projector_state = ml4.convert(native)
    if any(t.dtype != torch.bfloat16 for t in [*tower_state.values(), *projector_state.values()]):
        raise ValueError("expected BF16 tensors")
    out.mkdir(parents=True, exist_ok=True)
    config = ml4.vision_config(source)
    config.architectures = ["PixtralVisionModel"]
    config.dtype = "bfloat16"
    config.save_pretrained(out)
    save_file({k: v.contiguous() for k, v in tower_state.items()}, out / "model.safetensors", metadata={"format": "pt"})
    save_file(
        {k: v.contiguous() for k, v in projector_state.items()}, out / "projector.safetensors", metadata={"format": "pt"}
    )
    (out / "projector_config.json").write_text(json.dumps(projector_settings(source), indent=2) + "\n")
    image_processor = dict(processor["image_processor"])
    image_processor["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(json.dumps(image_processor, indent=2) + "\n")
    shutil.copyfile(license_path, out / "LICENSE")
    (out / "export_manifest.json").write_text(json.dumps(manifest(native, shards), indent=2) + "\n")


def export(out: Path) -> Path:
    native = ml4.source_weights()
    shards = sorted(set(ml4.source_index().values()))
    source = ml4.source_json("config.json")
    processor = ml4.source_json("processor_config.json")
    license_path = Path(hf_hub_download(ml4.SOURCE_REPO, "LICENSE", revision=ml4.SOURCE_REVISION))
    tracked = ml4.RELEASE_DIR
    if out.resolve() != tracked.resolve():
        for name in BUNDLE_FILES:
            if (tracked / name).is_file():
                (out / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tracked / name, out / name)
    write_bundle(native, source, processor, license_path, out, shards)
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"{path.stat().st_size / 1e6:10.2f} MB  {path.relative_to(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ml4.RELEASE_DIR)
    args = parser.parse_args()
    out = export(args.out)
    print(f"\nUpload with:\n  hf upload {ml4.MODEL_ID} {out}")


if __name__ == "__main__":
    main()
