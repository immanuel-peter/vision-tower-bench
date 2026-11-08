"""Build the standalone PixtralViT-Large-4 repository from the pinned Mistral Large 4 parent.

Only the 276 vision tensors are read, by byte range, from the shard that the
pinned consolidated index names. Tensors are renamed to HF Pixtral and Mistral3
names; ``wq`` and ``wk`` rows are reordered per head from Mistral's interleaved
rotary layout to HF's rotate-half layout. Values are never cast or rounded. The bundle ships vtb/models/pixtral_large4.py
as modeling_pixtral_large4.py, the forward in the parent's numeric order,
loaded through auto_map with trust_remote_code.
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

MODELING_SOURCE = ml4.ROOT / "vtb/models/pixtral_large4.py"
MODELING_FILE = "modeling_pixtral_large4.py"
MANIFEST = "export_manifest.json"
BUNDLE_FILES = (
    "README.md",
    ".gitattributes",
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
        "vision_config": ml4.projector_config(source).vision_config.to_dict(),
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
        "source_shards": list(shards),
        "source_shard_sha256": {shard: lfs[shard] for shard in shards},
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def distributed_files(out: Path) -> list[Path]:
    """Every file the release distributes, except the manifest itself."""
    return sorted(
        path for path in out.rglob("*")
        if path.is_file() and path.name != MANIFEST and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )


def finish_manifest(out: Path, base: dict) -> dict:
    """Add the SHA256 of every distributed file; the manifest cannot hash itself."""
    payload = dict(base)
    payload["files_sha256"] = {str(path.relative_to(out)): file_sha256(path) for path in distributed_files(out)}
    payload["hash_excludes"] = [MANIFEST]
    (out / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_bundle(native: dict[str, torch.Tensor], source: dict, processor: dict, license_path: Path, out: Path,
                 shards: list[str]) -> None:
    tower_state, projector_state = ml4.convert(native)
    if any(t.dtype != torch.bfloat16 for t in [*tower_state.values(), *projector_state.values()]):
        raise ValueError("expected BF16 tensors")
    out.mkdir(parents=True, exist_ok=True)
    config = ml4.vision_config(source)
    config.architectures = ["PixtralLarge4VisionModel"]
    config.dtype = "bfloat16"
    config.auto_map = {
        "AutoConfig": "modeling_pixtral_large4.PixtralLarge4VisionConfig",
        "AutoModel": "modeling_pixtral_large4.PixtralLarge4VisionModel",
    }
    config.save_pretrained(out)
    shutil.copyfile(MODELING_SOURCE, out / MODELING_FILE)
    shutil.copyfile(ml4.ROOT / "LICENSE", out / "LICENSE-CODE")
    save_file({k: v.contiguous() for k, v in tower_state.items()}, out / "model.safetensors", metadata={"format": "pt"})
    save_file(
        {k: v.contiguous() for k, v in projector_state.items()}, out / "projector.safetensors", metadata={"format": "pt"}
    )
    (out / "projector_config.json").write_text(json.dumps(projector_settings(source), indent=2) + "\n")
    image_processor = dict(processor["image_processor"])
    image_processor["processor_class"] = "AutoImageProcessor"
    (out / "preprocessor_config.json").write_text(json.dumps(image_processor, indent=2) + "\n")
    shutil.copyfile(license_path, out / "LICENSE")
    finish_manifest(out, manifest(native, shards))


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
