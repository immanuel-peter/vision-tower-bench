"""Verify restored model bundles against the tracked release manifests.

MiMo: export_manifest.json must equal the tracked copy and name the pinned
source; every tensor in model.safetensors (stored with a ``visual.`` prefix)
must match the manifest's per-tensor SHA256 of the raw bytes, and the config
must equal the tracked config after defaults are applied.

Nemotron 3.5: config.json and inventory.json must equal the tracked copies and
name the pinned source; every tensor in model.safetensors, projector.safetensors
and (Super only) final_norm.safetensors must match the inventory's name, shape,
dtype and SHA256. Bolt must not carry a final_norm file.

DINOv3: vtb.adapters.dinov3.verify_bundle (manifest provenance, every listed
file's size and SHA256, and the pinned weight hash).

Licence and notice files are reported with their SHA256.

    uv run python scripts/verify_bundles.py --bundles $VTB_WORK/bundles \
        --out $VTB_WORK/manifests/bundle_verification.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from vtb.adapters import dinov3, mimo26, nemotron35

ROOT = Path(__file__).resolve().parents[1]
LICENCE_FILES = ("LICENSE", "LICENSE.md", "LICENSE-CODE", "NOTICE", "THIRD_PARTY_NOTICES.md", "USAGE_POLICY.md")
NEMOTRON_LICENSE_SHA256 = "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df"
NEMOTRON_COUNTS = {"tower": 391, "projector": 3, "final_norm": 2}
# Bundle directory name -> (kind, checkpoint)
BUNDLES = {
    "mimo26_pro": ("mimo26", mimo26.PRO),
    "mimo26_flash": ("mimo26", mimo26.FLASH),
    "nemotron35_super": ("nemotron35", nemotron35.SUPER),
    "nemotron35_bolt": ("nemotron35", nemotron35.BOLT),
    "dinov3": ("dinov3", None),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_inventory(path: Path) -> dict[str, dict]:
    """Name -> shape, dtype and SHA256 of the raw little-endian bytes."""
    result = {}
    with safe_open(path, framework="pt") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_sha256(tensor)}
    return result


def same_json(a: Path, b: Path) -> bool:
    return json.loads(a.read_text()) == json.loads(b.read_text())


def licences(bundle: Path) -> dict[str, str]:
    return {name: sha256_file(bundle / name) for name in LICENCE_FILES if (bundle / name).is_file()}


def files(bundle: Path) -> dict[str, dict]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(bundle.iterdir()) if path.is_file()
    }


def verify_mimo26(bundle: Path, checkpoint, tracked: Path) -> dict:
    checks = {}
    manifest = json.loads((bundle / "export_manifest.json").read_text())
    checks["export_manifest_equals_tracked"] = (bundle / "export_manifest.json").read_bytes() == (
        tracked / "export_manifest.json").read_bytes()
    checks["provenance"] = (manifest["source_repo"], manifest["source_revision"]) == (checkpoint.repo, checkpoint.revision)
    inventory = tensor_inventory(bundle / "model.safetensors")
    expected = {"visual." + name: digest for name, digest in manifest["tensor_sha256"].items()}
    checks["tensor_count"] = len(inventory) == manifest["tensors"] == mimo26.VISUAL_TENSORS
    checks["tensor_names"] = set(inventory) == set(expected)
    mismatched = sorted(name for name in expected if inventory.get(name, {}).get("sha256") != expected[name])
    checks["tensor_sha256"] = not mismatched
    nbytes = 0
    for entry in inventory.values():
        count = 1
        for size in entry["shape"]:
            count *= size
        nbytes += count * torch.empty((), dtype=getattr(torch, entry["dtype"].removeprefix("torch."))).element_size()
    checks["tensor_bytes"] = nbytes == manifest["bytes"]
    from vtb.models.mimo26 import MiMoVisionConfig

    def config(path):
        values = MiMoVisionConfig.from_pretrained(path, local_files_only=True).to_dict()
        for key in ("transformers_version", "dtype", "torch_dtype"):
            values.pop(key, None)
        return values

    checks["config_equals_tracked_after_defaults"] = config(bundle) == config(tracked)
    for name in ("modeling_mimo_vision.py", "preprocessor_config.json", "LICENSE", "LICENSE-CODE", "NOTICE"):
        checks[f"{name}_equals_tracked"] = (bundle / name).read_bytes() == (tracked / name).read_bytes()
    # The release script rewrote config.json key order and explicit defaults; informational only.
    notes = {"config_bytes_equal_tracked": (bundle / "config.json").read_bytes() == (tracked / "config.json").read_bytes()}
    return {"checks": checks, "notes": notes, "mismatched_tensors": mismatched[:20], "tensors": len(inventory),
            "source": f"{checkpoint.repo}@{checkpoint.revision}"}


def verify_nemotron35(bundle: Path, checkpoint, tracked: Path) -> dict:
    checks = {}
    checks["config_equals_tracked"] = same_json(bundle / "config.json", tracked / "config.json")
    checks["inventory_equals_tracked"] = same_json(bundle / "inventory.json", tracked / "inventory.json")
    config = json.loads((bundle / "config.json").read_text())
    checks["provenance"] = (config["source"], config["revision"]) == (checkpoint.repo, checkpoint.revision)
    checks["final_layernorm_flag"] = config["vision_final_layernorm"] == checkpoint.final_layernorm
    inventory = json.loads((bundle / "inventory.json").read_text())
    counts, mismatched = {}, []
    for part, filename in (("tower", "model.safetensors"), ("projector", "projector.safetensors"),
                           ("final_norm", "final_norm.safetensors")):
        expected = inventory.get(part, {})
        path = bundle / filename
        if not expected:
            checks[f"{part}_absent"] = not path.exists()
            counts[part] = 0
            continue
        actual = tensor_inventory(path)
        counts[part] = len(actual)
        checks[f"{part}_names"] = set(actual) == set(expected)
        bad = sorted(name for name in expected if actual.get(name) != expected[name])
        checks[f"{part}_shape_dtype_sha256"] = not bad
        mismatched += [f"{part}:{name}" for name in bad]
    checks["tensor_counts"] = counts == {**NEMOTRON_COUNTS, "final_norm": NEMOTRON_COUNTS["final_norm"] if checkpoint.final_layernorm else 0}
    checks["license_is_official_openmdw"] = sha256_file(bundle / "LICENSE") == NEMOTRON_LICENSE_SHA256
    checks["license_equals_tracked"] = (bundle / "LICENSE").read_bytes() == (tracked / "LICENSE").read_bytes()
    checks["third_party_notices_present"] = (bundle / "THIRD_PARTY_NOTICES.md").is_file()
    return {"checks": checks, "mismatched_tensors": mismatched[:20], "tensor_counts": counts,
            "source": f"{checkpoint.repo}@{checkpoint.revision}"}


def verify_dinov3(bundle: Path) -> dict:
    checks = {}
    try:
        manifest = dinov3.verify_bundle(bundle)
        checks["adapter_verify_bundle"] = True
    except (OSError, ValueError) as error:
        return {"checks": {"adapter_verify_bundle": False}, "error": str(error)}
    with safe_open(bundle / dinov3.MODEL_FILE, framework="pt") as handle:
        shapes = {name: list(handle.get_slice(name).get_shape()) for name in handle.keys()}
    checks["tensor_count"] = len(shapes) == manifest["tensor_count"]
    checks["tensor_shapes"] = shapes == manifest["tensor_shapes"]
    checks["license_present"] = (bundle / "LICENSE.md").is_file()
    return {"checks": checks, "source": f"{dinov3.SOURCE_REPO}@{dinov3.SOURCE_REVISION}", "tensors": len(shapes)}


def verify(name: str, bundle: Path) -> dict:
    kind, checkpoint = BUNDLES[name]
    if kind == "mimo26":
        result = verify_mimo26(bundle, checkpoint, checkpoint.release_dir)
    elif kind == "nemotron35":
        result = verify_nemotron35(bundle, checkpoint, checkpoint.release_dir)
    else:
        result = verify_dinov3(bundle)
    result.update({"bundle": str(bundle), "files": files(bundle), "licences": licences(bundle)})
    result["ok"] = all(result["checks"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundles", type=Path, required=True, help="directory holding one subdirectory per bundle")
    parser.add_argument("--only", nargs="+", choices=sorted(BUNDLES), default=sorted(BUNDLES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {name: verify(name, args.bundles / name) for name in args.only}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for name, result in report.items():
        failed = [key for key, value in result["checks"].items() if not value]
        print(f"{name:18s} {'OK' if result['ok'] else 'FAIL'}  failed checks: {failed or '-'}")
    sys.exit(0 if all(result["ok"] for result in report.values()) else 1)


if __name__ == "__main__":
    main()
