"""Write standalone Tower bundles from the export scripts."""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def expect_tensors(weights: dict[str, torch.Tensor], count: int, label: str) -> dict[str, torch.Tensor]:
    if len(weights) != count:
        raise SystemExit(f"expected {count} {label} tensors, read {len(weights)}")
    return weights


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_bundle(
    out: Path,
    *,
    tower=None,
    architecture: str | None = None,
    weights: dict[str, dict[str, torch.Tensor]] | None = None,
    configs: dict[str, dict] | None = None,
    preprocessor: dict | None = None,
    license: Path | None = None,
    files: dict[str, Path] | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if tower is not None:
        tower.config.architectures = [architecture]
        tower.save_pretrained(out, safe_serialization=True)
    for name, state in (weights or {}).items():
        save_file(state, out / name, metadata={"format": "pt"})
    for name, config in (configs or {}).items():
        write_json(out / name, config)
    if preprocessor is not None:
        write_json(out / "preprocessor_config.json", {**preprocessor, "processor_class": "AutoImageProcessor"})
    copies = dict(files or {})
    if license is not None:
        copies["LICENSE"] = license
    for name, source in copies.items():
        if Path(source).resolve() != (out / name).resolve():
            shutil.copyfile(source, out / name)


def print_sizes(out: Path) -> None:
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"{path.stat().st_size / 1e6:9.2f} MB  {path.relative_to(out)}")
