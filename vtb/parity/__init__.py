"""Check every Tower against the checkpoint it came from.

Each adapter module has a parity module of the same name here. Its ``validate``
returns a report with ``weights`` (release tensors against the pinned parent),
``forward`` (bench features against an independent reference) and ``passed``.
Run one with ``python -m vtb.parity <slug>``.
"""

import argparse
import ast
import hashlib
import json
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

from vtb.shards import cache_root


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def pinned_sources(repo: str, commit: str, sources: dict[str, str]) -> Path:
    """Fetch files from a GitHub commit into the shard cache, checked against their SHA-256."""
    root = cache_root().parent / "vtb-upstream" / f"{repo.replace('/', '__')}@{commit}"
    for name, digest in sources.items():
        path = root / name
        if not path.is_file():
            response = requests.get(
                f"https://raw.githubusercontent.com/{repo}/{commit}/{name}", timeout=60
            )
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"{path} does not match {repo}@{commit}")
    return root


def compile_nodes(
    path: Path, wanted: set[str], namespace: dict, within: str | None = None
) -> None:
    """Run the named top-level classes and functions of a source file in ``namespace``, without its imports or decorators. ``within`` reads them from that class instead."""
    body = ast.parse(path.read_text()).body
    if within is not None:
        body = next(n for n in body if getattr(n, "name", None) == within).body
    nodes = [node for node in body if getattr(node, "name", None) in wanted]
    missing = wanted - {node.name for node in nodes}
    if missing:
        raise ValueError(f"{sorted(missing)} not found in {path}")
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape:
        raise ValueError(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    a, e = (
        actual.detach().cpu().double().flatten(),
        expected.detach().cpu().double().flatten(),
    )
    delta = a - e
    return {
        "shape": list(actual.shape),
        "bit_exact": torch.equal(actual.detach().cpu(), expected.detach().cpu()),
        "max_abs": delta.abs().max().item(),
        "reference_max_abs": e.abs().max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / e.norm().clamp_min(1e-12)).item(),
        "cosine": max(-1.0, min(1.0, F.cosine_similarity(a, e, dim=0).item())),
        "finite": bool(torch.isfinite(actual).all()),
    }


def weights_match(
    release: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
    count: int | None = None,
    *,
    extra_ok: bool = False,
) -> dict:
    """Bit-exact release tensors in the parent's own dtype. ``extra_ok`` allows buffers the release rebuilds."""
    missing = sorted(set(parent) - set(release))
    extra = sorted(set(release) - set(parent))
    differ = sorted(
        name
        for name in set(parent) & set(release)
        if release[name].dtype != parent[name].dtype
        or not torch.equal(release[name], parent[name])
    )
    exact = (
        not missing
        and not differ
        and (extra_ok or not extra)
        and (count is None or len(parent) == count)
    )
    return {
        "tensors": len(parent),
        "expected": count,
        "bit_exact": exact,
        "missing": missing[:5],
        "extra": extra[:5],
        "differ": differ[:5],
    }


def reference_raster(
    packed: torch.Tensor, grid_thw: torch.Tensor, column_major: bool = False
) -> torch.Tensor:
    """Scatter packed 2x2 merge blocks back to raster order by coordinate, independent of adapter reshapes."""
    restored, start = [], 0
    for t, h, w in grid_thw.tolist():
        assert t == 1
        image = torch.empty(
            h, w, packed.shape[-1], device=packed.device, dtype=packed.dtype
        )
        outer = [(y, x) for y in range(0, h, 2) for x in range(0, w, 2)]
        if column_major:
            outer.sort(key=lambda coord: (coord[1], coord[0]))
        for y, x in outer:
            for dy in range(2):
                for dx in range(2):
                    image[y + dy, x + dx] = packed[start]
                    start += 1
        restored.append(image.flatten(0, 1))
    assert start == len(packed)
    return torch.stack(restored)


def fixed_images() -> list[Image.Image]:
    """A landscape and a portrait raster with structure at every scale, so order and crop errors show."""
    y, x = np.mgrid[:376, :613]
    image = np.stack(
        [x % 256, y % 256, (x // 16 * 31 + y // 16 * 17) % 256], -1
    ).astype("uint8")
    return [
        Image.fromarray(image),
        Image.fromarray(image[::-1, ::-1].copy()).rotate(90, expand=True),
    ]


def features(adapter, images: list[Image.Image]) -> dict[str, torch.Tensor]:
    inputs = adapter.collate([adapter.preprocess()(image) for image in images])
    ids = [str(i) for i in range(len(images))]
    return {
        f"{batch.stage}:{batch.layer_index}": batch.tokens
        for batch in adapter.extract(inputs, ids)
    }


def release_against_source(
    adapter, source: dict[str, torch.nn.Module], images: list[Image.Image] | None = None
) -> dict[str, dict]:
    """Extract with the published release, then with modules rebuilt from the parent, through the same adapter."""
    images = images or fixed_images()
    released = features(adapter, images)
    for name, module in source.items():
        setattr(adapter, name, module.to(adapter.device).eval())
    rebuilt = features(adapter, images)
    return {key: compare(released[key], rebuilt[key]) for key in released}


def bit_exact(report: dict) -> bool:
    cells = [*report.get("weights", {}).values(), *report.get("forward", {}).values()]
    return bool(cells) and all(cell["bit_exact"] for cell in cells)


def module_name(slug: str) -> str:
    from vtb.roster import BY_SLUG

    return f"vtb.parity.{BY_SLUG[slug].adapter_path.split(':')[0].rsplit('.', 1)[-1]}"


def released() -> tuple[str, ...]:
    """Slugs we publish a release for. Upstream controls load their own release and have nothing to check."""
    from vtb.roster import SLUGS

    return tuple(slug for slug in SLUGS if find_spec(module_name(slug)) is not None)


def module_for(slug: str):
    from vtb.roster import BY_SLUG

    return import_module(module_name(slug)), getattr(
        BY_SLUG[slug].adapter(), "checkpoint", None
    )


def setting(slug: str, name: str, default):
    """A parity module's setting, given once or per checkpoint name when one module serves several."""
    module, checkpoint = module_for(slug)
    value = getattr(module, name, default)
    return value.get(checkpoint.name, default) if isinstance(value, dict) else value


def requires_cuda(slug: str) -> bool:
    return setting(slug, "REQUIRES_CUDA", False)


def default_dtype(slug: str) -> torch.dtype:
    """FP32 on every device, so a release that rounds a parent tensor fails; some references only run in BF16."""
    return setting(slug, "DTYPE", torch.float32)


def validate(slug: str, device: str = "cpu", dtype: torch.dtype | None = None) -> dict:
    module, checkpoint = module_for(slug)
    dtype = dtype or default_dtype(slug)
    args = (checkpoint,) if checkpoint is not None else ()
    report = module.validate(*args, device=device, dtype=dtype)
    return {"tower": slug, "device": str(device), "dtype": str(dtype), **report}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tower", choices=released())
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--out", type=Path, help="write the full report here")
    args = parser.parse_args()
    report = validate(
        args.tower, args.device, getattr(torch, args.dtype) if args.dtype else None
    )
    text = json.dumps(report, indent=2, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text)
    print(
        f"{args.tower}: {'passed' if report['passed'] else 'FAILED'}", file=sys.stderr
    )
    sys.exit(0 if report["passed"] else 1)
