"""Record which parent checkpoint a vision bundle was exported from."""

import json
import shutil
from pathlib import Path

from safetensors import safe_open


def copy_assets(source, out):
    source, out = Path(source), Path(out)
    for name in (
        "README.md",
        "NOTICE",
        "LICENSE-CODE",
        "projector.py",
        "examples/inference.py",
    ):
        src, dst = source / name, out / name
        if src.is_file() and src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


def write_manifest(
    out,
    source_repo,
    source_revision,
    source_shards,
    *,
    merger_in_model=False,
    extra=None,
    allow_mixed_source=False,
):
    out = Path(out)
    counts, dtypes = {}, {}
    for part, name in (
        ("tower", "model.safetensors"),
        ("projector", "projector.safetensors"),
    ):
        path = out / name
        if not path.exists():
            if part == "projector" and merger_in_model:
                continue
            raise ValueError(f"missing exported weight file: {path}")
        with safe_open(path, framework="pt") as weights:
            keys = list(weights.keys())
            for key in keys:
                dtype = weights.get_slice(key).get_dtype()
                if dtype != "BF16" and not (allow_mixed_source and dtype == "F32"):
                    raise ValueError("expected original BF16 source tensors")
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
        if part == "tower" and merger_in_model:
            counts["projector"] = sum(key.startswith("merger.") for key in keys)
            counts["tower"] = len(keys) - counts["projector"]
        else:
            counts[part] = len(keys)
    payload = dict(extra or {})
    payload.update(
        source_repo=source_repo,
        source_revision=source_revision,
        source_shards=sorted(set(source_shards)),
        tower_tensors=counts["tower"],
        projector_tensors=counts["projector"],
        dtype="bfloat16" if set(dtypes) == {"BF16"} else "mixed",
        tensor_dtypes=dtypes,
    )
    (out / "export_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
