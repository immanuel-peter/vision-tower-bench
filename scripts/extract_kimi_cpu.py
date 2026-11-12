"""Resume CPU-only BF16 Kimi extraction from verified, atomic full-slice chunks."""

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from vtb.adapters.moonvit_v2 import K3, KimiK31ProAdapter, MoonViTV2Adapter, collate
from vtb.feature_batch import concat
from vtb.images import ImageFolder

SLICES = [("tower", layer) for layer in (3, 7, 10, 14, 17, 20, 24, 27)] + [("merged", 27), ("projected", 27)]
WIDTHS = {"tower": 1024, "merged": 4096, "projected": 7168}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_batches(batches, image_id):
    keyed = {(batch.stage, batch.layer_index): batch for batch in batches}
    if len(batches) != 10 or set(keyed) != set(SLICES):
        raise ValueError("missing or duplicate extraction slices")
    for (stage, _), batch in keyed.items():
        count = 1024 if stage == "tower" else 256
        if batch.image_ids != [image_id] or tuple(batch.tokens.shape) != (1, count, WIDTHS[stage]):
            raise ValueError("incorrect full-grid shape or image identity")
        if batch.tokens.dtype != torch.bfloat16 or batch.tokens.device.type != "cpu" or not torch.isfinite(batch.tokens).all():
            raise ValueError("wrong dtype/device or nonfinite feature")
    tower = keyed["tower", 27].tokens
    raster = tower.reshape(1, 16, 2, 16, 2, 1024).permute(0, 1, 3, 2, 4, 5).reshape(1, 256, 4096)
    if not torch.equal(raster, keyed["merged", 27].tokens):
        raise ValueError("merged Stage does not preserve raster-order 2x2 groups")


def restore_chunks(run, expected_ids, provenance):
    completed = []
    for index, chunk in enumerate(sorted((run / ".chunks").glob("[0-9]*"))):
        if chunk.name != f"{index:05d}":
            raise ValueError("committed chunk indices are not contiguous")
        record = json.loads((chunk / "chunk.json").read_text())
        if record["provenance"] != provenance or record["image_ids"] != expected_ids[len(completed):len(completed) + len(record["image_ids"])]:
            raise ValueError("resume provenance or ordered image IDs differ")
        if len(record["files"]) != 10:
            raise ValueError("incomplete committed chunk")
        names = {f"{stage}_L{layer:02d}_{index:05d}.safetensors" for stage, layer in SLICES}
        if set(record["files"]) != names:
            raise ValueError("committed chunk does not contain the ten expected slices")
        for name, expected_hash in record["files"].items():
            source, target = chunk / name, run / name
            if digest(source) != expected_hash:
                raise ValueError(f"committed feature hash mismatch: {source}")
            if not target.exists():
                os.link(source, target)
            elif digest(target) != expected_hash:
                raise ValueError(f"flat cache hash mismatch: {target}")
        completed.extend(record["image_ids"])
    return completed


def extract(model, args):
    from huggingface_hub import hf_hub_download
    checkpoint = KimiK31ProAdapter.checkpoint if model == "kimi_k31_pro" else replace(K3, bundle_revision=args.k3_revision)
    run = args.out / f"{model}_448_full"
    (run / ".chunks").mkdir(parents=True, exist_ok=True)
    expected_ids = json.loads(args.ids.read_text())["ids"]
    if len(expected_ids) != 771 or len(set(expected_ids)) != 771:
        raise ValueError("DIODE requires the complete 771-image source manifest")
    dataset = ImageFolder(args.images, lambda image: image)
    ids = [path.stem for path in dataset.paths]
    if ids != expected_ids:
        raise ValueError("prepared image order differs from audited DIODE manifest")
    provenance = dict(model_id=checkpoint.model_id, release_revision=checkpoint.bundle_revision,
                      parent_repo=checkpoint.repo, parent_revision=checkpoint.revision,
                      device="cpu", dtype="bfloat16", batch_size=1, threads=4, workers=0,
                      resolution=448, pooled_to=None, torch_version=torch.__version__, python=platform.python_version(),
                      images_per_shard=args.chunk_images,
                      adapter_sha256=digest(Path(__file__).parents[1] / "vtb/adapters/moonvit_v2.py"),
                      runner_sha256=digest(__file__), ordered_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    done = restore_chunks(run, ids, provenance)
    if len(done) == len(ids):
        from scripts.validate_kimi_results import validate_cache
        atomic_json(run / "cache-validation.json", validate_cache(run, checkpoint.model_id, ids, "geometry"))
        print(f"{model}: all 771 images already verified; no forward repeated", flush=True)
        return
    for stale in run.glob(".staging-*"):
        shutil.rmtree(stale)
    adapter_class = type("PinnedCPUAdapter", (MoonViTV2Adapter,), {"checkpoint": checkpoint, "model_id": checkpoint.model_id})
    adapter = adapter_class(resolution=448, dtype=torch.bfloat16, device="cpu")
    for filename, expected_count in (("model.safetensors", 165), ("projector.safetensors", 3)):
        from safetensors.torch import load_file
        weights = load_file(hf_hub_download(checkpoint.model_id, filename, revision=checkpoint.bundle_revision))
        actual = adapter.model.state_dict() if filename == "model.safetensors" else adapter.projector.state_dict()
        if len(weights) != expected_count or any(not torch.equal(actual[name], weight) for name, weight in weights.items()):
            raise ValueError("loaded checkpoint differs from immutable release")
    atomic_json(run / "provenance.json", provenance)
    started, new_images, durations = time.monotonic(), 0, []
    for start in range(len(done), len(ids), args.chunk_images):
        stop = min(start + args.chunk_images, len(ids))
        chunk_index = start // args.chunk_images
        staging = run / f".staging-{chunk_index:05d}"
        staging.mkdir()
        parts = {key: [] for key in SLICES}
        for index in range(start, stop):
            image_started = time.monotonic()
            pixels, image_id = dataset[index]
            batches = list(adapter.extract(collate([adapter.preprocess()(pixels)]), [image_id]))
            validate_batches(batches, image_id)
            for batch in batches:
                parts[batch.stage, batch.layer_index].append(batch)
            durations.append(time.monotonic() - image_started)
        files = {}
        for (stage, layer), batches in parts.items():
            name = f"{stage}_L{layer:02d}_{chunk_index:05d}.safetensors"
            concat(batches).save(staging / name)
            files[name] = digest(staging / name)
        record = dict(provenance=provenance, image_ids=ids[start:stop], files=files)
        atomic_json(staging / "chunk.json", record)
        committed = run / ".chunks" / f"{chunk_index:05d}"
        staging.rename(committed)
        for name in files:
            os.link(committed / name, run / name)
        new_images += stop - start
        rate = new_images / (time.monotonic() - started)
        progress = dict(provenance=provenance, completed_images=stop, required_images=771, complete=False,
                        valid_slices=10, images_per_second=rate, estimated_full_seconds=771 / rate,
                        remaining_seconds=(771 - stop) / rate, elapsed_seconds=time.monotonic() - started,
                        recent_image_seconds=durations[-args.chunk_images:])
        atomic_json(run / "progress.json", progress)
        print(json.dumps(dict(model=model, **{k: progress[k] for k in ("completed_images", "images_per_second", "estimated_full_seconds", "remaining_seconds")})), flush=True)
        del parts, batches
        if new_images >= args.chunk_images and progress["estimated_full_seconds"] > args.max_projected_seconds:
            print(f"{model}: projected duration exceeds bound; verified chunks retained for explicit resume", flush=True)
            return
    from scripts.validate_kimi_results import validate_cache
    atomic_json(run / "cache-validation.json", validate_cache(run, checkpoint.model_id, ids, "geometry"))
    progress["complete"] = True
    atomic_json(run / "progress.json", progress)
    del adapter
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("kimi_k31_pro", "moonvit_v2"), default=["kimi_k31_pro", "moonvit_v2"])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k3-revision", default="bcaedfccd1f07d23ab20de8a0a168668c27cacb3")
    parser.add_argument("--chunk-images", type=int, default=8)
    parser.add_argument("--max-projected-seconds", type=float, default=5400)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU extraction requires CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(4)
    for model in args.models:
        extract(model, args)


if __name__ == "__main__":
    main()
