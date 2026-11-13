"""Extract pinned Kimi siblings into separate, resumable GPU feature caches."""

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
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from vtb.adapters.moonvit_v2 import K3, K31_PRO, MoonViTV2Adapter
from vtb.extract import Collate
from vtb.extract_kimi_cpu import SLICES, atomic_json, digest, validate_batches
from vtb.feature_batch import concat
from vtb.images import ImageFolder
from vtb.validate_kimi_results import validate_cache

ROOT = Path("/tmp/vtb-delivery/issues/24")
DATA = Path("/tmp/vtb-delivery/data")
OUT = ROOT / "features/gpu1-bf16-bs1"
CHUNK = 256


def selected_slices(kind):
    if kind == "perturbation":
        return [(stage, 27) for stage in ("tower", "merged", "projected")]
    if kind == "kitti":
        return [("tower", 27), ("projected", 27)]
    return SLICES


def restore(run, ids, provenance, slices):
    done = []
    for index, chunk in enumerate(sorted((run / ".chunks").glob("[0-9]*"))):
        record = json.loads((chunk / "chunk.json").read_text())
        if chunk.name != f"{index:05d}" or record["provenance"] != provenance:
            raise ValueError("resume provenance/chunk order differs")
        if record["image_ids"] != ids[len(done):len(done) + len(record["image_ids"])]:
            raise ValueError("resume image IDs differ")
        expected = {f"{stage}_L{layer:02d}_{index:05d}.safetensors" for stage, layer in slices}
        if set(record["files"]) != expected:
            raise ValueError("resume Stage/depth files differ")
        for name, fingerprint in record["files"].items():
            source, target = chunk / name, run / name
            if digest(source) != fingerprint:
                raise ValueError(f"invalid committed feature file: {source}")
            if not target.exists():
                os.link(source, target)
            elif digest(target) != fingerprint:
                raise ValueError(f"invalid flat feature file: {target}")
        done.extend(record["image_ids"])
    return done


def extract(adapter, model, kind, images, ids, condition=None):
    torch.set_num_threads(8)
    pool = None if kind == "kitti" else 4
    parent = OUT / kind
    if condition:
        parent /= condition
    run = parent / f"{model}_448_{'full' if pool is None else 'pool4'}"
    (run / ".chunks").mkdir(parents=True, exist_ok=True)
    dataset = ImageFolder(images, adapter.preprocess(), len(ids))
    if [path.stem for path in dataset.paths] != ids or len(set(ids)) != len(ids):
        raise ValueError(f"actual {kind}/{condition} selection differs from manifest")
    slices = selected_slices(kind)
    checkpoint = adapter.checkpoint
    properties = torch.cuda.get_device_properties(0)
    provenance = dict(
        model_id=checkpoint.model_id, release_revision=checkpoint.bundle_revision,
        parent_repo=checkpoint.repo, parent_revision=checkpoint.revision,
        device="cuda:0", physical_gpu=1, cuda_visible_devices="1",
        gpu_name=properties.name, gpu_uuid=str(properties.uuid),
        dtype="bfloat16", batch_size=1, threads=8, workers=2,
        resolution=448, pooled_to=pool, torch_version=torch.__version__,
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        python=platform.python_version(), images_per_shard=CHUNK,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        adapter_sha256=digest(Path(__file__).parents[1] / "vtb/adapters/moonvit_v2.py"),
        runner_sha256=digest(__file__), kind=kind, condition=condition,
        ordered_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        shared_inputs_sha256=digest(Path("/tmp/vtb-delivery/data-sha256.txt")),
        transform_manifest_sha256=digest(DATA / "perturbation/transforms.manifest.json") if condition else None,
        attention_backend="published adapter masked three-dimensional SDPA; production official BF16 disagreement retained",
    )
    done = restore(run, ids, provenance, slices)
    atomic_json(run / "provenance.json", provenance)
    for stale in run.glob(".staging-*"):
        shutil.rmtree(stale)
    started, newly_written = time.monotonic(), 0
    loader = DataLoader(Subset(dataset, range(len(done), len(ids))), batch_size=1,
                        num_workers=2, collate_fn=Collate(adapter.collate),
                        prefetch_factor=2, persistent_workers=False)
    parts = {key: [] for key in slices}
    pending_ids = []
    for inputs, image_ids in loader:
        batches = list(adapter.extract(inputs, image_ids))
        validate_batches(batches, image_ids[0])
        for batch in batches:
            key = batch.stage, batch.layer_index
            if key in parts:
                parts[key].append(batch if pool is None else batch.pooled(pool))
        pending_ids.extend(image_ids)
        if len(pending_ids) < CHUNK and len(done) + len(pending_ids) < len(ids):
            continue
        index = len(done) // CHUNK
        staging = run / f".staging-{index:05d}"
        staging.mkdir()
        files = {}
        for (stage, layer), batches in parts.items():
            name = f"{stage}_L{layer:02d}_{index:05d}.safetensors"
            concat(batches).save(staging / name)
            files[name] = digest(staging / name)
        atomic_json(staging / "chunk.json", dict(provenance=provenance, image_ids=pending_ids, files=files))
        staging.rename(run / ".chunks" / f"{index:05d}")
        for name in files:
            os.link(run / ".chunks" / f"{index:05d}" / name, run / name)
        newly_written += len(pending_ids)
        done.extend(pending_ids)
        elapsed = time.monotonic() - started
        progress = dict(complete=False, completed_images=len(done), required_images=len(ids),
                        slices=len(slices), elapsed_seconds=elapsed,
                        new_images=newly_written, images_per_second=newly_written / elapsed)
        atomic_json(run / "progress.json", progress)
        print(json.dumps(dict(model=model, kind=kind, condition=condition, **progress)), flush=True)
        parts = {key: [] for key in slices}
        pending_ids = []
    del loader, parts
    validation = validate_cache(run, checkpoint.model_id, ids, kind)
    torch.set_num_threads(8)
    atomic_json(run / "cache-validation.json", validation)
    elapsed = time.monotonic() - started
    summary = dict(provenance=provenance, complete=True, images=len(ids), slices=len(slices),
                   shards=validation["shards"], bytes=sum(p.stat().st_size for p in run.glob("*.safetensors")),
                   seconds=elapsed, new_images=newly_written,
                   images_per_second=newly_written / elapsed if newly_written else None,
                   content_validation_sha256=digest(run / "cache-validation.json"))
    atomic_json(run / "progress.json", summary)
    atomic_json(run / "summary.json", {**provenance, **summary})
    print(json.dumps(dict(completed=str(run), images=len(ids), slices=len(slices), seconds=elapsed)), flush=True)
    return dict(run=str(run), **summary)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("requires explicit physical GPU1 grant and enclosing flock")
    grant = json.loads((ROOT / "gpu-grant.json").read_text())
    if grant["physical_gpu"] != 1 or grant["phase"] != "remaining-feature-extraction":
        raise SystemExit("missing extraction reservation")
    torch.set_num_threads(8)
    began = time.monotonic()
    manifest = json.loads((DATA / "perturbation/transforms.manifest.json").read_text())
    reports = []
    for model, checkpoint in (("kimi_k31_pro", K31_PRO), ("moonvit_v2", replace(K3, bundle_revision="bcaedfccd1f07d23ab20de8a0a168668c27cacb3"))):
        adapter_class = type("PinnedGPUAdapter", (MoonViTV2Adapter,), {"checkpoint": checkpoint, "model_id": checkpoint.model_id})
        adapter = adapter_class(resolution=448, dtype=torch.bfloat16, device="cuda:0")
        for filename, count in (("model.safetensors", 165), ("projector.safetensors", 3)):
            weights = load_file(hf_hub_download(checkpoint.model_id, filename, revision=checkpoint.bundle_revision))
            actual = adapter.model.state_dict() if filename == "model.safetensors" else adapter.projector.state_dict()
            if len(actual) != count or set(actual) != set(weights) or any(not torch.equal(actual[key].cpu(), value) for key, value in weights.items()):
                raise ValueError("loaded GPU tensors differ from audited immutable checkpoint")
            del actual, weights
        for kind, root, ids_name in (("semantic", DATA / "imagenet100/validation", "imagenet100"),
                                      ("kitti", DATA / "kitti/val", "kitti")):
            ids = json.loads((ROOT / f"ids-{ids_name}.json").read_text())["ids"]
            reports.append(extract(adapter, model, kind, root, ids))
            atomic_json(ROOT / "gpu-extraction-phase-progress.json", dict(complete=False, reports=reports))
        ids = json.loads((ROOT / "ids-perturbation.json").read_text())["ids"]
        if len(ids) != 2000 or len(manifest["conditions"]) != 13:
            raise ValueError("perturbation protocol count differs")
        for condition in manifest["conditions"]:
            images = DATA / "imagenet100/validation" if condition == "identity" else DATA / "perturbation" / condition
            reports.append(extract(adapter, model, "perturbation", images, ids, condition))
            atomic_json(ROOT / "gpu-extraction-phase-progress.json", dict(complete=False, reports=reports))
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(ROOT / "gpu-extraction-phase-progress.json", dict(complete=True, reports=reports, seconds=time.monotonic() - began))


if __name__ == "__main__":
    main()
