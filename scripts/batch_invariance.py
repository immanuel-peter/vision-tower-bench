"""Load one adapter, run its final Stages at an extraction batch size, and
compare every image with the same image run alone (batch 1).

Records load time, peak GPU memory, per-Stage shapes and the max absolute and
relative differences between batch B and batch 1, so a batch size can be
chosen with evidence. Images are read with vtb.images.ImageFolder (sorted by
path, ID = file stem), exactly as vtb.extract reads them.

    CUDA_VISIBLE_DEVICES=1 uv run python scripts/batch_invariance.py --model qwen3_5 \
        --images $VTB_WORK/features/smoke/inputs --batch-size 16 --out report.json
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from vtb.extract import ADAPTERS
from vtb.images import ImageFolder


def final_cells(adapter, inputs, image_ids, final_stages=None) -> dict[str, torch.Tensor]:
    """Stage -> tokens (N, T, D) at the final Tower layer, as vtb.extract --final-stages keeps them."""
    cells = {}
    for batch in adapter.extract(inputs, image_ids):
        if batch.layer_index != adapter.num_layers:
            continue
        if final_stages is not None and batch.stage not in final_stages:
            continue
        if list(batch.image_ids) != list(image_ids):
            raise ValueError("adapter changed the image ID order")
        cells[batch.stage] = batch.tokens
    return cells


def compare(batched: dict[str, torch.Tensor], single: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Per-Stage max |batch B - batch 1| over every image, absolute and relative to max |batch 1|."""
    if set(batched) != set(single):
        raise ValueError(f"Stage sets differ: {sorted(batched)} vs {sorted(single)}")
    result = {}
    for stage, tokens in batched.items():
        reference = single[stage]
        if tokens.shape != reference.shape:
            raise ValueError(f"{stage}: shape {tuple(tokens.shape)} vs {tuple(reference.shape)}")
        a, b = tokens.double(), reference.double()
        delta = (a - b).abs()
        per_image = delta.flatten(1).amax(1)
        scale = b.abs().max().item()
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        result[stage] = {
            "shape_per_image": list(tokens.shape[1:]),
            "dtype": str(tokens.dtype),
            "finite": bool(torch.isfinite(a).all()),
            "bit_exact": torch.equal(tokens, reference),
            "max_abs_diff": delta.max().item(),
            "max_rel_diff": delta.max().item() / scale if scale else 0.0,
            "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
            "min_token_cosine": cosine.min().item(),
            "reference_max_abs": scale,
            "per_image_max_abs_diff": [round(value, 8) for value in per_image.tolist()],
        }
    return result


def one_at_a_time(adapter, values, ids, final_stages) -> dict[str, torch.Tensor]:
    single: dict[str, list[torch.Tensor]] = {}
    for value, image_id in zip(values, ids, strict=True):
        for stage, tokens in final_cells(adapter, adapter.collate([value]), [image_id], final_stages).items():
            single.setdefault(stage, []).append(tokens)
    return {stage: torch.cat(parts) for stage, parts in single.items()}


def in_batches(adapter, values, ids, batch_size, final_stages) -> dict[str, torch.Tensor]:
    parts: dict[str, list[torch.Tensor]] = {}
    for at in range(0, len(values), batch_size):
        cells = final_cells(adapter, adapter.collate(values[at:at + batch_size]), ids[at:at + batch_size], final_stages)
        for stage, tokens in cells.items():
            parts.setdefault(stage, []).append(tokens)
    return {stage: torch.cat(chunks) for stage, chunks in parts.items()}


def run(args) -> dict:
    device = args.device
    options = {"stages": args.stages} if args.stages else {}
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    adapter = ADAPTERS[args.model](resolution=args.resolution, device=device, **options)
    load_seconds = time.perf_counter() - start
    dataset = ImageFolder(args.images, adapter.preprocess(), args.limit)
    samples = [dataset[i] for i in range(len(dataset))]
    values = [value for value, _ in samples]
    ids = [image_id for _, image_id in samples]

    batched: dict[str, list[torch.Tensor]] = {}
    seconds = 0.0
    for at in range(0, len(values), args.batch_size):
        chunk_ids = ids[at:at + args.batch_size]
        inputs = adapter.collate(values[at:at + args.batch_size])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        tick = time.perf_counter()
        cells = final_cells(adapter, inputs, chunk_ids, args.final_stages)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds += time.perf_counter() - tick
        for stage, tokens in cells.items():
            batched.setdefault(stage, []).append(tokens)
    stacked_batched = {stage: torch.cat(parts) for stage, parts in batched.items()}
    stacked_single = one_at_a_time(adapter, values, ids, args.final_stages)
    report = {
        "model": args.model,
        "model_id": adapter.model_id,
        "resolution": adapter.resolution,
        "num_layers": adapter.num_layers,
        "adapter_stages": list(adapter.stages),
        "final_stages": sorted(stacked_batched),
        "batch_size": args.batch_size,
        "images": ids,
        "image_count": len(ids),
        "load_seconds": round(load_seconds, 2),
        "batched_images_per_second": round(len(ids) / seconds, 2) if seconds else None,
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.startswith("cuda") else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "stages": compare(stacked_batched, stacked_single),
    }
    if args.float32_reference:
        # Rounding scale: the same batch-1 forward with float32 weights and activations.
        del adapter
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        reference = ADAPTERS[args.model](resolution=args.resolution, device=device, dtype=torch.float32, **options)
        exact = one_at_a_time(reference, values, ids, args.final_stages)
        exact_batched = in_batches(reference, values, ids, args.batch_size, args.final_stages)
        # Cross-image leakage shows in float32 too; BF16 kernel-choice rounding does not.
        report["float32_reference"] = {
            "batch1_vs_float32": compare(stacked_single, exact),
            "batched_vs_float32": compare(stacked_batched, exact),
            "float32_batched_vs_float32_batch1": compare(exact_batched, exact),
        }
    if args.save_features:
        from safetensors.torch import save_file

        args.save_features.parent.mkdir(parents=True, exist_ok=True)
        save_file({f"{stage}": tokens.contiguous() for stage, tokens in stacked_batched.items()}, args.save_features,
                  metadata={"image_ids": json.dumps(ids), "batch_size": str(args.batch_size)})
        report["features"] = str(args.save_features)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stages", nargs="+", choices=("tower", "merged", "projected"), default=None)
    parser.add_argument("--final-stages", nargs="+", choices=("tower", "merged", "projected"), default=None)
    parser.add_argument("--float32-reference", action="store_true",
                        help="also run batch 1 in float32 to show the BF16 rounding scale")
    parser.add_argument("--save-features", type=Path, default=None, help="also save the batch-B final cells")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {stage: {k: cell[k] for k in ("shape_per_image", "finite", "bit_exact", "max_abs_diff", "max_rel_diff", "relative_l2", "min_token_cosine")}
               for stage, cell in report["stages"].items()}
    print(json.dumps({k: report[k] for k in ("model", "batch_size", "image_count", "load_seconds",
                                             "batched_images_per_second", "peak_gpu_memory_gib")}, indent=1))
    print(json.dumps(summary, indent=1))
    for name, cells in report.get("float32_reference", {}).items():
        print(name, json.dumps({stage: {k: cell[k] for k in ("max_abs_diff", "relative_l2", "min_token_cosine")}
                                for stage, cell in cells.items()}))


if __name__ == "__main__":
    main()
