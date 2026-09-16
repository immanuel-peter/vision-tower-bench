"""Forward parity: MiniMax-M3 adapter vs HF processor + MiniMaxM3VLVisionModel.

Compares, on a handful of ImageNet-100 images:
  (a) our Preprocess+collate vs HF MiniMaxM3VLImageProcessor, same tower
  (b) adapter.extract tower tokens vs direct VisionModel forward + raster
  (c) grid_thw [[1,32,32]] vs [[1,16,16]] for degeneracy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers.models.minimax_m3_vl.image_processing_minimax_m3_vl import (
    MiniMaxM3VLImageProcessor,
)
from vtb.adapters.minimax_m3 import (
    BUNDLE,
    MiniMaxM3Adapter,
    Preprocess,
    collate,
    raster,
)
from vtb.images import square_crop


def _stats(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs()
    out = {
        "name": name,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "max_abs": float(diff.max()) if diff.numel() else float("nan"),
        "mean_abs": float(diff.mean()) if diff.numel() else float("nan"),
        "a_finite": bool(torch.isfinite(a.float()).all()),
        "b_finite": bool(torch.isfinite(b.float()).all()),
    }
    print(
        f"{name}: max_abs={out['max_abs']:.6g}  mean_abs={out['mean_abs']:.6g}  "
        f"shapes {out['a_shape']} vs {out['b_shape']}"
    )
    return out


def _rank_and_norms(tokens: torch.Tensor) -> dict:
    flat = tokens.float().reshape(-1, tokens.shape[-1])
    finite = bool(torch.isfinite(flat).all())
    norms = flat.norm(dim=-1)
    # Cheap rank proxy on a token subsample.
    sample = flat[: min(4096, flat.shape[0])]
    sample = sample - sample.mean(dim=0, keepdim=True)
    q = min(64, *sample.shape)
    if q == 0:
        s = torch.zeros(0)
    else:
        _, s, _ = torch.svd_lowrank(sample, q=q)
    return {
        "finite": finite,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(unbiased=False)),
        "norm_min": float(norms.min()),
        "zero_frac": float((norms < 1e-6).float().mean()),
        "nan": bool(torch.isnan(tokens.float()).any()),
        "svd_top": [float(x) for x in s[:8]],
        "effective_rank_proxy": float((s > 1e-3 * s[0]).sum()) if s.numel() else 0.0,
    }


def load_images(root: Path, n: int) -> list[Image.Image]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) < n:
        raise SystemExit(f"need {n} images under {root}, found {len(paths)}")
    images = []
    for path in paths[:n]:
        with Image.open(path) as img:
            images.append(img.convert("RGB").copy())
    return images


def hf_processor() -> MiniMaxM3VLImageProcessor:
    """Load the installed transformers processor. The local bundle's
    preprocessor_config auto_map points at files that are not in the
    standalone vision repo, so construct the class from the JSON fields."""
    cfg = json.loads((BUNDLE / "preprocessor_config.json").read_text())
    return MiniMaxM3VLImageProcessor(
        image_mean=cfg["image_mean"],
        image_std=cfg["image_std"],
        patch_size=cfg["patch_size"],
        size=cfg.get("size"),
        do_normalize=True,
        do_rescale=True,
        do_resize=True,
    )


def hf_processor_on_pils(images: list[Image.Image], size: int | None) -> dict[str, torch.Tensor]:
    processor = hf_processor()
    kwargs = {}
    if size is not None:
        # Force a square grid at `size` so token counts can match the bench.
        kwargs["size"] = {"shortest_edge": size * size, "longest_edge": size * size}
        kwargs["min_pixels"] = size * size
        kwargs["max_pixels"] = size * size
    out = processor(images, return_tensors="pt", **kwargs)
    return {"pixel_values": out["pixel_values"], "grid_thw": out["image_grid_thw"]}


def our_collate(images: list[Image.Image], resolution: int) -> dict[str, torch.Tensor]:
    prep = Preprocess(resolution)
    return collate([prep(im) for im in images])


@torch.inference_mode()
def tower_tokens(model, inputs: dict[str, torch.Tensor], device: str, dtype):
    pixel_values = inputs["pixel_values"].to(device, dtype)
    grid_thw = inputs["grid_thw"].to(device)
    out = model(pixel_values, grid_thw)
    hidden = out.last_hidden_state
    if hidden.ndim == 3:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    return hidden, out.pooler_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/imagenet100/validation"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("/tmp/mm3_forward_parity.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    images = load_images(args.images, args.n)
    adapter = MiniMaxM3Adapter(resolution=args.resolution, device=args.device)
    model = adapter.model
    dtype = adapter.dtype
    rows = args.n
    width = model.config.hidden_size
    side = args.resolution // 14

    report: dict = {"n": args.n, "resolution": args.resolution, "device": args.device}

    # --- (a) preprocess: ours vs HF, same tower --------------------------------
    ours = our_collate(images, args.resolution)
    hf_default = hf_processor_on_pils(images, size=None)
    hf_forced = hf_processor_on_pils(images, size=args.resolution)

    report["preprocess"] = {
        "ours_pixel_values": list(ours["pixel_values"].shape),
        "ours_grid_thw": ours["grid_thw"].tolist(),
        "hf_default_pixel_values": list(hf_default["pixel_values"].shape),
        "hf_default_grid_thw": hf_default["grid_thw"].tolist(),
        "hf_forced_pixel_values": list(hf_forced["pixel_values"].shape),
        "hf_forced_grid_thw": hf_forced["grid_thw"].tolist(),
    }
    print("preprocess shapes:", json.dumps(report["preprocess"], indent=2))

    if ours["pixel_values"].shape == hf_forced["pixel_values"].shape:
        report["a_pixel_ours_vs_hf_forced"] = _stats(
            "pixel_values ours vs HF-forced-square",
            ours["pixel_values"],
            hf_forced["pixel_values"],
        )
    else:
        report["a_pixel_ours_vs_hf_forced"] = {
            "name": "pixel_values ours vs HF-forced-square",
            "a_shape": list(ours["pixel_values"].shape),
            "b_shape": list(hf_forced["pixel_values"].shape),
            "note": "shape mismatch; cannot compare token-for-token on pixels",
        }
        print("pixel_values shape mismatch between ours and HF-forced")

    # Same tower, two preprocessors.
    ours_hidden, ours_pool = tower_tokens(model, ours, args.device, dtype)
    hf_forced_hidden, hf_forced_pool = tower_tokens(model, hf_forced, args.device, dtype)
    hf_default_hidden, hf_default_pool = tower_tokens(model, hf_default, args.device, dtype)

    if ours_hidden.shape == hf_forced_hidden.shape:
        report["a_hidden_ours_vs_hf_forced"] = _stats(
            "(a) last_hidden_state ours-prep vs HF-forced-prep",
            ours_hidden,
            hf_forced_hidden,
        )
    else:
        report["a_hidden_ours_vs_hf_forced"] = {
            "ours_shape": list(ours_hidden.shape),
            "hf_forced_shape": list(hf_forced_hidden.shape),
            "note": "shape mismatch",
        }
        print("(a) hidden shape mismatch", ours_hidden.shape, hf_forced_hidden.shape)

    report["a_hf_default"] = {
        "hidden_shape": list(hf_default_hidden.shape),
        "grid_thw": hf_default["grid_thw"].tolist(),
        "stats": _rank_and_norms(hf_default_hidden),
    }

    # Also: crop to 448 first, then both pipelines (isolates patchify from resize).
    cropped = [square_crop(args.resolution)(im) for im in images]
    ours_c = our_collate(cropped, args.resolution)
    hf_c = hf_processor_on_pils(cropped, size=args.resolution)
    ours_c_h, _ = tower_tokens(model, ours_c, args.device, dtype)
    hf_c_h, _ = tower_tokens(model, hf_c, args.device, dtype)
    if ours_c["pixel_values"].shape == hf_c["pixel_values"].shape:
        report["a_pixel_pre_cropped"] = _stats(
            "pixel_values pre-cropped 448 ours vs HF",
            ours_c["pixel_values"],
            hf_c["pixel_values"],
        )
    if ours_c_h.shape == hf_c_h.shape:
        report["a_hidden_pre_cropped"] = _stats(
            "(a) hidden pre-cropped 448 ours vs HF",
            ours_c_h,
            hf_c_h,
        )

    # --- (b) adapter.extract vs direct forward + raster ------------------------
    image_ids = [f"img{i}" for i in range(rows)]
    extracted = {
        (b.stage, b.layer_index): b.tokens
        for b in adapter.extract({k: v.clone() for k, v in ours.items()}, image_ids)
    }
    packed = ours_hidden
    if packed.ndim == 2:
        packed = packed.view(rows, -1, width)
    rasters = raster(packed, rows).cpu()
    last_layer = adapter.num_layers
    adapter_tower = extracted[("tower", last_layer)]
    report["b_extract_vs_direct_raster"] = _stats(
        "(b) adapter.extract tower L32 vs direct forward+raster",
        adapter_tower,
        rasters,
    )
    # pooler_output is hidden_states[:, 0] on the packed (1, N, C) tensor:
    # the first token of image 0, not a per-image CLS.
    packed_3d = ours_hidden.view(1, -1, width) if ours_hidden.ndim == 2 else ours_hidden
    report["b_pooler_vs_packed_token0"] = _stats(
        "(b) pooler_output vs packed hidden[:, 0] (HF definition)",
        ours_pool.float().cpu(),
        packed_3d[:, 0].float().cpu(),
    )

    # --- (c) grid_thw 32 vs 16 -------------------------------------------------
    # Wrong grid_thw does not silently degrade: RoPE length must match the
    # packed token count (n_images * h * w). Catch that instead of dying.
    pix = ours["pixel_values"].to(args.device, dtype)
    g32 = torch.tensor([[1, side, side]] * rows, device=args.device)
    g16 = torch.tensor([[1, side // 2, side // 2]] * rows, device=args.device)
    h32 = model(pix, g32).last_hidden_state
    report["c_grid_32"] = {"shape": list(h32.shape), **_rank_and_norms(h32)}
    try:
        h16 = model(pix, g16).last_hidden_state
        report["c_grid_16"] = {"shape": list(h16.shape), **_rank_and_norms(h16)}
        if h32.shape == h16.shape:
            report["c_32_vs_16"] = _stats("(c) hidden grid 32 vs 16", h32, h16)
        else:
            report["c_32_vs_16"] = {
                "note": "shape mismatch as expected if RoPE/packing uses grid_thw",
                "h32": list(h32.shape),
                "h16": list(h16.shape),
            }
            print("(c) shape 32", h32.shape, "16", h16.shape)
    except RuntimeError as exc:
        report["c_grid_16"] = {
            "error": str(exc),
            "note": "wrong grid_thw raises rather than emitting degenerate features",
            "pixel_tokens": int(pix.shape[0]),
            "grid_16_expected_tokens": int(rows * (side // 2) ** 2),
        }
        print("(c) grid 16 raised:", exc)

    report["wall_seconds"] = round(time.perf_counter() - t0, 2)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out} in {report['wall_seconds']}s")


if __name__ == "__main__":
    main()
