"""Feature stats on MiniMax-M3 full-grid tower slices (L20 and L32).

Reports per-token L2 norms, PCA spectrum / effective rank, and cosine-kNN
(k=20) top-1 using the same split as `vtb.probe_run.split_indices`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from vtb import cache
from vtb.probe_run import split_indices


def load_slice(run_dir: Path, stage: str, layer: int) -> tuple[torch.Tensor, list[str]]:
    parts: list[torch.Tensor] = []
    image_ids: list[str] = []
    for path in cache.shards(run_dir, stage, layer):
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            parts.append(handle.get_tensor("tokens"))
            image_ids.extend(metadata["image_ids"].split("\n"))
    return torch.cat(parts), image_ids


def load_pooled_views(run_dir: Path, stage: str, layer: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Mean-pooled and token-0 features without materializing the full grid as float32."""
    means: list[torch.Tensor] = []
    token0: list[torch.Tensor] = []
    image_ids: list[str] = []
    for path in cache.shards(run_dir, stage, layer):
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            tokens = handle.get_tensor("tokens")
            means.append(tokens.float().mean(dim=1))
            token0.append(tokens[:, 0, :].float())
            image_ids.extend(metadata["image_ids"].split("\n"))
    return torch.cat(means), torch.cat(token0), image_ids


def token_norms(tokens: torch.Tensor) -> dict:
    # tokens: (N, T, C) bf16. Compute L2 per token in chunks.
    n, t, _ = tokens.shape
    means = []
    sqs = []
    mx = torch.tensor(0.0)
    # top-1% share: gather a sample of norms rather than all N*T if huge.
    sample_norms = []
    chunk = 512
    total = n * t
    for start in range(0, n, chunk):
        sl = tokens[start : start + chunk].float()
        norms = sl.norm(dim=-1)  # (b, T)
        means.append(norms.mean())
        sqs.append((norms * norms).mean())
        mx = torch.maximum(mx, norms.max())
        sample_norms.append(norms.reshape(-1)[:: max(1, norms.numel() // 4096)])
    mean = float(torch.stack(means).mean())
    # population std from mean of squares
    mean_sq = float(torch.stack(sqs).mean())
    std = (mean_sq - mean * mean) ** 0.5
    all_sample = torch.cat(sample_norms)
    thresh = torch.quantile(all_sample, 0.99)
    top_share = float((all_sample >= thresh).float().mean())
    # per-position mean norm (token 0 vs rest)
    pos_mean = tokens[: min(n, 2048)].float().norm(dim=-1).mean(dim=0)
    return {
        "n_images": n,
        "n_tokens": t,
        "norm_mean": mean,
        "norm_std": std,
        "norm_max": float(mx),
        "top1pct_share_of_sampled_tokens": top_share,
        "token0_norm_mean": float(pos_mean[0]),
        "token_rest_norm_mean": float(pos_mean[1:].mean()) if t > 1 else None,
        "token0_over_rest": float(pos_mean[0] / pos_mean[1:].mean()) if t > 1 else None,
        "pos_norm_mean_first8": [float(x) for x in pos_mean[:8]],
        "sampled_tokens_for_quantile": int(all_sample.numel()),
        "total_tokens": total,
    }


def pca_spectrum(tokens: torch.Tensor, q: int = 256, n_images: int = 2048) -> dict:
    sl = tokens[: min(len(tokens), n_images)].float()
    flat = sl.reshape(-1, sl.shape[-1])
    mean = flat.mean(dim=0, keepdim=True)
    q = min(q, *flat.shape)
    _, s, _ = torch.svd_lowrank(flat - mean, q=q)
    energy = (s * s)
    total = float(energy.sum())
    cum = torch.cumsum(energy, dim=0) / total
    # Effective rank: exp(entropy of normalized eigenvalues)
    p = energy / total
    p = p.clamp_min(1e-12)
    erank = float(torch.exp(-(p * p.log()).sum()))
    return {
        "q": q,
        "n_images": int(sl.shape[0]),
        "top8_singular": [float(x) for x in s[:8]],
        "energy_frac_top8": float(cum[7]) if len(cum) > 7 else float(cum[-1]),
        "energy_frac_top32": float(cum[31]) if len(cum) > 31 else float(cum[-1]),
        "k_for_90pct": int((cum < 0.90).sum() + 1),
        "k_for_99pct": int((cum < 0.99).sum() + 1),
        "effective_rank": erank,
    }


@torch.inference_mode()
def cosine_knn(
    feats: torch.Tensor,
    labels: torch.Tensor,
    split,
    k: int = 20,
    pool: str = "mean",
) -> dict:
    """Image-level cosine kNN. `feats` is already (N, C)."""
    feats = torch.nn.functional.normalize(feats, dim=-1)
    gallery = feats[split.train]
    gallery_y = labels[split.train]
    query = feats[split.test]
    query_y = labels[split.test]
    # Chunked matmul to keep peak RAM down.
    topk_idx = []
    bs = 256
    for start in range(0, len(query), bs):
        sim = query[start : start + bs] @ gallery.T
        topk_idx.append(sim.topk(k, dim=-1).indices.cpu())
    idx = torch.cat(topk_idx)
    neigh = gallery_y[idx]
    # majority vote
    pred = []
    for row in neigh:
        counts = torch.bincount(row, minlength=int(labels.max()) + 1)
        pred.append(int(counts.argmax()))
    pred_t = torch.tensor(pred)
    top1 = float((pred_t == query_y).float().mean())
    return {"k": k, "pool": pool, "test_top1": round(top1, 4), "n_test": int(len(query_y))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("cache/minimax_m3_448_full"))
    parser.add_argument("--labels", type=Path, default=Path("data/imagenet100/validation_labels.json"))
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 32])
    parser.add_argument("--stage", default="tower")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("/tmp/mm3_feature_stats.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    label_map = json.loads(args.labels.read_text())["labels"]
    layers = []
    for layer in args.layers:
        print(f"\n=== L{layer} ===", flush=True)
        means, token0, image_ids = load_pooled_views(args.run, args.stage, layer)
        labels = torch.tensor([label_map[i] for i in image_ids])
        split = split_indices(len(image_ids))
        # Norms/PCA still need a spatial slice; stream one shard-worth via load_slice
        # would OOM at 34 GB bf16 + workspace. Use token-0 and mean views plus a
        # 2048-image subsample of the raw grid for spectrum/norms.
        sample_parts = []
        n_sample = 0
        for path in cache.shards(args.run, args.stage, layer):
            with safe_open(path, framework="pt") as handle:
                tokens = handle.get_tensor("tokens")
            take = min(tokens.shape[0], 2048 - n_sample)
            sample_parts.append(tokens[:take])
            n_sample += take
            if n_sample >= 2048:
                break
        sample = torch.cat(sample_parts)
        print(f"sample {tuple(sample.shape)} {sample.dtype}; mean feats {tuple(means.shape)}", flush=True)
        norms = token_norms(sample)
        print("norms", json.dumps(norms, indent=2), flush=True)
        spec = pca_spectrum(sample)
        print("pca", json.dumps(spec, indent=2), flush=True)
        knn_mean = cosine_knn(means, labels, split, k=args.k, pool="mean")
        knn_t0 = cosine_knn(token0, labels, split, k=args.k, pool="token0")
        print("knn mean", knn_mean, "knn token0", knn_t0, flush=True)
        layers.append(
            {
                "layer": layer,
                "shape": [len(image_ids), "full_grid_not_materialized", means.shape[-1]],
                "n_images": len(image_ids),
                "sample_shape": list(sample.shape),
                "norms": norms,
                "pca": spec,
                "knn_mean": knn_mean,
                "knn_token0": knn_t0,
            }
        )
        del sample, means, token0

    payload = {
        "run": str(args.run),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "layers": layers,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out} in {payload['wall_seconds']}s")


if __name__ == "__main__":
    main()
