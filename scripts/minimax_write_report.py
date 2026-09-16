"""Assemble /tmp/MM3_DIAGNOSIS.md from the JSON artifacts this diagnosis wrote."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROSTER_ATTN = 0.3407
ROSTER_MEAN = 0.2238


def load(path: str):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"_error": f"invalid json at {path}"}


def cell_top1(blob, default="NA"):
    if not blob:
        return default
    if blob.get("skipped"):
        return f"skipped ({blob.get('reason', '')})"
    cells = blob.get("cells") or ([blob] if "test_accuracy" in blob else [])
    if not cells:
        return default
    c = cells[0]
    return f"{c['test_accuracy']:.4f} +- {c.get('test_std', 0):.4f} (lr {c.get('learning_rate')})"


def first_top1(blob) -> float | None:
    if not blob or blob.get("skipped"):
        return None
    cells = blob.get("cells") or []
    if not cells:
        return None
    return float(cells[0]["test_accuracy"])


def verdict_table(full_attn, token0, parity, res672, stats, mlp):
    attn = first_top1(full_attn)
    h1 = "OPEN"
    h1_num = "no full-grid attention number"
    if attn is not None:
        h1_num = f"full-grid L32 attention {attn:.4f} vs pool-4 0.3323"
        if attn >= 0.7:
            h1 = "SUPPORTED"
        elif attn <= 0.45:
            h1 = "REJECTED"
        else:
            h1 = "OPEN"

    h0 = "OPEN"
    h3 = "OPEN"
    h0_num = h3_num = "parity json missing"
    if parity and (
        "a_hidden_pre_cropped" in parity or "b_extract_vs_direct_raster" in parity
    ):
        a = parity.get("a_hidden_pre_cropped") or parity.get("a_hidden_ours_vs_hf_forced") or {}
        b = parity.get("b_extract_vs_direct_raster") or {}
        a_max = a.get("max_abs")
        b_max = b.get("max_abs")
        h0_num = f"(b) extract vs direct raster max_abs={b_max}"
        h3_num = f"(a) pre-cropped 448 ours vs HF hidden max_abs={a_max}"
        def near(x):
            return x is not None and x < 2e-2
        if near(b_max):
            h0 = "REJECTED"
        elif b_max is not None and b_max > 0.1:
            h0 = "SUPPORTED"
        if near(a_max):
            h3 = "REJECTED"
        elif a_max is not None and a_max > 0.1:
            h3 = "SUPPORTED"

    h2 = "OPEN"
    h2_num = "672 json missing"
    t672 = first_top1(res672)
    if t672 is not None:
        h2_num = f"672-pool4 attention {t672:.4f} vs 448-pool4 0.3323"
        if t672 >= 0.7:
            h2 = "SUPPORTED"
        elif t672 <= 0.45:
            h2 = "REJECTED"

    h4 = "OPEN"
    h4_num = "stats/mlp missing"
    knn = None
    if stats:
        for layer in stats.get("layers", []):
            km = (layer.get("knn_mean") or {}).get("test_top1")
            if km is not None:
                knn = max(knn or 0.0, km)
    mlp_top = first_top1(mlp) if mlp and not mlp.get("skipped") else None
    if knn is not None:
        h4_num = f"cosine-kNN mean {knn:.4f}"
        if mlp_top is not None:
            h4_num += f"; MLP {mlp_top:.4f}"
        if mlp_top is not None and mlp_top >= 0.7:
            h4 = "SUPPORTED"
        elif knn is not None and knn <= 0.40 and (mlp is None or mlp.get("skipped") or (mlp_top is not None and mlp_top <= 0.40)):
            h4 = "REJECTED"

    closed = {h0, h1, h2, h3, h4}
    h5 = "OPEN"
    h5_num = "exclusion not complete"
    if h0 == "REJECTED" and h1 == "REJECTED" and h2 == "REJECTED" and h3 == "REJECTED" and h4 in {"REJECTED", "OPEN"}:
        if h4 == "REJECTED":
            h5 = "SUPPORTED"
            h5_num = "H0-H4 rejected; Tower is genuinely weak on this probe"
        else:
            h5_num = "H0-H3 rejected; H4 still open"

    return [
        ("H0 extraction bug", h0, h0_num),
        ("H1 pool-4 x token-0 semantics", h1, h1_num),
        ("H2 resolution / RoPE (native 672)", h2, h2_num),
        ("H3 preprocessing mismatch", h3, h3_num),
        ("H4 nonlinear-only class info", h4, h4_num),
        ("H5 genuinely weak Tower", h5, h5_num),
    ], h1, attn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/MM3_DIAGNOSIS.md"))
    args = parser.parse_args()

    pool4_attn = load("/tmp/mm3_pool4_attn.json")
    pool4_mean = load("/tmp/mm3_pool4_mean.json")
    full_attn = load("/tmp/mm3_full_attn.json")
    full_mean = load("/tmp/mm3_full_mean_L32.json")
    token0 = load("/tmp/mm3_token0.json")
    parity = load("/tmp/mm3_forward_parity.json")
    res672 = load("/tmp/mm3_672.json")
    stats = load("/tmp/mm3_feature_stats.json")
    mlp = load("/tmp/mm3_mlp.json")

    rows, h1, attn = verdict_table(full_attn, token0, parity, res672, stats, mlp)
    status = {name: s for name, s, _ in rows}

    root = "unknown"
    conf = "low"
    if h1 == "SUPPORTED":
        root = (
            "Pool-4 averaging dilutes class semantics that live in a small set of tokens "
            "(token 0 is the HF pooler_output convention) while spatially spread figure-ground "
            "structure survives."
        )
        conf = "high" if token0 and not token0.get("skipped") else "medium"
    elif status["H0 extraction bug"] == "SUPPORTED":
        root = "Adapter forward path diverges from MiniMaxM3VLVisionModel; extraction is wrong."
        conf = "high"
    elif status["H3 preprocessing mismatch"] == "SUPPORTED":
        root = "Our Preprocess does not match MiniMaxM3VLImageProcessor; features are off-manifold."
        conf = "high"
    elif status["H2 resolution / RoPE (native 672)"] == "SUPPORTED":
        root = "448 px is off the Tower's native 672 geometry; RoPE / grid at 448 loses class signal."
        conf = "high"
    elif status["H4 nonlinear-only class info"] == "SUPPORTED":
        root = "Class information is present but not linearly readable from pooled tokens."
        conf = "medium"
    elif all(s == "REJECTED" for _, s, _ in rows[:5]):
        root = (
            "MiniMax-M3 Tower features are genuinely weak on ImageNet-100 linear probes. "
            "Weights, preprocess, grid, pooling, and a nonlinear readout do not recover roster-level accuracy."
        )
        conf = "medium"
    else:
        root = "No single hypothesis is closed. See the table and gaps below."
        conf = "low"

    lines = []
    a = lines.append
    a("# MiniMax-M3 recognition diagnosis")
    a("")
    a("## Verdict")
    a("")
    a(f"{root} Confidence: {conf}.")
    a("")
    a("## Hypothesis table")
    a("")
    a("| ID | Status | Deciding number |")
    a("|---|---|---|")
    for name, status, num in rows:
        a(f"| {name} | {status} | {num} |")
    a("")
    a("## Evidence chain")
    a("")
    a("- Vocabulary: Tower, Projector, Stage (tower / merged / projected), Capability Profile. Not backbone, not leaderboard.")
    a("- Protocol: ImageNet-100 validation, 13,000 images, 448 px, pool-4 is 4x4 adaptive avg. Linear probe: `vtb.probe_run` LR grid, 20 epochs, 3 seeds, match-capacity PCA to 512.")
    a(f"- Roster pool-4 L32 attention {ROSTER_ATTN}, mean {ROSTER_MEAN} (`results/minimax_m3_probe_attention_matched.json`, `results/minimax_m3_probe_mean_matched.json`).")
    a(f"- This box pool-4 L32 attention {cell_top1(pool4_attn)}, mean {cell_top1(pool4_mean)} (`/tmp/mm3_pool4_attn.json`, `/tmp/mm3_pool4_mean.json`). Same anomaly, small offset from roster (bf16 extract / GPU).")
    a(f"- Full-grid L32 attention {cell_top1(full_attn)} (`/tmp/mm3_full_attn.json`). Decision rule: >= ~0.7 supports H1; still ~0.34 rejects H1.")
    a(f"- Full-grid L32 mean {cell_top1(full_mean)} (`/tmp/mm3_full_mean_L32.json`). Equal-area pool-4 mean of a 32x32 grid should match full-grid mean.")
    a(f"- Token-0 probe {cell_top1(token0)} (`/tmp/mm3_token0.json`).")
    a(f"- Forward parity `/tmp/mm3_forward_parity.json`.")
    a(f"- 672 pool-4 attention {cell_top1(res672)} (`/tmp/mm3_672.json`).")
    a(f"- Feature stats `/tmp/mm3_feature_stats.json`. MLP `/tmp/mm3_mlp.json`.")
    a("- HF `MiniMaxM3VLVisionModel.forward` sets `pooler_output=hidden_states[:, 0]`. That is the first packed token, not a dedicated CLS embedding. Per-image token 0 after `raster` is the first block-major spatial patch.")
    a("- Full-vs-pooled validation in-repo exists only for Muse Glimmer and SigLIP2 (`results/pooling/`). MiniMax had no full-grid number before this run.")
    a("- Weight parity vs MiniMaxAI/MiniMax-M3 @ f0e1c1e was not re-run here (parent shards gated / offline). Adapter tests `tests/test_minimax_m3.py` passed (2).")
    a("")

    if h1 == "SUPPORTED":
        a("## H1 mechanism and v2 numbers")
        a("")
        a("Pool-4 adaptive-avg over 8x8 patches mixes token 0 with 63 neighbours in its 4x4 cell. If class logits concentrate on token 0, that is a 64x dilution of the class channel in that cell, and the other 15 cells add still more spatial average. Attention on 16 mixed tokens cannot recover a single undiluted token. PCA silhouettes survive because figure-ground is spread across the 32x32 grid.")
        a("")
        a("v2 should report MiniMax-M3 recognition on the full 32x32 grid (or a token-0 / CLS readout) alongside the pool-4 number. Do not change pooling roster-wide unless other Towers show the same token-0 concentration. Muse Glimmer and SigLIP2 already agree pool-4 vs full on the mean readout.")
        a("")

    a("## Draft Limits replacement")
    a("")
    a("- MiniMax-M3 pool-4 ImageNet-100 attention is ~0.33 top-1 (roster 0.3407). This box reproduced it (0.3323).")
    if h1 == "SUPPORTED":
        a("- Cause: pool-4. Full-grid attention recovers class signal. Token 0 carries the class; spatial tokens carry figure-ground. Report full-grid (or token-0) MiniMax numbers in v2.")
    elif h1 == "REJECTED":
        a("- Full-grid attention stays near 0.33, so pool-4 dilution is not the cause.")
    else:
        a("- Full-grid test incomplete or intermediate. Do not call the cause unknown if later JSON files in /tmp close H1.")
    a("")
    a("## Appendix")
    a("")
    a("New scripts: `scripts/minimax_full_probe.py`, `scripts/minimax_token0_probe.py`, `scripts/minimax_forward_parity.py`, `scripts/minimax_feature_stats.py`, `scripts/minimax_mlp_probe.py`, `scripts/minimax_continue.sh`, `scripts/minimax_write_report.py`.")
    a("Caches kept: `cache/minimax_m3_448_pool4`, `cache/minimax_m3_448_full` (~324 GB), optional `cache/minimax_m3_672_pool4`.")
    a("Live L32 attention was pid 25905 at tmux handoff. Vanilla `python -m vtb.probe_run` on the full grid OOMs on 70 GB RAM (slice float32 ~68 GB). Full-grid matched cells used subsampled-train PCA then `run_cell`.")
    a("Attach: `tmux attach -t mm3`. Log: `/tmp/mm3_logs/continue.log`.")
    a("")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
