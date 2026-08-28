# Adapters return tokens in raster order

Two Towers on the roster do not hand back tokens in raster order, and neither says so.
Both would corrupt the geometry pillar while leaving the semantic pillar looking healthy,
because an attention pool over a scrambled grid still trains, and a dense decoder over one
does not.

Qwen orders patches by 2x2 merge block. At 448 square its raw sequence runs
`0, 1, 28, 29, 2, 3` rather than `0, 1, 2, 3`. The order is required: it is what makes the
merger's `view` group the right four tokens. So the adapter feeds Qwen its own order and
undoes the grouping on the `tower` Stage only.

Muse Glimmer permutes tokens into attention windows before its blocks and reverses that
after them. A forward hook on an intermediate block therefore sees window order. The
adapter recomputes the same permutation and inverts it. Its deepest `tower` point hooks
`ln_post` instead, which runs after the reversal, so that point needs no correction and
keeps its norm.

Every adapter now yields Stages that reshape to a square grid. `tests/test_adapters.py`
pins Qwen's correction against a synthetic grid of known patch positions, which fails if
the permutation is ever dropped or inverted the wrong way.

The related trap is that spatial merging groups a 2x2 square of the grid, not four tokens
in a row. A flat reshape of the `tower` Stage does not reproduce `merged` and must not be
used to check that merging is lossless. The Kimi K2.6 test asserts both directions: the
2x2 regroup matches bit for bit, and the flat reshape does not.
