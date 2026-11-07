"""Regenerate only default panels whose new measurements are complete."""

from pathlib import Path

from scripts.figures import label_budget, occlusion, relative_depth

output = Path("results/diagnostics/kimi-k31-pro-checkpoint/ready-default-figures")
output.mkdir(parents=True, exist_ok=True)
for name, draw in (("label-budget", label_budget), ("occlusion", occlusion), ("relative-depth-semantic", relative_depth)):
    for suffix in ("svg", "png", "pdf"):
        draw(output / f"{name}.{suffix}")
