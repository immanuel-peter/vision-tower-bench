# Diagnostics

These experiments test the measurement procedure. They are retained even when
they fail or do not support a conclusion.

| Directory | Purpose |
|---|---|
| [pooling/](pooling/) | Initial pooled-versus-full-token checks on SigLIP2 and Muse Glimmer. |
| [pooling-diagnostic/](pooling-diagnostic/) | Attention failures, larger training budgets, and independent baseline repeats. Full-token attention validation remains inconclusive. |
| [pooling-roster/](pooling-roster/) | Mean-readout checks for the remaining original Towers, including Kimi extraction batch-size variants. |
| [grid_search/](grid_search/) | Targeted wider geometry learning-rate searches, not a replacement for the roster matrices. |

See the [results index](../README.md) for the primary measurements and correction
scope. Repeated files are independent-run records, even when their bytes match.
