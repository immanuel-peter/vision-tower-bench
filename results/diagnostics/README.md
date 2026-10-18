# Diagnostics

These experiments test the measurement procedure. They are retained even when
they fail or do not support a conclusion.

| Directory | Purpose |
|---|---|
| [pooling/](pooling/) | Initial pooled-versus-full-token checks on SigLIP2 and Muse Glimmer. |
| [pooling-diagnostic/](pooling-diagnostic/) | Attention failures, larger training budgets, and independent baseline repeats. Full-token attention validation remains inconclusive. |
| [pooling-roster/](pooling-roster/) | Mean-readout checks for the remaining original Towers, including Kimi extraction batch-size variants. |
| [grid_search/](grid_search/) | Targeted wider geometry learning-rate searches, not a replacement for the roster matrices. |
| [nemotron35-bolt-tower/](nemotron35-bolt-tower/) | Final Tower recognition for Nemotron Bolt against Nemotron Super on the identical protocol. Bolt's matrices cover only `merged` and `projected`. |
| [dinov3-equal-pixels/](dinov3-equal-pixels/) | DINOv3 at 448 px (28x28), the same input pixels as DINOv2, against its 512 px equal-grid primary run. Final Tower layer, matched capacity. |
| [dinov3-conditioning/](dinov3-conditioning/) | Token norm and spectrum statistics for DINOv3 and DINOv2 at every probed depth, on twelve fixed DIODE images. Descriptive only. |
| [mimo26-attention-sinks/](mimo26-attention-sinks/) | MiMo 2.6 Pro and MiMo 2.6 Flash vision attention: the Hugging Face key-0 sink bias against the pinned vLLM Triton kernel in BF16. |

See the [results index](../README.md) for the primary measurements and correction
scope. Repeated files are independent-run records, even when their bytes match.
