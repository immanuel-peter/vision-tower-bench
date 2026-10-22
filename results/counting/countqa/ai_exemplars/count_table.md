> **EXPERIMENTAL, SUPPLEMENTARY: AI-AUTHORED EXEMPLARS (NOT HUMAN-REVIEWED), 325 OF 439 QUESTIONS**: experimental, supplementary: AI-authored exemplars (not human-reviewed), 325 of 439 questions numbers, not final results.

## CountQA-derived transfer (experimental, supplementary; frozen FSC-147 heads)

Experimental, supplementary: exemplar-based frozen-feature counting transfer on a curated CountQA-derived subset. These are not official CountQA benchmark scores, not MLLM question answering, and not comparable to published CountQA MLLM results. Results with AI-authored exemplars (not human-reviewed) carry that provenance in their JSON. Exact match: floor(pred + 0.5) == answer.

| Model | Arm | Capacity | Stage | L | Test IDs | n | MAE ± sd | RMSE ± sd | pooled RMSE | bias | exact | LR | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek_v41 | free | raw | `tower` | 32 | all | 325 | 5.39 ± 0.11 | 12.52 ± 0.46 | 12.53 | 1.76 | 0.103 | n/a | complete |
| deepseek_v41 | free | raw | `tower` | 32 | eligible | 325 | 5.39 ± 0.11 | 12.52 ± 0.46 | 12.53 | 1.76 | 0.103 | n/a | complete |
| deepseek_v41 | free | raw | `merged` | 32 | all | 325 | 5.90 ± 0.61 | 13.90 ± 4.42 | 14.59 | 1.42 | 0.080 | n/a | complete |
| deepseek_v41 | free | raw | `projected` | 32 | all | 325 | 6.99 ± 0.57 | 12.01 ± 1.52 | 12.11 | 3.76 | 0.078 | n/a | complete |
| deepseek_v41 | free | matched | `tower` | 32 | all | 325 | 5.64 ± 0.13 | 11.18 ± 0.52 | 11.19 | 2.21 | 0.102 | n/a | complete |
| deepseek_v41 | free | matched | `tower` | 32 | eligible | 325 | 5.64 ± 0.13 | 11.18 ± 0.52 | 11.19 | 2.21 | 0.102 | n/a | complete |
| deepseek_v41 | free | matched | `merged` | 32 | all | 325 | 5.83 ± 0.06 | 14.24 ± 0.22 | 14.25 | 1.78 | 0.070 | n/a | complete |
| deepseek_v41 | free | matched | `projected` | 32 | all | 325 | 6.85 ± 0.20 | 17.36 ± 1.43 | 17.42 | 2.60 | 0.079 | n/a | complete |
| deepseek_v41 | exemplar | raw | `tower` | 32 | eligible | 325 | 4.93 ± 0.05 | 9.11 ± 0.53 | 9.12 | 1.94 | 0.130 | n/a | complete |
| deepseek_v41 | exemplar | matched | `tower` | 32 | eligible | 325 | 5.74 ± 0.18 | 10.39 ± 0.45 | 10.40 | 2.56 | 0.103 | n/a | complete |
| dinov2 | free | raw | `tower` | 24 | all | 325 | 7.45 ± 0.15 | 23.88 ± 0.15 | 23.88 | -3.21 | 0.059 | n/a | complete |
| dinov2 | free | raw | `tower` | 24 | eligible | 325 | 7.45 ± 0.15 | 23.88 ± 0.15 | 23.88 | -3.21 | 0.059 | n/a | complete |
| dinov2 | free | matched | `tower` | 24 | all | 325 | 7.71 ± 0.07 | 24.09 ± 0.09 | 24.09 | -3.48 | 0.056 | n/a | complete |
| dinov2 | free | matched | `tower` | 24 | eligible | 325 | 7.71 ± 0.07 | 24.09 ± 0.09 | 24.09 | -3.48 | 0.056 | n/a | complete |
| dinov2 | exemplar | raw | `tower` | 24 | eligible | 325 | 7.65 ± 0.04 | 23.91 ± 0.15 | 23.92 | -2.07 | 0.070 | n/a | complete |
| dinov2 | exemplar | matched | `tower` | 24 | eligible | 325 | 7.70 ± 0.13 | 24.02 ± 0.18 | 24.02 | -2.65 | 0.070 | n/a | complete |
| dinov3 | free | raw | `tower` | 24 | all | 325 | 8.88 ± 0.48 | 24.37 ± 0.52 | 24.38 | -2.07 | 0.051 | n/a | complete |
| dinov3 | free | raw | `tower` | 24 | eligible | 325 | 8.88 ± 0.48 | 24.37 ± 0.52 | 24.38 | -2.07 | 0.051 | n/a | complete |
| dinov3 | free | matched | `tower` | 24 | all | 325 | 8.45 ± 0.34 | 24.40 ± 0.15 | 24.40 | -4.60 | 0.045 | n/a | complete |
| dinov3 | free | matched | `tower` | 24 | eligible | 325 | 8.45 ± 0.34 | 24.40 ± 0.15 | 24.40 | -4.60 | 0.045 | n/a | complete |
| dinov3 | exemplar | raw | `tower` | 24 | eligible | 325 | 8.86 ± 0.26 | 24.43 ± 0.37 | 24.43 | -2.51 | 0.057 | n/a | complete |
| dinov3 | exemplar | matched | `tower` | 24 | eligible | 325 | 8.33 ± 0.30 | 23.42 ± 0.13 | 23.42 | -4.14 | 0.059 | n/a | complete |
| gemma4 | free | raw | `tower` | 27 | all | 325 | 7.86 ± 0.13 | 24.27 ± 0.10 | 24.27 | 0.68 | 0.067 | n/a | complete |
| gemma4 | free | raw | `tower` | 27 | eligible | 325 | 7.86 ± 0.13 | 24.27 ± 0.10 | 24.27 | 0.68 | 0.067 | n/a | complete |
| gemma4 | free | raw | `merged` | 27 | all | 325 | 9.18 ± 0.36 | 25.06 ± 0.41 | 25.06 | 1.74 | 0.072 | n/a | complete |
| gemma4 | free | raw | `projected` | 27 | all | 325 | 8.26 ± 0.12 | 24.00 ± 0.67 | 24.01 | 1.30 | 0.065 | n/a | complete |
| gemma4 | free | matched | `tower` | 27 | all | 325 | 7.64 ± 0.11 | 23.72 ± 0.17 | 23.72 | -0.12 | 0.055 | n/a | complete |
| gemma4 | free | matched | `tower` | 27 | eligible | 325 | 7.64 ± 0.11 | 23.72 ± 0.17 | 23.72 | -0.12 | 0.055 | n/a | complete |
| gemma4 | free | matched | `merged` | 27 | all | 325 | 9.67 ± 0.11 | 25.07 ± 0.10 | 25.07 | 2.12 | 0.061 | n/a | complete |
| gemma4 | free | matched | `projected` | 27 | all | 325 | 8.95 ± 0.19 | 24.69 ± 0.13 | 24.69 | 1.07 | 0.066 | n/a | complete |
| gemma4 | exemplar | raw | `tower` | 27 | eligible | 325 | 8.09 ± 0.22 | 24.38 ± 0.19 | 24.38 | 0.90 | 0.061 | n/a | complete |
| gemma4 | exemplar | matched | `tower` | 27 | eligible | 325 | 7.83 ± 0.31 | 23.76 ± 0.09 | 23.76 | 0.60 | 0.069 | n/a | complete |
| glm5 | free | raw | `tower` | 24 | all | 325 | 7.07 ± 0.44 | 20.29 ± 0.30 | 20.29 | -0.17 | 0.073 | n/a | complete |
| glm5 | free | raw | `tower` | 24 | eligible | 325 | 7.07 ± 0.44 | 20.29 ± 0.30 | 20.29 | -0.17 | 0.073 | n/a | complete |
| glm5 | free | raw | `merged` | 24 | all | 325 | 7.62 ± 0.17 | 22.85 ± 0.12 | 22.85 | 1.11 | 0.065 | n/a | complete |
| glm5 | free | raw | `projected` | 24 | all | 325 | 6.88 ± 0.38 | 15.90 ± 0.53 | 15.91 | 0.12 | 0.067 | n/a | complete |
| glm5 | free | matched | `tower` | 24 | all | 325 | 6.87 ± 0.10 | 21.22 ± 0.20 | 21.22 | -0.67 | 0.055 | n/a | complete |
| glm5 | free | matched | `tower` | 24 | eligible | 325 | 6.87 ± 0.10 | 21.22 ± 0.20 | 21.22 | -0.67 | 0.055 | n/a | complete |
| glm5 | free | matched | `merged` | 24 | all | 325 | 6.93 ± 0.12 | 21.88 ± 0.28 | 21.88 | -2.00 | 0.063 | n/a | complete |
| glm5 | free | matched | `projected` | 24 | all | 325 | 6.10 ± 0.08 | 17.90 ± 0.80 | 17.92 | -0.99 | 0.077 | n/a | complete |
| glm5 | exemplar | raw | `tower` | 24 | eligible | 325 | 7.24 ± 0.38 | 20.48 ± 0.55 | 20.49 | 0.90 | 0.078 | n/a | complete |
| glm5 | exemplar | matched | `tower` | 24 | eligible | 325 | 6.89 ± 0.07 | 21.21 ± 0.22 | 21.21 | -0.21 | 0.079 | n/a | complete |
| kimi_k26 | free | raw | `tower` | 27 | all | 325 | 5.69 ± 0.21 | 15.02 ± 0.13 | 15.02 | 1.20 | 0.115 | n/a | complete |
| kimi_k26 | free | raw | `tower` | 27 | eligible | 325 | 5.69 ± 0.21 | 15.02 ± 0.13 | 15.02 | 1.20 | 0.115 | n/a | complete |
| kimi_k26 | free | raw | `merged` | 27 | all | 325 | 6.96 ± 0.36 | 14.56 ± 0.58 | 14.57 | 2.69 | 0.075 | n/a | complete |
| kimi_k26 | free | raw | `projected` | 27 | all | 325 | 5.83 ± 0.21 | 12.04 ± 1.20 | 12.10 | 1.44 | 0.101 | n/a | complete |
| kimi_k26 | free | matched | `tower` | 27 | all | 325 | 5.47 ± 0.06 | 14.83 ± 0.74 | 14.85 | 0.07 | 0.103 | n/a | complete |
| kimi_k26 | free | matched | `tower` | 27 | eligible | 325 | 5.47 ± 0.06 | 14.83 ± 0.74 | 14.85 | 0.07 | 0.103 | n/a | complete |
| kimi_k26 | free | matched | `merged` | 27 | all | 325 | 5.90 ± 0.16 | 13.73 ± 0.66 | 13.75 | 0.62 | 0.069 | n/a | complete |
| kimi_k26 | free | matched | `projected` | 27 | all | 325 | 5.71 ± 0.25 | 14.19 ± 1.21 | 14.24 | -0.67 | 0.079 | n/a | complete |
| kimi_k26 | exemplar | raw | `tower` | 27 | eligible | 325 | 6.46 ± 0.39 | 15.50 ± 1.04 | 15.54 | 1.69 | 0.086 | n/a | complete |
| kimi_k26 | exemplar | matched | `tower` | 27 | eligible | 325 | 5.61 ± 0.21 | 15.62 ± 0.71 | 15.63 | -0.07 | 0.102 | n/a | complete |
| mimo26_flash | free | raw | `tower` | 28 | all | 325 | 10.53 ± 0.69 | 23.63 ± 0.44 | 23.63 | 6.29 | 0.042 | n/a | complete |
| mimo26_flash | free | raw | `tower` | 28 | eligible | 325 | 10.53 ± 0.69 | 23.63 ± 0.44 | 23.63 | 6.29 | 0.042 | n/a | complete |
| mimo26_flash | free | raw | `merged` | 28 | all | 325 | 8.50 ± 0.41 | 20.00 ± 0.44 | 20.00 | 4.18 | 0.061 | n/a | complete |
| mimo26_flash | free | raw | `projected` | 28 | all | 325 | 8.02 ± 0.48 | 19.01 ± 1.36 | 19.06 | 4.50 | 0.062 | n/a | complete |
| mimo26_flash | free | matched | `tower` | 28 | all | 325 | 9.45 ± 0.40 | 23.52 ± 0.39 | 23.53 | 5.36 | 0.045 | n/a | complete |
| mimo26_flash | free | matched | `tower` | 28 | eligible | 325 | 9.45 ± 0.40 | 23.52 ± 0.39 | 23.53 | 5.36 | 0.045 | n/a | complete |
| mimo26_flash | free | matched | `merged` | 28 | all | 325 | 6.49 ± 0.15 | 20.51 ± 0.15 | 20.52 | 1.08 | 0.105 | n/a | complete |
| mimo26_flash | free | matched | `projected` | 28 | all | 325 | 6.26 ± 0.17 | 18.43 ± 0.83 | 18.44 | 2.09 | 0.094 | n/a | complete |
| mimo26_flash | exemplar | raw | `tower` | 28 | eligible | 325 | 11.74 ± 0.98 | 24.46 ± 0.47 | 24.46 | 7.91 | 0.029 | n/a | complete |
| mimo26_flash | exemplar | matched | `tower` | 28 | eligible | 325 | 12.13 ± 2.28 | 24.93 ± 1.18 | 24.96 | 8.70 | 0.028 | n/a | complete |
| mimo26_pro | free | raw | `tower` | 28 | all | 325 | 10.27 ± 0.20 | 23.22 ± 0.19 | 23.22 | 5.84 | 0.049 | n/a | complete |
| mimo26_pro | free | raw | `tower` | 28 | eligible | 325 | 10.27 ± 0.20 | 23.22 ± 0.19 | 23.22 | 5.84 | 0.049 | n/a | complete |
| mimo26_pro | free | raw | `merged` | 28 | all | 325 | 8.14 ± 0.43 | 19.64 ± 0.54 | 19.65 | 3.90 | 0.056 | n/a | complete |
| mimo26_pro | free | raw | `projected` | 28 | all | 325 | 7.91 ± 0.15 | 18.60 ± 0.48 | 18.60 | 4.28 | 0.066 | n/a | complete |
| mimo26_pro | free | matched | `tower` | 28 | all | 325 | 9.74 ± 0.24 | 23.64 ± 0.19 | 23.64 | 5.37 | 0.063 | n/a | complete |
| mimo26_pro | free | matched | `tower` | 28 | eligible | 325 | 9.74 ± 0.24 | 23.64 ± 0.19 | 23.64 | 5.37 | 0.063 | n/a | complete |
| mimo26_pro | free | matched | `merged` | 28 | all | 325 | 7.18 ± 0.28 | 21.67 ± 0.37 | 21.67 | 2.13 | 0.068 | n/a | complete |
| mimo26_pro | free | matched | `projected` | 28 | all | 325 | 7.03 ± 0.37 | 19.13 ± 0.21 | 19.13 | 2.26 | 0.094 | n/a | complete |
| mimo26_pro | exemplar | raw | `tower` | 28 | eligible | 325 | 11.71 ± 0.99 | 24.20 ± 0.59 | 24.20 | 7.76 | 0.034 | n/a | complete |
| mimo26_pro | exemplar | matched | `tower` | 28 | eligible | 325 | 11.63 ± 1.65 | 24.72 ± 1.04 | 24.74 | 7.90 | 0.029 | n/a | complete |
| minimax_m3 | free | raw | `tower` | 32 | all | 325 | 16.34 ± 0.47 | 24.28 ± 0.59 | 24.29 | 14.03 | 0.018 | n/a | complete |
| minimax_m3 | free | raw | `tower` | 32 | eligible | 325 | 16.34 ± 0.47 | 24.28 ± 0.59 | 24.29 | 14.03 | 0.018 | n/a | complete |
| minimax_m3 | free | raw | `merged` | 32 | all | 325 | 17.21 ± 1.47 | 25.33 ± 1.47 | 25.37 | 14.51 | 0.024 | n/a | complete |
| minimax_m3 | free | raw | `projected` | 32 | all | 325 | 9.18 ± 0.25 | 19.24 ± 0.91 | 19.26 | 3.82 | 0.040 | n/a | complete |
| minimax_m3 | free | matched | `tower` | 32 | all | 325 | 19.97 ± 1.10 | 31.04 ± 1.19 | 31.06 | 17.33 | 0.027 | n/a | complete |
| minimax_m3 | free | matched | `tower` | 32 | eligible | 325 | 19.97 ± 1.10 | 31.04 ± 1.19 | 31.06 | 17.33 | 0.027 | n/a | complete |
| minimax_m3 | free | matched | `merged` | 32 | all | 325 | 22.51 ± 1.43 | 34.07 ± 2.34 | 34.15 | 20.00 | 0.016 | n/a | complete |
| minimax_m3 | free | matched | `projected` | 32 | all | 325 | 8.32 ± 0.39 | 19.48 ± 0.73 | 19.50 | 1.73 | 0.063 | n/a | complete |
| minimax_m3 | exemplar | raw | `tower` | 32 | eligible | 325 | 16.50 ± 1.57 | 24.07 ± 1.77 | 24.13 | 14.65 | 0.021 | n/a | complete |
| minimax_m3 | exemplar | matched | `tower` | 32 | eligible | 325 | 20.77 ± 1.14 | 30.94 ± 1.17 | 30.96 | 18.64 | 0.021 | n/a | complete |
| moonvit_v2 | free | raw | `tower` | 27 | all | 325 | 6.84 ± 0.05 | 18.94 ± 0.75 | 18.96 | 1.89 | 0.086 | n/a | complete |
| moonvit_v2 | free | raw | `tower` | 27 | eligible | 325 | 6.84 ± 0.05 | 18.94 ± 0.75 | 18.96 | 1.89 | 0.086 | n/a | complete |
| moonvit_v2 | free | raw | `merged` | 27 | all | 325 | 8.11 ± 0.57 | 20.32 ± 1.23 | 20.36 | 3.93 | 0.086 | n/a | complete |
| moonvit_v2 | free | raw | `projected` | 27 | all | 325 | 8.58 ± 1.17 | 21.56 ± 1.58 | 21.61 | 4.50 | 0.067 | n/a | complete |
| moonvit_v2 | free | matched | `tower` | 27 | all | 325 | 7.09 ± 0.17 | 18.01 ± 1.15 | 18.05 | 3.61 | 0.087 | n/a | complete |
| moonvit_v2 | free | matched | `tower` | 27 | eligible | 325 | 7.09 ± 0.17 | 18.01 ± 1.15 | 18.05 | 3.61 | 0.087 | n/a | complete |
| moonvit_v2 | free | matched | `merged` | 27 | all | 325 | 6.91 ± 0.48 | 20.55 ± 0.42 | 20.55 | 2.73 | 0.086 | n/a | complete |
| moonvit_v2 | free | matched | `projected` | 27 | all | 325 | 6.98 ± 0.28 | 19.41 ± 0.84 | 19.43 | 2.49 | 0.077 | n/a | complete |
| moonvit_v2 | exemplar | raw | `tower` | 27 | eligible | 325 | 6.78 ± 0.20 | 19.61 ± 0.28 | 19.61 | 1.88 | 0.092 | n/a | complete |
| moonvit_v2 | exemplar | matched | `tower` | 27 | eligible | 325 | 7.44 ± 0.11 | 20.35 ± 0.34 | 20.35 | 3.67 | 0.088 | n/a | complete |
| muse_glimmer | free | raw | `tower` | 50 | all | 325 | 6.73 ± 0.31 | 21.34 ± 0.24 | 21.34 | -0.36 | 0.087 | n/a | complete |
| muse_glimmer | free | raw | `tower` | 50 | eligible | 325 | 6.73 ± 0.31 | 21.34 ± 0.24 | 21.34 | -0.36 | 0.087 | n/a | complete |
| muse_glimmer | free | raw | `merged` | 50 | all | 325 | 7.07 ± 0.25 | 22.14 ± 0.39 | 22.14 | -0.42 | 0.086 | n/a | complete |
| muse_glimmer | free | raw | `projected` | 50 | all | 325 | 5.54 ± 0.10 | 19.28 ± 0.27 | 19.29 | -0.67 | 0.091 | n/a | complete |
| muse_glimmer | free | matched | `tower` | 50 | all | 325 | 6.10 ± 0.15 | 20.54 ± 0.24 | 20.54 | -0.35 | 0.078 | n/a | complete |
| muse_glimmer | free | matched | `tower` | 50 | eligible | 325 | 6.10 ± 0.15 | 20.54 ± 0.24 | 20.54 | -0.35 | 0.078 | n/a | complete |
| muse_glimmer | free | matched | `merged` | 50 | all | 325 | 6.26 ± 0.13 | 21.15 ± 0.32 | 21.15 | -0.91 | 0.079 | n/a | complete |
| muse_glimmer | free | matched | `projected` | 50 | all | 325 | 5.23 ± 0.08 | 19.40 ± 0.90 | 19.43 | -2.23 | 0.109 | n/a | complete |
| muse_glimmer | exemplar | raw | `tower` | 50 | eligible | 325 | 6.60 ± 0.57 | 21.87 ± 0.04 | 21.87 | -0.24 | 0.078 | n/a | complete |
| muse_glimmer | exemplar | matched | `tower` | 50 | eligible | 325 | 6.22 ± 0.09 | 20.76 ± 0.24 | 20.76 | 0.63 | 0.091 | n/a | complete |
| nemotron35_bolt | free | raw | `tower` | 32 | all | 325 | 5.91 ± 0.10 | 18.00 ± 0.16 | 18.00 | -2.06 | 0.072 | n/a | complete |
| nemotron35_bolt | free | raw | `tower` | 32 | eligible | 325 | 5.91 ± 0.10 | 18.00 ± 0.16 | 18.00 | -2.06 | 0.072 | n/a | complete |
| nemotron35_bolt | free | raw | `merged` | 32 | all | 325 | 6.93 ± 0.06 | 18.74 ± 0.57 | 18.74 | -2.68 | 0.055 | n/a | complete |
| nemotron35_bolt | free | raw | `projected` | 32 | all | 325 | 6.68 ± 0.14 | 18.10 ± 0.65 | 18.11 | 0.41 | 0.088 | n/a | complete |
| nemotron35_bolt | free | matched | `tower` | 32 | all | 325 | 6.21 ± 0.07 | 18.42 ± 0.19 | 18.42 | -2.87 | 0.077 | n/a | complete |
| nemotron35_bolt | free | matched | `tower` | 32 | eligible | 325 | 6.21 ± 0.07 | 18.42 ± 0.19 | 18.42 | -2.87 | 0.077 | n/a | complete |
| nemotron35_bolt | free | matched | `merged` | 32 | all | 325 | 6.19 ± 0.10 | 16.40 ± 0.81 | 16.42 | -2.03 | 0.076 | n/a | complete |
| nemotron35_bolt | free | matched | `projected` | 32 | all | 325 | 5.86 ± 0.25 | 19.68 ± 0.50 | 19.69 | -0.46 | 0.078 | n/a | complete |
| nemotron35_bolt | exemplar | raw | `tower` | 32 | eligible | 325 | 6.02 ± 0.20 | 17.13 ± 0.97 | 17.16 | -0.90 | 0.076 | n/a | complete |
| nemotron35_bolt | exemplar | matched | `tower` | 32 | eligible | 325 | 6.35 ± 0.14 | 19.72 ± 1.19 | 19.75 | -3.53 | 0.073 | n/a | complete |
| nemotron35_super | free | raw | `tower` | 32 | all | 325 | 6.10 ± 0.24 | 18.12 ± 0.46 | 18.12 | -2.98 | 0.070 | n/a | complete |
| nemotron35_super | free | raw | `tower` | 32 | eligible | 325 | 6.10 ± 0.24 | 18.12 ± 0.46 | 18.12 | -2.98 | 0.070 | n/a | complete |
| nemotron35_super | free | raw | `merged` | 32 | all | 325 | 6.65 ± 0.14 | 18.75 ± 0.20 | 18.75 | -2.36 | 0.062 | n/a | complete |
| nemotron35_super | free | raw | `projected` | 32 | all | 325 | 6.59 ± 0.16 | 17.60 ± 0.56 | 17.61 | 0.80 | 0.079 | n/a | complete |
| nemotron35_super | free | matched | `tower` | 32 | all | 325 | 6.13 ± 0.12 | 18.59 ± 0.26 | 18.60 | -2.72 | 0.070 | n/a | complete |
| nemotron35_super | free | matched | `tower` | 32 | eligible | 325 | 6.13 ± 0.12 | 18.59 ± 0.26 | 18.60 | -2.72 | 0.070 | n/a | complete |
| nemotron35_super | free | matched | `merged` | 32 | all | 325 | 6.34 ± 0.19 | 15.57 ± 1.23 | 15.62 | -3.02 | 0.055 | n/a | complete |
| nemotron35_super | free | matched | `projected` | 32 | all | 325 | 6.99 ± 0.22 | 19.25 ± 0.62 | 19.26 | 1.74 | 0.077 | n/a | complete |
| nemotron35_super | exemplar | raw | `tower` | 32 | eligible | 325 | 6.16 ± 0.09 | 17.77 ± 0.55 | 17.78 | -1.63 | 0.077 | n/a | complete |
| nemotron35_super | exemplar | matched | `tower` | 32 | eligible | 325 | 6.48 ± 0.15 | 18.69 ± 1.64 | 18.76 | -4.08 | 0.057 | n/a | complete |
| nemotron_omni | free | raw | `tower` | 32 | all | 325 | 7.38 ± 0.65 | 15.14 ± 0.86 | 15.17 | 2.49 | 0.062 | n/a | complete |
| nemotron_omni | free | raw | `tower` | 32 | eligible | 325 | 7.38 ± 0.65 | 15.14 ± 0.86 | 15.17 | 2.49 | 0.062 | n/a | complete |
| nemotron_omni | free | raw | `merged` | 32 | all | 325 | 8.67 ± 1.32 | 16.51 ± 1.09 | 16.55 | 3.93 | 0.049 | n/a | complete |
| nemotron_omni | free | raw | `projected` | 32 | all | 325 | 6.45 ± 0.27 | 15.75 ± 0.53 | 15.76 | 1.84 | 0.085 | n/a | complete |
| nemotron_omni | free | matched | `tower` | 32 | all | 325 | 6.59 ± 0.11 | 18.06 ± 0.55 | 18.07 | -0.93 | 0.064 | n/a | complete |
| nemotron_omni | free | matched | `tower` | 32 | eligible | 325 | 6.59 ± 0.11 | 18.06 ± 0.55 | 18.07 | -0.93 | 0.064 | n/a | complete |
| nemotron_omni | free | matched | `merged` | 32 | all | 325 | 7.13 ± 0.37 | 18.34 ± 0.32 | 18.35 | 0.67 | 0.070 | n/a | complete |
| nemotron_omni | free | matched | `projected` | 32 | all | 325 | 6.66 ± 0.33 | 18.52 ± 0.38 | 18.53 | 0.80 | 0.084 | n/a | complete |
| nemotron_omni | exemplar | raw | `tower` | 32 | eligible | 325 | 7.73 ± 0.60 | 15.29 ± 0.45 | 15.30 | 3.34 | 0.083 | n/a | complete |
| nemotron_omni | exemplar | matched | `tower` | 32 | eligible | 325 | 6.51 ± 0.34 | 17.69 ± 0.37 | 17.69 | -1.38 | 0.077 | n/a | complete |
| qwen3_5 | free | raw | `tower` | 27 | all | 325 | 8.89 ± 0.45 | 22.06 ± 0.13 | 22.06 | 3.98 | 0.052 | n/a | complete |
| qwen3_5 | free | raw | `tower` | 27 | eligible | 325 | 8.89 ± 0.45 | 22.06 ± 0.13 | 22.06 | 3.98 | 0.052 | n/a | complete |
| qwen3_5 | free | raw | `merged` | 27 | all | 325 | 9.37 ± 0.16 | 23.50 ± 0.16 | 23.50 | 2.48 | 0.058 | n/a | complete |
| qwen3_5 | free | raw | `projected` | 27 | all | 325 | 6.65 ± 0.22 | 19.33 ± 0.61 | 19.34 | -0.32 | 0.067 | n/a | complete |
| qwen3_5 | free | matched | `tower` | 27 | all | 325 | 6.07 ± 0.12 | 14.84 ± 0.55 | 14.85 | -0.75 | 0.077 | n/a | complete |
| qwen3_5 | free | matched | `tower` | 27 | eligible | 325 | 6.07 ± 0.12 | 14.84 ± 0.55 | 14.85 | -0.75 | 0.077 | n/a | complete |
| qwen3_5 | free | matched | `merged` | 27 | all | 325 | 8.20 ± 0.06 | 19.98 ± 0.13 | 19.98 | 0.63 | 0.046 | n/a | complete |
| qwen3_5 | free | matched | `projected` | 27 | all | 325 | 6.34 ± 0.15 | 19.32 ± 1.17 | 19.36 | -0.81 | 0.080 | n/a | complete |
| qwen3_5 | exemplar | raw | `tower` | 27 | eligible | 325 | 9.60 ± 1.12 | 22.57 ± 0.92 | 22.59 | 5.09 | 0.053 | n/a | complete |
| qwen3_5 | exemplar | matched | `tower` | 27 | eligible | 325 | 6.38 ± 0.11 | 15.46 ± 0.63 | 15.48 | -0.72 | 0.086 | n/a | complete |
| siglip2 | free | raw | `tower` | 27 | all | 325 | 7.69 ± 0.17 | 19.19 ± 0.94 | 19.21 | -0.42 | 0.070 | n/a | complete |
| siglip2 | free | raw | `tower` | 27 | eligible | 325 | 7.69 ± 0.17 | 19.19 ± 0.94 | 19.21 | -0.42 | 0.070 | n/a | complete |
| siglip2 | free | matched | `tower` | 27 | all | 325 | 7.73 ± 0.19 | 20.35 ± 0.27 | 20.35 | 0.24 | 0.047 | n/a | complete |
| siglip2 | free | matched | `tower` | 27 | eligible | 325 | 7.73 ± 0.19 | 20.35 ± 0.27 | 20.35 | 0.24 | 0.047 | n/a | complete |
| siglip2 | exemplar | raw | `tower` | 27 | eligible | 325 | 7.75 ± 0.15 | 20.11 ± 0.63 | 20.12 | -1.05 | 0.067 | n/a | complete |
| siglip2 | exemplar | matched | `tower` | 27 | eligible | 325 | 7.68 ± 0.08 | 21.03 ± 0.34 | 21.03 | -0.16 | 0.070 | n/a | complete |

### Per count bin (true answer; † marks n < 30)

| Model | Arm | Capacity | Stage | Test IDs | le50 n | le50 MAE | le50 RMSE | 50to200 n | 50to200 MAE | 50to200 RMSE | gt200 n | gt200 MAE | gt200 RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek_v41 | free | raw | `tower` | all | 321 | 4.55 | 7.10 | 3† | 43.69 | 54.01 | 1† | 160.92 | 160.92 |
| deepseek_v41 | free | raw | `tower` | eligible | 321 | 4.55 | 7.10 | 3† | 43.69 | 54.01 | 1† | 160.92 | 160.92 |
| deepseek_v41 | free | raw | `merged` | all | 321 | 5.01 | 7.83 | 3† | 46.01 | 56.60 | 1† | 171.03 | 171.03 |
| deepseek_v41 | free | raw | `projected` | all | 321 | 6.47 | 10.07 | 3† | 37.48 | 45.22 | 1† | 84.31 | 84.31 |
| deepseek_v41 | free | matched | `tower` | all | 321 | 4.99 | 7.56 | 3† | 34.17 | 42.54 | 1† | 129.44 | 129.44 |
| deepseek_v41 | free | matched | `tower` | eligible | 321 | 4.99 | 7.56 | 3† | 34.17 | 42.54 | 1† | 129.44 | 129.44 |
| deepseek_v41 | free | matched | `merged` | all | 321 | 4.87 | 7.08 | 3† | 42.56 | 51.46 | 1† | 204.60 | 204.60 |
| deepseek_v41 | free | matched | `projected` | all | 321 | 5.79 | 8.88 | 3† | 37.19 | 47.82 | 1† | 255.91 | 255.91 |
| deepseek_v41 | exemplar | raw | `tower` | eligible | 321 | 4.45 | 6.95 | 3† | 40.38 | 49.63 | 1† | 50.71 | 50.71 |
| deepseek_v41 | exemplar | matched | `tower` | eligible | 321 | 5.25 | 8.44 | 3† | 30.43 | 38.91 | 1† | 87.48 | 87.48 |
| dinov2 | free | raw | `tower` | all | 321 | 5.82 | 8.06 | 3† | 55.11 | 67.82 | 1† | 388.14 | 388.14 |
| dinov2 | free | raw | `tower` | eligible | 321 | 5.82 | 8.06 | 3† | 55.11 | 67.82 | 1† | 388.14 | 388.14 |
| dinov2 | free | matched | `tower` | all | 321 | 6.09 | 8.35 | 3† | 52.35 | 62.19 | 1† | 393.21 | 393.21 |
| dinov2 | free | matched | `tower` | eligible | 321 | 6.09 | 8.35 | 3† | 52.35 | 62.19 | 1† | 393.21 | 393.21 |
| dinov2 | exemplar | raw | `tower` | eligible | 321 | 6.03 | 8.41 | 3† | 52.70 | 62.03 | 1† | 389.39 | 389.39 |
| dinov2 | exemplar | matched | `tower` | eligible | 321 | 6.10 | 8.43 | 3† | 50.30 | 58.86 | 1† | 392.66 | 392.66 |
| dinov3 | free | raw | `tower` | all | 321 | 7.29 | 11.49 | 3† | 58.65 | 67.79 | 1† | 369.78 | 369.78 |
| dinov3 | free | raw | `tower` | eligible | 321 | 7.29 | 11.49 | 3† | 58.65 | 67.79 | 1† | 369.78 | 369.78 |
| dinov3 | free | matched | `tower` | all | 321 | 6.79 | 9.60 | 3† | 59.63 | 68.87 | 1† | 386.55 | 386.55 |
| dinov3 | free | matched | `tower` | eligible | 321 | 6.79 | 9.60 | 3† | 59.63 | 68.87 | 1† | 386.55 | 386.55 |
| dinov3 | exemplar | raw | `tower` | eligible | 321 | 7.28 | 11.84 | 3† | 57.41 | 66.52 | 1† | 368.27 | 368.27 |
| dinov3 | exemplar | matched | `tower` | eligible | 321 | 6.74 | 9.59 | 3† | 59.42 | 68.45 | 1† | 366.88 | 366.88 |
| gemma4 | free | raw | `tower` | all | 321 | 6.40 | 10.44 | 3† | 38.98 | 49.75 | 1† | 385.97 | 385.97 |
| gemma4 | free | raw | `tower` | eligible | 321 | 6.40 | 10.44 | 3† | 38.98 | 49.75 | 1† | 385.97 | 385.97 |
| gemma4 | free | raw | `merged` | all | 321 | 7.67 | 12.86 | 3† | 48.71 | 56.65 | 1† | 375.68 | 375.68 |
| gemma4 | free | raw | `projected` | all | 321 | 6.75 | 11.33 | 3† | 50.72 | 59.45 | 1† | 367.91 | 367.91 |
| gemma4 | free | matched | `tower` | all | 321 | 6.21 | 10.08 | 3† | 37.11 | 47.36 | 1† | 378.85 | 378.85 |
| gemma4 | free | matched | `tower` | eligible | 321 | 6.21 | 10.08 | 3† | 37.11 | 47.36 | 1† | 378.85 | 378.85 |
| gemma4 | free | matched | `merged` | all | 321 | 8.18 | 13.47 | 3† | 49.75 | 59.14 | 1† | 368.00 | 368.00 |
| gemma4 | free | matched | `projected` | all | 321 | 7.41 | 12.31 | 3† | 52.36 | 59.54 | 1† | 372.38 | 372.38 |
| gemma4 | exemplar | raw | `tower` | eligible | 321 | 6.65 | 10.92 | 3† | 37.03 | 47.52 | 1† | 384.64 | 384.64 |
| gemma4 | exemplar | matched | `tower` | eligible | 321 | 6.43 | 10.46 | 3† | 34.68 | 46.82 | 1† | 376.32 | 376.32 |
| glm5 | free | raw | `tower` | all | 321 | 5.96 | 9.20 | 3† | 20.93 | 22.84 | 1† | 323.78 | 323.78 |
| glm5 | free | raw | `tower` | eligible | 321 | 5.96 | 9.20 | 3† | 20.93 | 22.84 | 1† | 323.78 | 323.78 |
| glm5 | free | raw | `merged` | all | 321 | 6.39 | 9.66 | 3† | 18.58 | 22.51 | 1† | 371.76 | 371.76 |
| glm5 | free | raw | `projected` | all | 321 | 6.00 | 9.49 | 3† | 30.80 | 37.94 | 1† | 218.75 | 218.75 |
| glm5 | free | matched | `tower` | all | 321 | 5.57 | 8.12 | 3† | 34.04 | 44.19 | 1† | 345.36 | 345.36 |
| glm5 | free | matched | `tower` | eligible | 321 | 5.57 | 8.12 | 3† | 34.04 | 44.19 | 1† | 345.36 | 345.36 |
| glm5 | free | matched | `merged` | all | 321 | 5.54 | 8.24 | 3† | 39.38 | 49.85 | 1† | 355.40 | 355.40 |
| glm5 | free | matched | `projected` | all | 321 | 4.98 | 7.36 | 3† | 33.50 | 45.21 | 1† | 283.53 | 283.53 |
| glm5 | exemplar | raw | `tower` | eligible | 321 | 6.08 | 9.32 | 3† | 25.57 | 31.58 | 1† | 324.59 | 324.59 |
| glm5 | exemplar | matched | `tower` | eligible | 321 | 5.60 | 8.31 | 3† | 31.92 | 40.72 | 1† | 345.07 | 345.07 |
| kimi_k26 | free | raw | `tower` | all | 321 | 4.86 | 9.18 | 3† | 28.00 | 36.15 | 1† | 205.03 | 205.03 |
| kimi_k26 | free | raw | `tower` | eligible | 321 | 4.86 | 9.18 | 3† | 28.00 | 36.15 | 1† | 205.03 | 205.03 |
| kimi_k26 | free | raw | `merged` | all | 321 | 6.37 | 12.11 | 3† | 27.44 | 34.00 | 1† | 135.38 | 135.38 |
| kimi_k26 | free | raw | `projected` | all | 321 | 5.32 | 9.40 | 3† | 22.01 | 30.97 | 1† | 119.87 | 119.87 |
| kimi_k26 | free | matched | `tower` | all | 321 | 4.59 | 8.25 | 3† | 30.42 | 37.38 | 1† | 212.99 | 212.99 |
| kimi_k26 | free | matched | `tower` | eligible | 321 | 4.59 | 8.25 | 3† | 30.42 | 37.38 | 1† | 212.99 | 212.99 |
| kimi_k26 | free | matched | `merged` | all | 321 | 5.13 | 8.88 | 3† | 31.89 | 40.17 | 1† | 175.81 | 175.81 |
| kimi_k26 | free | matched | `projected` | all | 321 | 4.85 | 8.54 | 3† | 35.54 | 41.57 | 1† | 191.42 | 191.42 |
| kimi_k26 | exemplar | raw | `tower` | eligible | 321 | 5.65 | 10.40 | 3† | 29.10 | 37.00 | 1† | 198.20 | 198.20 |
| kimi_k26 | exemplar | matched | `tower` | eligible | 321 | 4.68 | 7.97 | 3† | 29.19 | 35.94 | 1† | 234.50 | 234.50 |
| mimo26_flash | free | raw | `tower` | all | 321 | 9.40 | 13.09 | 3† | 17.49 | 23.94 | 1† | 352.92 | 352.92 |
| mimo26_flash | free | raw | `tower` | eligible | 321 | 9.40 | 13.09 | 3† | 17.49 | 23.94 | 1† | 352.92 | 352.92 |
| mimo26_flash | free | raw | `merged` | all | 321 | 7.55 | 13.16 | 3† | 24.25 | 28.76 | 1† | 267.54 | 267.54 |
| mimo26_flash | free | raw | `projected` | all | 321 | 7.10 | 11.32 | 3† | 18.70 | 23.19 | 1† | 272.85 | 272.85 |
| mimo26_flash | free | matched | `tower` | all | 321 | 8.24 | 11.67 | 3† | 20.42 | 27.65 | 1† | 365.81 | 365.81 |
| mimo26_flash | free | matched | `tower` | eligible | 321 | 8.24 | 11.67 | 3† | 20.42 | 27.65 | 1† | 365.81 | 365.81 |
| mimo26_flash | free | matched | `merged` | all | 321 | 5.23 | 8.83 | 3† | 34.79 | 42.43 | 1† | 326.05 | 326.05 |
| mimo26_flash | free | matched | `projected` | all | 321 | 5.19 | 8.36 | 3† | 26.56 | 32.99 | 1† | 290.84 | 290.84 |
| mimo26_flash | exemplar | raw | `tower` | eligible | 321 | 10.63 | 14.67 | 3† | 16.93 | 22.11 | 1† | 351.29 | 351.29 |
| mimo26_flash | exemplar | matched | `tower` | eligible | 321 | 10.94 | 14.33 | 3† | 22.95 | 30.34 | 1† | 362.72 | 362.72 |
| mimo26_pro | free | raw | `tower` | all | 321 | 9.18 | 12.98 | 3† | 14.81 | 21.63 | 1† | 345.99 | 345.99 |
| mimo26_pro | free | raw | `tower` | eligible | 321 | 9.18 | 12.98 | 3† | 14.81 | 21.63 | 1† | 345.99 | 345.99 |
| mimo26_pro | free | raw | `merged` | all | 321 | 7.17 | 12.62 | 3† | 25.63 | 33.72 | 1† | 266.12 | 266.12 |
| mimo26_pro | free | raw | `projected` | all | 321 | 7.01 | 11.65 | 3† | 22.06 | 33.91 | 1† | 255.64 | 255.64 |
| mimo26_pro | free | matched | `tower` | all | 321 | 8.54 | 12.16 | 3† | 20.80 | 29.21 | 1† | 362.69 | 362.69 |
| mimo26_pro | free | matched | `tower` | eligible | 321 | 8.54 | 12.16 | 3† | 20.80 | 29.21 | 1† | 362.69 | 362.69 |
| mimo26_pro | free | matched | `merged` | all | 321 | 5.90 | 9.89 | 3† | 32.54 | 42.83 | 1† | 340.04 | 340.04 |
| mimo26_pro | free | matched | `projected` | all | 321 | 5.90 | 9.73 | 3† | 34.29 | 44.78 | 1† | 286.58 | 286.58 |
| mimo26_pro | exemplar | raw | `tower` | eligible | 321 | 10.64 | 14.81 | 3† | 15.10 | 20.39 | 1† | 343.87 | 343.87 |
| mimo26_pro | exemplar | matched | `tower` | eligible | 321 | 10.46 | 14.09 | 3† | 19.36 | 28.40 | 1† | 362.64 | 362.64 |
| minimax_m3 | free | raw | `tower` | all | 321 | 15.89 | 23.05 | 3† | 24.72 | 27.79 | 1† | 136.30 | 136.30 |
| minimax_m3 | free | raw | `tower` | eligible | 321 | 15.89 | 23.05 | 3† | 24.72 | 27.79 | 1† | 136.30 | 136.30 |
| minimax_m3 | free | raw | `merged` | all | 321 | 16.59 | 23.28 | 3† | 32.13 | 33.05 | 1† | 171.61 | 171.61 |
| minimax_m3 | free | raw | `projected` | all | 321 | 8.18 | 12.49 | 3† | 34.68 | 38.17 | 1† | 256.18 | 256.18 |
| minimax_m3 | free | matched | `tower` | all | 321 | 19.55 | 30.41 | 3† | 37.30 | 41.10 | 1† | 104.08 | 104.08 |
| minimax_m3 | free | matched | `tower` | eligible | 321 | 19.55 | 30.41 | 3† | 37.30 | 41.10 | 1† | 104.08 | 104.08 |
| minimax_m3 | free | matched | `merged` | all | 321 | 22.06 | 32.93 | 3† | 25.29 | 34.43 | 1† | 157.54 | 157.54 |
| minimax_m3 | free | matched | `projected` | all | 321 | 7.22 | 10.58 | 3† | 32.13 | 33.06 | 1† | 290.08 | 290.08 |
| minimax_m3 | exemplar | raw | `tower` | eligible | 321 | 16.21 | 23.56 | 3† | 23.33 | 24.85 | 1† | 88.30 | 88.30 |
| minimax_m3 | exemplar | matched | `tower` | eligible | 321 | 20.55 | 30.71 | 3† | 27.78 | 29.86 | 1† | 68.21 | 68.21 |
| moonvit_v2 | free | raw | `tower` | all | 321 | 5.93 | 12.17 | 3† | 21.43 | 26.68 | 1† | 256.54 | 256.54 |
| moonvit_v2 | free | raw | `tower` | eligible | 321 | 5.93 | 12.17 | 3† | 21.43 | 26.68 | 1† | 256.54 | 256.54 |
| moonvit_v2 | free | raw | `merged` | all | 321 | 7.28 | 15.70 | 3† | 25.23 | 33.95 | 1† | 225.09 | 225.09 |
| moonvit_v2 | free | raw | `projected` | all | 321 | 7.60 | 15.65 | 3† | 29.91 | 36.50 | 1† | 258.06 | 258.06 |
| moonvit_v2 | free | matched | `tower` | all | 321 | 6.19 | 11.93 | 3† | 26.63 | 35.35 | 1† | 235.61 | 235.61 |
| moonvit_v2 | free | matched | `tower` | eligible | 321 | 6.19 | 11.93 | 3† | 26.63 | 35.35 | 1† | 235.61 | 235.61 |
| moonvit_v2 | free | matched | `merged` | all | 321 | 5.85 | 12.53 | 3† | 26.93 | 37.76 | 1† | 286.21 | 286.21 |
| moonvit_v2 | free | matched | `projected` | all | 321 | 5.91 | 11.42 | 3† | 32.15 | 40.58 | 1† | 274.68 | 274.68 |
| moonvit_v2 | exemplar | raw | `tower` | eligible | 321 | 5.81 | 11.64 | 3† | 19.97 | 30.02 | 1† | 280.20 | 280.20 |
| moonvit_v2 | exemplar | matched | `tower` | eligible | 321 | 6.43 | 12.78 | 3† | 25.47 | 36.67 | 1† | 278.18 | 278.18 |
| muse_glimmer | free | raw | `tower` | all | 321 | 5.49 | 7.73 | 3† | 23.69 | 28.28 | 1† | 355.46 | 355.46 |
| muse_glimmer | free | raw | `tower` | eligible | 321 | 5.49 | 7.73 | 3† | 23.69 | 28.28 | 1† | 355.46 | 355.46 |
| muse_glimmer | free | raw | `merged` | all | 321 | 5.76 | 8.27 | 3† | 27.37 | 33.03 | 1† | 366.12 | 366.12 |
| muse_glimmer | free | raw | `projected` | all | 321 | 4.34 | 6.41 | 3† | 28.29 | 34.43 | 1† | 322.58 | 322.58 |
| muse_glimmer | free | matched | `tower` | all | 321 | 4.87 | 7.22 | 3† | 25.79 | 30.07 | 1† | 342.88 | 342.88 |
| muse_glimmer | free | matched | `tower` | eligible | 321 | 4.87 | 7.22 | 3† | 25.79 | 30.07 | 1† | 342.88 | 342.88 |
| muse_glimmer | free | matched | `merged` | all | 321 | 4.95 | 7.11 | 3† | 30.71 | 36.51 | 1† | 353.66 | 353.66 |
| muse_glimmer | free | matched | `projected` | all | 321 | 3.98 | 5.57 | 3† | 32.14 | 39.57 | 1† | 328.08 | 328.08 |
| muse_glimmer | exemplar | raw | `tower` | eligible | 321 | 5.33 | 7.89 | 3† | 22.45 | 27.10 | 1† | 364.45 | 364.45 |
| muse_glimmer | exemplar | matched | `tower` | eligible | 321 | 4.99 | 7.45 | 3† | 24.38 | 28.57 | 1† | 346.17 | 346.17 |
| nemotron35_bolt | free | raw | `tower` | all | 321 | 4.69 | 6.51 | 3† | 41.95 | 51.41 | 1† | 289.45 | 289.45 |
| nemotron35_bolt | free | raw | `tower` | eligible | 321 | 4.69 | 6.51 | 3† | 41.95 | 51.41 | 1† | 289.45 | 289.45 |
| nemotron35_bolt | free | raw | `merged` | all | 321 | 5.73 | 8.35 | 3† | 41.65 | 50.38 | 1† | 289.75 | 289.75 |
| nemotron35_bolt | free | raw | `projected` | all | 321 | 5.58 | 8.46 | 3† | 33.87 | 44.39 | 1† | 278.38 | 278.38 |
| nemotron35_bolt | free | matched | `tower` | all | 321 | 5.01 | 7.69 | 3† | 39.88 | 48.66 | 1† | 290.09 | 290.09 |
| nemotron35_bolt | free | matched | `tower` | eligible | 321 | 5.01 | 7.69 | 3† | 39.88 | 48.66 | 1† | 290.09 | 290.09 |
| nemotron35_bolt | free | matched | `merged` | all | 321 | 5.15 | 7.87 | 3† | 37.77 | 47.98 | 1† | 245.65 | 245.65 |
| nemotron35_bolt | free | matched | `projected` | all | 321 | 4.58 | 6.47 | 3† | 36.12 | 46.36 | 1† | 325.55 | 325.55 |
| nemotron35_bolt | exemplar | raw | `tower` | eligible | 321 | 4.90 | 7.08 | 3† | 38.78 | 49.17 | 1† | 267.53 | 267.53 |
| nemotron35_bolt | exemplar | matched | `tower` | eligible | 321 | 5.06 | 7.87 | 3† | 42.14 | 50.66 | 1† | 313.56 | 313.56 |
| nemotron35_super | free | raw | `tower` | all | 321 | 4.86 | 6.52 | 3† | 44.73 | 55.18 | 1† | 289.60 | 289.60 |
| nemotron35_super | free | raw | `tower` | eligible | 321 | 4.86 | 6.52 | 3† | 44.73 | 55.18 | 1† | 289.60 | 289.60 |
| nemotron35_super | free | raw | `merged` | all | 321 | 5.42 | 7.80 | 3† | 42.49 | 51.13 | 1† | 294.61 | 294.61 |
| nemotron35_super | free | raw | `projected` | all | 321 | 5.46 | 8.11 | 3† | 40.89 | 52.80 | 1† | 266.54 | 266.54 |
| nemotron35_super | free | matched | `tower` | all | 321 | 4.91 | 7.54 | 3† | 39.56 | 48.90 | 1† | 294.85 | 294.85 |
| nemotron35_super | free | matched | `tower` | eligible | 321 | 4.91 | 7.54 | 3† | 39.56 | 48.90 | 1† | 294.85 | 294.85 |
| nemotron35_super | free | matched | `merged` | all | 321 | 5.33 | 7.61 | 3† | 40.52 | 50.18 | 1† | 228.72 | 228.72 |
| nemotron35_super | free | matched | `projected` | all | 321 | 5.75 | 8.46 | 3† | 43.24 | 55.17 | 1† | 296.94 | 296.94 |
| nemotron35_super | exemplar | raw | `tower` | eligible | 321 | 4.98 | 7.05 | 3† | 41.24 | 50.99 | 1† | 280.63 | 280.63 |
| nemotron35_super | exemplar | matched | `tower` | eligible | 321 | 5.27 | 8.02 | 3† | 41.25 | 50.28 | 1† | 290.47 | 290.47 |
| nemotron_omni | free | raw | `tower` | all | 321 | 6.54 | 9.44 | 3† | 33.59 | 45.47 | 1† | 197.94 | 197.94 |
| nemotron_omni | free | raw | `tower` | eligible | 321 | 6.54 | 9.44 | 3† | 33.59 | 45.47 | 1† | 197.94 | 197.94 |
| nemotron_omni | free | raw | `merged` | all | 321 | 7.87 | 11.65 | 3† | 32.10 | 45.38 | 1† | 193.92 | 193.92 |
| nemotron_omni | free | raw | `projected` | all | 321 | 5.45 | 7.63 | 3† | 38.17 | 47.47 | 1† | 234.69 | 234.69 |
| nemotron_omni | free | matched | `tower` | all | 321 | 5.46 | 7.60 | 3† | 35.40 | 47.85 | 1† | 283.89 | 283.89 |
| nemotron_omni | free | matched | `tower` | eligible | 321 | 5.46 | 7.60 | 3† | 35.40 | 47.85 | 1† | 283.89 | 283.89 |
| nemotron_omni | free | matched | `merged` | all | 321 | 5.97 | 8.49 | 3† | 40.64 | 52.03 | 1† | 279.26 | 279.26 |
| nemotron_omni | free | matched | `projected` | all | 321 | 5.50 | 7.89 | 3† | 36.13 | 48.14 | 1† | 290.46 | 290.46 |
| nemotron_omni | exemplar | raw | `tower` | eligible | 321 | 6.96 | 10.28 | 3† | 29.20 | 40.47 | 1† | 191.29 | 191.29 |
| nemotron_omni | exemplar | matched | `tower` | eligible | 321 | 5.40 | 7.55 | 3† | 36.00 | 49.22 | 1† | 275.68 | 275.68 |
| qwen3_5 | free | raw | `tower` | all | 321 | 7.86 | 12.88 | 3† | 14.64 | 16.58 | 1† | 321.93 | 321.93 |
| qwen3_5 | free | raw | `tower` | eligible | 321 | 7.86 | 12.88 | 3† | 14.64 | 16.58 | 1† | 321.93 | 321.93 |
| qwen3_5 | free | raw | `merged` | all | 321 | 8.26 | 13.28 | 3† | 14.93 | 20.10 | 1† | 348.70 | 348.70 |
| qwen3_5 | free | raw | `projected` | all | 321 | 5.41 | 8.33 | 3† | 41.47 | 51.55 | 1† | 301.63 | 301.63 |
| qwen3_5 | free | matched | `tower` | all | 321 | 5.12 | 7.42 | 3† | 36.93 | 44.43 | 1† | 218.78 | 218.78 |
| qwen3_5 | free | matched | `tower` | eligible | 321 | 5.12 | 7.42 | 3† | 36.93 | 44.43 | 1† | 218.78 | 218.78 |
| qwen3_5 | free | matched | `merged` | all | 321 | 7.09 | 11.92 | 3† | 36.29 | 41.58 | 1† | 281.05 | 281.05 |
| qwen3_5 | free | matched | `projected` | all | 321 | 5.07 | 7.88 | 3† | 43.82 | 56.09 | 1† | 302.75 | 302.75 |
| qwen3_5 | exemplar | raw | `tower` | eligible | 321 | 8.60 | 14.36 | 3† | 14.80 | 16.29 | 1† | 312.63 | 312.63 |
| qwen3_5 | exemplar | matched | `tower` | eligible | 321 | 5.41 | 8.28 | 3† | 37.21 | 43.65 | 1† | 223.52 | 223.52 |
| siglip2 | free | raw | `tower` | all | 321 | 6.47 | 9.64 | 3† | 45.87 | 56.68 | 1† | 283.00 | 283.00 |
| siglip2 | free | raw | `tower` | eligible | 321 | 6.47 | 9.64 | 3† | 45.87 | 56.68 | 1† | 283.00 | 283.00 |
| siglip2 | free | matched | `tower` | all | 321 | 6.47 | 9.51 | 3† | 41.78 | 52.15 | 1† | 312.08 | 312.08 |
| siglip2 | free | matched | `tower` | eligible | 321 | 6.47 | 9.51 | 3† | 41.78 | 52.15 | 1† | 312.08 | 312.08 |
| siglip2 | exemplar | raw | `tower` | eligible | 321 | 6.48 | 9.56 | 3† | 44.42 | 53.84 | 1† | 305.45 | 305.45 |
| siglip2 | exemplar | matched | `tower` | eligible | 321 | 6.38 | 9.46 | 3† | 40.18 | 51.07 | 1† | 327.41 | 327.41 |

