## FSC-147 test (official split)

MAE and RMSE: mean ± sd over seeds of each seed's value. Pooled RMSE: sqrt of the mean squared error over all seed x image rows. Test IDs `eligible`: the exemplar arm's test IDs (all three release boxes valid after the crop); the free Tower is repeated on them for the arm comparison. Partial results use finished cells of an incomplete run.

| Model | Arm | Capacity | Stage | L | Test IDs | n | MAE ± sd | RMSE ± sd | pooled RMSE | bias | LR | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek_v41 | free | raw | `tower` | 32 | all | 1190 | 13.99 ± 0.22 | 84.28 ± 0.80 | 84.29 | -8.45 | 0.001 | complete |
| deepseek_v41 | free | raw | `tower` | 32 | eligible | 1019 | 13.60 ± 0.19 | 85.15 ± 0.45 | 85.15 | -8.21 | 0.001 | complete |
| deepseek_v41 | free | raw | `merged` | 32 | all | 1190 | 14.24 ± 0.48 | 82.58 ± 1.44 | 82.59 | -7.96 | 0.001 | complete |
| deepseek_v41 | free | raw | `projected` | 32 | all | 1190 | 15.34 ± 0.32 | 82.73 ± 1.84 | 82.75 | -7.38 | 0.001 | complete |
| deepseek_v41 | free | matched | `tower` | 32 | all | 1190 | 14.56 ± 0.34 | 85.47 ± 0.74 | 85.47 | -8.28 | 0.001 | complete |
| deepseek_v41 | free | matched | `tower` | 32 | eligible | 1019 | 14.05 ± 0.39 | 85.98 ± 0.64 | 85.98 | -7.84 | 0.001 | complete |
| deepseek_v41 | free | matched | `merged` | 32 | all | 1190 | 14.83 ± 0.24 | 84.51 ± 0.94 | 84.51 | -7.40 | 0.001 | complete |
| deepseek_v41 | free | matched | `projected` | 32 | all | 1190 | 14.96 ± 0.28 | 84.29 ± 0.92 | 84.30 | -7.91 | 0.003 | complete |
| deepseek_v41 | exemplar | raw | `tower` | 32 | eligible | 1019 | 13.36 ± 0.17 | 84.12 ± 0.14 | 84.12 | -7.12 | 0.001 | complete |
| deepseek_v41 | exemplar | matched | `tower` | 32 | eligible | 1019 | 14.35 ± 0.35 | 85.48 ± 1.06 | 85.49 | -6.89 | 0.003 | complete |
| dinov2 | free | raw | `tower` | 24 | all | 1190 | 23.64 ± 0.62 | 95.31 ± 0.69 | 95.32 | -12.79 | 0.0003 | complete |
| dinov2 | free | raw | `tower` | 24 | eligible | 1019 | 22.64 ± 0.67 | 91.13 ± 0.86 | 91.13 | -11.68 | 0.0003 | complete |
| dinov2 | free | matched | `tower` | 24 | all | 1190 | 24.01 ± 0.01 | 96.41 ± 0.23 | 96.41 | -14.21 | 0.003 | complete |
| dinov2 | free | matched | `tower` | 24 | eligible | 1019 | 23.19 ± 0.06 | 92.13 ± 0.30 | 92.13 | -13.53 | 0.003 | complete |
| dinov2 | exemplar | raw | `tower` | 24 | eligible | 1019 | 23.16 ± 0.23 | 90.21 ± 0.53 | 90.21 | -12.77 | 0.0001 | complete |
| dinov2 | exemplar | matched | `tower` | 24 | eligible | 1019 | 22.95 ± 0.26 | 91.64 ± 0.04 | 91.64 | -11.14 | 0.001 | complete |
| dinov3 | free | raw | `tower` | 24 | all | 1190 | 22.93 ± 1.01 | 84.89 ± 0.36 | 84.89 | -6.27 | 0.003 | complete |
| dinov3 | free | raw | `tower` | 24 | eligible | 1019 | 22.54 ± 0.96 | 86.13 ± 0.31 | 86.13 | -5.56 | 0.003 | complete |
| dinov3 | free | matched | `tower` | 24 | all | 1190 | 24.60 ± 1.06 | 87.82 ± 1.85 | 87.84 | -10.98 | 0.001 | complete |
| dinov3 | free | matched | `tower` | 24 | eligible | 1019 | 23.91 ± 1.06 | 87.04 ± 1.51 | 87.05 | -10.33 | 0.001 | complete |
| dinov3 | exemplar | raw | `tower` | 24 | eligible | 1019 | 24.75 ± 0.65 | 85.55 ± 0.59 | 85.55 | 1.22 | 0.0003 | complete |
| dinov3 | exemplar | matched | `tower` | 24 | eligible | 1019 | 25.66 ± 0.53 | 86.61 ± 0.68 | 86.61 | -5.16 | 0.003 | complete |
| gemma4 | free | raw | `tower` | 27 | all | 1190 | 16.82 ± 0.40 | 88.19 ± 0.63 | 88.19 | -9.33 | 0.0003 | complete |
| gemma4 | free | raw | `tower` | 27 | eligible | 1019 | 16.40 ± 0.44 | 88.06 ± 0.54 | 88.06 | -8.83 | 0.0003 | complete |
| gemma4 | free | raw | `merged` | 27 | all | 1190 | 16.71 ± 0.21 | 86.45 ± 0.47 | 86.45 | -7.44 | 0.001 | complete |
| gemma4 | free | raw | `projected` | 27 | all | 1190 | 16.40 ± 0.43 | 86.77 ± 1.01 | 86.78 | -8.87 | 0.0003 | complete |
| gemma4 | free | matched | `tower` | 27 | all | 1190 | 17.54 ± 0.12 | 89.15 ± 0.23 | 89.15 | -10.23 | 0.001 | complete |
| gemma4 | free | matched | `tower` | 27 | eligible | 1019 | 17.32 ± 0.16 | 89.48 ± 0.10 | 89.48 | -9.88 | 0.001 | complete |
| gemma4 | free | matched | `merged` | 27 | all | 1190 | 17.70 ± 0.09 | 88.36 ± 0.57 | 88.36 | -10.33 | 0.003 | complete |
| gemma4 | free | matched | `projected` | 27 | all | 1190 | 16.65 ± 0.06 | 87.34 ± 0.80 | 87.34 | -9.77 | 0.001 | complete |
| gemma4 | exemplar | raw | `tower` | 27 | eligible | 1019 | 16.82 ± 0.16 | 87.11 ± 0.55 | 87.11 | -8.31 | 0.0003 | complete |
| gemma4 | exemplar | matched | `tower` | 27 | eligible | 1019 | 17.57 ± 0.22 | 88.58 ± 0.22 | 88.58 | -8.10 | 0.0003 | complete |
| glm5 | free | raw | `tower` | 24 | all | 1190 | 19.76 ± 0.15 | 87.74 ± 0.83 | 87.74 | -10.87 | 0.001 | complete |
| glm5 | free | raw | `tower` | 24 | eligible | 1019 | 19.10 ± 0.20 | 86.05 ± 0.62 | 86.05 | -10.57 | 0.001 | complete |
| glm5 | free | raw | `merged` | 24 | all | 1190 | 22.03 ± 0.12 | 93.66 ± 0.43 | 93.66 | -9.47 | 0.0003 | complete |
| glm5 | free | raw | `projected` | 24 | all | 1190 | 18.14 ± 0.79 | 88.21 ± 1.42 | 88.22 | -7.49 | 0.003 | complete |
| glm5 | free | matched | `tower` | 24 | all | 1190 | 19.37 ± 0.36 | 88.01 ± 0.67 | 88.01 | -11.15 | 0.001 | complete |
| glm5 | free | matched | `tower` | 24 | eligible | 1019 | 18.61 ± 0.42 | 85.47 ± 0.52 | 85.47 | -10.55 | 0.001 | complete |
| glm5 | free | matched | `merged` | 24 | all | 1190 | 17.88 ± 0.36 | 86.67 ± 0.61 | 86.67 | -11.81 | 0.003 | complete |
| glm5 | free | matched | `projected` | 24 | all | 1190 | 16.25 ± 0.07 | 83.94 ± 1.31 | 83.95 | -10.59 | 0.003 | complete |
| glm5 | exemplar | raw | `tower` | 24 | eligible | 1019 | 18.38 ± 0.75 | 85.49 ± 0.48 | 85.50 | -8.16 | 0.0003 | complete |
| glm5 | exemplar | matched | `tower` | 24 | eligible | 1019 | 19.06 ± 0.31 | 85.02 ± 0.92 | 85.03 | -9.41 | 0.001 | complete |
| kimi_k26 | free | raw | `tower` | 27 | all | 1190 | 16.18 ± 0.22 | 82.46 ± 0.82 | 82.47 | -7.54 | 0.0003 | complete |
| kimi_k26 | free | raw | `tower` | 27 | eligible | 1019 | 16.00 ± 0.24 | 84.06 ± 0.75 | 84.06 | -7.25 | 0.0003 | complete |
| kimi_k26 | free | raw | `merged` | 27 | all | 1190 | 17.46 ± 0.23 | 81.57 ± 0.18 | 81.57 | -8.36 | 0.0001 | complete |
| kimi_k26 | free | raw | `projected` | 27 | all | 1190 | 14.87 ± 0.18 | 80.47 ± 1.21 | 80.48 | -6.48 | 0.001 | complete |
| kimi_k26 | free | matched | `tower` | 27 | all | 1190 | 15.82 ± 0.36 | 79.58 ± 0.85 | 79.59 | -9.87 | 0.003 | complete |
| kimi_k26 | free | matched | `tower` | 27 | eligible | 1019 | 15.61 ± 0.43 | 81.63 ± 0.75 | 81.63 | -9.63 | 0.003 | complete |
| kimi_k26 | free | matched | `merged` | 27 | all | 1190 | 15.87 ± 0.33 | 82.27 ± 1.33 | 82.28 | -9.21 | 0.003 | complete |
| kimi_k26 | free | matched | `projected` | 27 | all | 1190 | 15.22 ± 0.20 | 82.86 ± 0.85 | 82.86 | -8.54 | 0.003 | complete |
| kimi_k26 | exemplar | raw | `tower` | 27 | eligible | 1019 | 16.92 ± 0.79 | 82.49 ± 0.44 | 82.49 | -5.18 | 0.003 | complete |
| kimi_k26 | exemplar | matched | `tower` | 27 | eligible | 1019 | 14.88 ± 0.22 | 79.53 ± 0.83 | 79.53 | -8.24 | 0.003 | complete |
| mimo26_flash | free | raw | `tower` | 28 | all | 1190 | 22.34 ± 0.52 | 93.75 ± 0.49 | 93.76 | -9.73 | 1e-05 | complete |
| mimo26_flash | free | raw | `tower` | 28 | eligible | 1019 | 21.81 ± 0.51 | 92.07 ± 0.34 | 92.07 | -9.57 | 1e-05 | complete |
| mimo26_flash | free | raw | `merged` | 28 | all | 1190 | 17.75 ± 0.60 | 88.85 ± 1.36 | 88.86 | -8.30 | 0.0003 | complete |
| mimo26_flash | free | raw | `projected` | 28 | all | 1190 | 17.53 ± 0.44 | 89.88 ± 0.92 | 89.89 | -6.77 | 0.001 | complete |
| mimo26_flash | free | matched | `tower` | 28 | all | 1190 | 22.71 ± 0.49 | 94.42 ± 0.18 | 94.42 | -6.48 | 1e-05 | complete |
| mimo26_flash | free | matched | `tower` | 28 | eligible | 1019 | 22.13 ± 0.49 | 92.76 ± 0.12 | 92.76 | -6.21 | 1e-05 | complete |
| mimo26_flash | free | matched | `merged` | 28 | all | 1190 | 15.95 ± 0.19 | 87.11 ± 1.13 | 87.12 | -7.71 | 0.003 | complete |
| mimo26_flash | free | matched | `projected` | 28 | all | 1190 | 16.41 ± 0.41 | 87.06 ± 0.36 | 87.06 | -7.05 | 0.001 | complete |
| mimo26_flash | exemplar | raw | `tower` | 28 | eligible | 1019 | 22.11 ± 0.49 | 91.75 ± 0.43 | 91.75 | -7.28 | 1e-05 | complete |
| mimo26_flash | exemplar | matched | `tower` | 28 | eligible | 1019 | 23.72 ± 2.14 | 92.75 ± 0.28 | 92.75 | -1.93 | 1e-05 | complete |
| mimo26_pro | free | raw | `tower` | 28 | all | 1190 | 21.90 ± 0.56 | 92.59 ± 0.43 | 92.59 | -9.58 | 1e-05 | complete |
| mimo26_pro | free | raw | `tower` | 28 | eligible | 1019 | 21.41 ± 0.60 | 91.51 ± 0.37 | 91.51 | -9.53 | 1e-05 | complete |
| mimo26_pro | free | raw | `merged` | 28 | all | 1190 | 17.01 ± 0.30 | 87.72 ± 0.58 | 87.72 | -7.72 | 0.0003 | complete |
| mimo26_pro | free | raw | `projected` | 28 | all | 1190 | 17.94 ± 0.16 | 90.42 ± 0.60 | 90.42 | -6.32 | 0.0003 | complete |
| mimo26_pro | free | matched | `tower` | 28 | all | 1190 | 21.89 ± 0.40 | 93.66 ± 0.50 | 93.66 | -8.94 | 1e-05 | complete |
| mimo26_pro | free | matched | `tower` | 28 | eligible | 1019 | 21.31 ± 0.46 | 92.23 ± 0.33 | 92.23 | -8.64 | 1e-05 | complete |
| mimo26_pro | free | matched | `merged` | 28 | all | 1190 | 16.85 ± 0.28 | 88.31 ± 0.16 | 88.31 | -7.90 | 0.0003 | complete |
| mimo26_pro | free | matched | `projected` | 28 | all | 1190 | 17.07 ± 0.25 | 88.85 ± 0.90 | 88.86 | -8.64 | 0.003 | complete |
| mimo26_pro | exemplar | raw | `tower` | 28 | eligible | 1019 | 21.74 ± 0.47 | 91.13 ± 0.27 | 91.13 | -6.60 | 1e-05 | complete |
| mimo26_pro | exemplar | matched | `tower` | 28 | eligible | 1019 | 22.53 ± 0.97 | 92.23 ± 0.66 | 92.23 | -4.01 | 1e-05 | complete |
| minimax_m3 | free | raw | `tower` | 32 | all | 1190 | 23.47 ± 0.08 | 92.37 ± 0.55 | 92.37 | -10.31 | 0.0001 | complete |
| minimax_m3 | free | raw | `tower` | 32 | eligible | 1019 | 23.08 ± 0.09 | 92.00 ± 0.52 | 92.00 | -10.08 | 0.0001 | complete |
| minimax_m3 | free | raw | `merged` | 32 | all | 1190 | 23.89 ± 0.17 | 91.65 ± 0.94 | 91.65 | -9.09 | 0.0001 | complete |
| minimax_m3 | free | raw | `projected` | 32 | all | 1190 | 23.76 ± 0.09 | 93.88 ± 0.29 | 93.88 | -13.00 | 0.0003 | complete |
| minimax_m3 | free | matched | `tower` | 32 | all | 1190 | 21.74 ± 0.21 | 87.93 ± 0.48 | 87.93 | -11.54 | 0.0003 | complete |
| minimax_m3 | free | matched | `tower` | 32 | eligible | 1019 | 21.45 ± 0.23 | 88.17 ± 0.50 | 88.17 | -11.23 | 0.0003 | complete |
| minimax_m3 | free | matched | `merged` | 32 | all | 1190 | 21.95 ± 0.28 | 87.82 ± 0.51 | 87.82 | -10.70 | 0.0003 | complete |
| minimax_m3 | free | matched | `projected` | 32 | all | 1190 | 23.04 ± 0.26 | 93.54 ± 0.33 | 93.54 | -12.90 | 0.001 | complete |
| minimax_m3 | exemplar | raw | `tower` | 32 | eligible | 1019 | 23.44 ± 0.19 | 90.47 ± 0.69 | 90.47 | -7.39 | 0.0003 | complete |
| minimax_m3 | exemplar | matched | `tower` | 32 | eligible | 1019 | 21.91 ± 0.14 | 87.87 ± 0.38 | 87.87 | -11.27 | 0.001 | complete |
| moonvit_v2 | free | raw | `tower` | 27 | all | 1190 | 14.11 ± 0.13 | 82.90 ± 0.55 | 82.90 | -6.57 | 0.003 | complete |
| moonvit_v2 | free | raw | `tower` | 27 | eligible | 1019 | 13.66 ± 0.12 | 82.77 ± 0.49 | 82.77 | -6.17 | 0.003 | complete |
| moonvit_v2 | free | raw | `merged` | 27 | all | 1190 | 14.26 ± 0.37 | 81.88 ± 0.37 | 81.89 | -5.26 | 0.003 | complete |
| moonvit_v2 | free | raw | `projected` | 27 | all | 1190 | 14.27 ± 0.16 | 83.90 ± 0.26 | 83.90 | -6.41 | 0.001 | complete |
| moonvit_v2 | free | matched | `tower` | 27 | all | 1190 | 13.93 ± 0.21 | 82.72 ± 0.30 | 82.72 | -7.83 | 0.001 | complete |
| moonvit_v2 | free | matched | `tower` | 27 | eligible | 1019 | 13.46 ± 0.26 | 82.56 ± 0.40 | 82.56 | -7.49 | 0.001 | complete |
| moonvit_v2 | free | matched | `merged` | 27 | all | 1190 | 13.61 ± 0.16 | 82.63 ± 0.55 | 82.63 | -6.56 | 0.003 | complete |
| moonvit_v2 | free | matched | `projected` | 27 | all | 1190 | 13.62 ± 0.14 | 84.88 ± 1.06 | 84.88 | -6.73 | 0.003 | complete |
| moonvit_v2 | exemplar | raw | `tower` | 27 | eligible | 1019 | 13.84 ± 0.14 | 81.90 ± 0.95 | 81.91 | -4.94 | 0.001 | complete |
| moonvit_v2 | exemplar | matched | `tower` | 27 | eligible | 1019 | 13.19 ± 0.06 | 83.01 ± 0.77 | 83.01 | -7.04 | 0.003 | complete |
| muse_glimmer | free | raw | `tower` | 50 | all | 1190 | 16.48 ± 0.44 | 85.70 ± 1.22 | 85.71 | -6.43 | 0.003 | complete |
| muse_glimmer | free | raw | `tower` | 50 | eligible | 1019 | 15.84 ± 0.51 | 84.39 ± 1.18 | 84.40 | -5.94 | 0.003 | complete |
| muse_glimmer | free | raw | `merged` | 50 | all | 1190 | 16.50 ± 0.48 | 86.49 ± 1.21 | 86.50 | -7.40 | 0.003 | complete |
| muse_glimmer | free | raw | `projected` | 50 | all | 1190 | 14.62 ± 0.20 | 86.79 ± 0.51 | 86.79 | -5.24 | 0.0001 | complete |
| muse_glimmer | free | matched | `tower` | 50 | all | 1190 | 15.49 ± 0.20 | 85.50 ± 0.34 | 85.50 | -5.90 | 0.003 | complete |
| muse_glimmer | free | matched | `tower` | 50 | eligible | 1019 | 14.74 ± 0.21 | 83.36 ± 0.40 | 83.36 | -5.38 | 0.003 | complete |
| muse_glimmer | free | matched | `merged` | 50 | all | 1190 | 16.20 ± 0.35 | 86.99 ± 0.59 | 87.00 | -5.85 | 0.001 | complete |
| muse_glimmer | free | matched | `projected` | 50 | all | 1190 | 14.84 ± 0.28 | 86.43 ± 0.67 | 86.44 | -7.10 | 0.001 | complete |
| muse_glimmer | exemplar | raw | `tower` | 50 | eligible | 1019 | 15.88 ± 0.46 | 84.16 ± 1.92 | 84.18 | -5.37 | 0.001 | complete |
| muse_glimmer | exemplar | matched | `tower` | 50 | eligible | 1019 | 14.93 ± 0.43 | 82.52 ± 0.53 | 82.52 | -3.82 | 0.003 | complete |
| nemotron35_bolt | free | raw | `tower` | 32 | all | 1190 | 17.89 ± 0.27 | 86.54 ± 0.30 | 86.54 | -7.71 | 0.0003 | complete |
| nemotron35_bolt | free | raw | `tower` | 32 | eligible | 1019 | 17.61 ± 0.28 | 87.64 ± 0.63 | 87.65 | -7.11 | 0.0003 | complete |
| nemotron35_bolt | free | raw | `merged` | 32 | all | 1190 | 18.70 ± 0.29 | 85.56 ± 1.35 | 85.57 | -7.54 | 3e-05 | complete |
| nemotron35_bolt | free | raw | `projected` | 32 | all | 1190 | 16.00 ± 0.50 | 84.68 ± 1.70 | 84.70 | -9.47 | 0.001 | complete |
| nemotron35_bolt | free | matched | `tower` | 32 | all | 1190 | 16.74 ± 0.23 | 83.21 ± 0.94 | 83.22 | -6.04 | 0.001 | complete |
| nemotron35_bolt | free | matched | `tower` | 32 | eligible | 1019 | 16.22 ± 0.24 | 84.16 ± 0.77 | 84.16 | -5.65 | 0.001 | complete |
| nemotron35_bolt | free | matched | `merged` | 32 | all | 1190 | 16.85 ± 0.24 | 86.51 ± 0.79 | 86.51 | -5.14 | 0.001 | complete |
| nemotron35_bolt | free | matched | `projected` | 32 | all | 1190 | 15.45 ± 0.44 | 85.36 ± 0.74 | 85.37 | -7.63 | 0.001 | complete |
| nemotron35_bolt | exemplar | raw | `tower` | 32 | eligible | 1019 | 17.94 ± 0.39 | 84.25 ± 0.81 | 84.25 | -2.18 | 0.0003 | complete |
| nemotron35_bolt | exemplar | matched | `tower` | 32 | eligible | 1019 | 16.99 ± 0.27 | 83.21 ± 0.69 | 83.21 | -3.80 | 0.001 | complete |
| nemotron35_super | free | raw | `tower` | 32 | all | 1190 | 18.24 ± 0.05 | 86.99 ± 0.77 | 86.99 | -7.60 | 0.0003 | complete |
| nemotron35_super | free | raw | `tower` | 32 | eligible | 1019 | 17.96 ± 0.11 | 87.89 ± 0.86 | 87.89 | -7.02 | 0.0003 | complete |
| nemotron35_super | free | raw | `merged` | 32 | all | 1190 | 18.96 ± 0.33 | 86.33 ± 1.41 | 86.34 | -7.67 | 0.0001 | complete |
| nemotron35_super | free | raw | `projected` | 32 | all | 1190 | 15.26 ± 0.37 | 84.24 ± 1.69 | 84.26 | -8.37 | 0.001 | complete |
| nemotron35_super | free | matched | `tower` | 32 | all | 1190 | 16.98 ± 0.24 | 84.13 ± 1.06 | 84.13 | -5.29 | 0.001 | complete |
| nemotron35_super | free | matched | `tower` | 32 | eligible | 1019 | 16.41 ± 0.27 | 84.39 ± 0.33 | 84.39 | -4.80 | 0.001 | complete |
| nemotron35_super | free | matched | `merged` | 32 | all | 1190 | 15.53 ± 0.57 | 83.93 ± 2.37 | 83.97 | -5.05 | 0.003 | complete |
| nemotron35_super | free | matched | `projected` | 32 | all | 1190 | 16.16 ± 0.22 | 85.07 ± 1.23 | 85.08 | -9.35 | 0.003 | complete |
| nemotron35_super | exemplar | raw | `tower` | 32 | eligible | 1019 | 18.52 ± 0.52 | 85.58 ± 0.76 | 85.58 | -3.01 | 0.0001 | complete |
| nemotron35_super | exemplar | matched | `tower` | 32 | eligible | 1019 | 16.84 ± 0.06 | 83.01 ± 0.93 | 83.01 | -4.05 | 0.001 | complete |
| nemotron_omni | free | raw | `tower` | 32 | all | 1190 | 19.86 ± 0.30 | 87.11 ± 0.20 | 87.11 | -6.98 | 0.001 | complete |
| nemotron_omni | free | raw | `tower` | 32 | eligible | 1019 | 19.29 ± 0.30 | 86.52 ± 0.09 | 86.52 | -6.14 | 0.001 | complete |
| nemotron_omni | free | raw | `merged` | 32 | all | 1190 | 19.44 ± 0.41 | 85.62 ± 0.84 | 85.63 | -7.27 | 0.001 | complete |
| nemotron_omni | free | raw | `projected` | 32 | all | 1190 | 18.45 ± 0.21 | 90.18 ± 0.36 | 90.19 | -7.70 | 0.0003 | complete |
| nemotron_omni | free | matched | `tower` | 32 | all | 1190 | 17.91 ± 0.18 | 89.37 ± 0.56 | 89.37 | -9.34 | 0.001 | complete |
| nemotron_omni | free | matched | `tower` | 32 | eligible | 1019 | 17.23 ± 0.21 | 87.22 ± 0.51 | 87.22 | -8.81 | 0.001 | complete |
| nemotron_omni | free | matched | `merged` | 32 | all | 1190 | 19.80 ± 0.41 | 89.25 ± 0.69 | 89.26 | -10.03 | 0.001 | complete |
| nemotron_omni | free | matched | `projected` | 32 | all | 1190 | 18.71 ± 0.25 | 88.61 ± 0.71 | 88.61 | -8.70 | 0.001 | complete |
| nemotron_omni | exemplar | raw | `tower` | 32 | eligible | 1019 | 19.90 ± 0.10 | 84.85 ± 0.43 | 84.86 | -4.66 | 0.001 | complete |
| nemotron_omni | exemplar | matched | `tower` | 32 | eligible | 1019 | 18.43 ± 0.22 | 86.26 ± 0.52 | 86.26 | -7.61 | 0.001 | complete |
| qwen3_5 | free | raw | `tower` | 27 | all | 1190 | 19.96 ± 0.59 | 89.32 ± 1.38 | 89.33 | -12.68 | 0.0001 | complete |
| qwen3_5 | free | raw | `tower` | 27 | eligible | 1019 | 19.23 ± 0.55 | 86.50 ± 0.90 | 86.50 | -12.14 | 0.0001 | complete |
| qwen3_5 | free | raw | `merged` | 27 | all | 1190 | 20.77 ± 0.09 | 92.67 ± 0.27 | 92.67 | -15.42 | 0.0001 | complete |
| qwen3_5 | free | raw | `projected` | 27 | all | 1190 | 17.70 ± 0.30 | 88.92 ± 0.65 | 88.92 | -10.23 | 0.001 | complete |
| qwen3_5 | free | matched | `tower` | 27 | all | 1190 | 20.32 ± 0.21 | 88.46 ± 0.25 | 88.46 | -12.95 | 0.001 | complete |
| qwen3_5 | free | matched | `tower` | 27 | eligible | 1019 | 19.56 ± 0.26 | 85.37 ± 0.45 | 85.37 | -12.58 | 0.001 | complete |
| qwen3_5 | free | matched | `merged` | 27 | all | 1190 | 21.49 ± 0.28 | 89.51 ± 0.14 | 89.51 | -15.84 | 0.0003 | complete |
| qwen3_5 | free | matched | `projected` | 27 | all | 1190 | 18.19 ± 0.82 | 89.12 ± 0.71 | 89.13 | -12.36 | 0.003 | complete |
| qwen3_5 | exemplar | raw | `tower` | 27 | eligible | 1019 | 19.10 ± 0.29 | 84.36 ± 0.84 | 84.37 | -9.59 | 0.0001 | complete |
| qwen3_5 | exemplar | matched | `tower` | 27 | eligible | 1019 | 18.85 ± 0.37 | 81.68 ± 0.56 | 81.68 | -10.78 | 0.001 | complete |
| siglip2 | free | raw | `tower` | 27 | all | 1190 | 21.99 ± 0.54 | 90.86 ± 0.66 | 90.86 | -10.27 | 0.001 | complete |
| siglip2 | free | raw | `tower` | 27 | eligible | 1019 | 21.21 ± 0.50 | 86.16 ± 0.76 | 86.16 | -10.11 | 0.001 | complete |
| siglip2 | free | matched | `tower` | 27 | all | 1190 | 25.24 ± 0.87 | 95.64 ± 0.74 | 95.64 | -13.72 | 0.001 | complete |
| siglip2 | free | matched | `tower` | 27 | eligible | 1019 | 24.68 ± 0.80 | 92.35 ± 0.80 | 92.36 | -14.09 | 0.001 | complete |
| siglip2 | exemplar | raw | `tower` | 27 | eligible | 1019 | 20.92 ± 0.44 | 85.11 ± 0.59 | 85.11 | -9.11 | 0.001 | complete |
| siglip2 | exemplar | matched | `tower` | 27 | eligible | 1019 | 24.06 ± 0.25 | 90.29 ± 0.70 | 90.30 | -12.37 | 0.001 | complete |

### Per count bin (true count; † marks n < 30)

| Model | Arm | Capacity | Stage | Test IDs | le50 n | le50 MAE | le50 RMSE | 50to200 n | 50to200 MAE | 50to200 RMSE | gt200 n | gt200 MAE | gt200 RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek_v41 | free | raw | `tower` | all | 836 | 4.49 | 7.20 | 315 | 16.67 | 25.37 | 39 | 196.06 | 458.74 |
| deepseek_v41 | free | raw | `tower` | eligible | 714 | 4.30 | 7.08 | 270 | 16.23 | 24.70 | 35 | 183.08 | 453.19 |
| deepseek_v41 | free | raw | `merged` | all | 836 | 5.05 | 8.03 | 315 | 16.80 | 24.80 | 39 | 190.56 | 449.13 |
| deepseek_v41 | free | raw | `projected` | all | 836 | 6.17 | 9.10 | 315 | 17.37 | 25.06 | 39 | 195.63 | 449.43 |
| deepseek_v41 | free | matched | `tower` | all | 836 | 5.05 | 7.84 | 315 | 16.98 | 25.47 | 39 | 198.90 | 465.13 |
| deepseek_v41 | free | matched | `tower` | eligible | 714 | 4.79 | 7.59 | 270 | 16.43 | 24.97 | 35 | 184.61 | 457.42 |
| deepseek_v41 | free | matched | `merged` | all | 836 | 5.52 | 8.49 | 315 | 17.04 | 25.76 | 39 | 196.46 | 459.33 |
| deepseek_v41 | free | matched | `projected` | all | 836 | 5.56 | 8.45 | 315 | 17.47 | 27.29 | 39 | 196.07 | 457.45 |
| deepseek_v41 | exemplar | raw | `tower` | eligible | 714 | 4.37 | 7.20 | 270 | 15.79 | 23.62 | 35 | 178.19 | 447.95 |
| deepseek_v41 | exemplar | matched | `tower` | eligible | 714 | 5.39 | 8.75 | 270 | 16.21 | 23.93 | 35 | 182.85 | 454.69 |
| dinov2 | free | raw | `tower` | all | 836 | 10.05 | 14.53 | 315 | 31.74 | 43.14 | 39 | 249.48 | 507.57 |
| dinov2 | free | raw | `tower` | eligible | 714 | 9.69 | 14.15 | 270 | 30.54 | 41.64 | 35 | 225.80 | 473.60 |
| dinov2 | free | matched | `tower` | all | 836 | 10.02 | 14.71 | 315 | 32.53 | 42.79 | 39 | 255.16 | 513.99 |
| dinov2 | free | matched | `tower` | eligible | 714 | 9.72 | 14.39 | 270 | 31.89 | 42.14 | 35 | 230.90 | 478.73 |
| dinov2 | exemplar | raw | `tower` | eligible | 714 | 9.64 | 13.74 | 270 | 32.39 | 42.71 | 35 | 227.74 | 467.94 |
| dinov2 | exemplar | matched | `tower` | eligible | 714 | 10.20 | 15.25 | 270 | 29.48 | 39.84 | 35 | 232.63 | 476.95 |
| dinov3 | free | raw | `tower` | all | 836 | 11.47 | 16.74 | 315 | 31.39 | 42.29 | 39 | 200.25 | 446.48 |
| dinov3 | free | raw | `tower` | eligible | 714 | 11.24 | 16.52 | 270 | 31.25 | 42.24 | 35 | 185.77 | 443.35 |
| dinov3 | free | matched | `tower` | all | 836 | 11.43 | 17.90 | 315 | 35.73 | 48.20 | 39 | 217.04 | 457.51 |
| dinov3 | free | matched | `tower` | eligible | 714 | 11.01 | 17.14 | 270 | 35.42 | 48.05 | 35 | 198.14 | 443.15 |
| dinov3 | exemplar | raw | `tower` | eligible | 714 | 14.81 | 21.35 | 270 | 31.82 | 43.07 | 35 | 172.75 | 435.23 |
| dinov3 | exemplar | matched | `tower` | eligible | 714 | 13.88 | 19.83 | 270 | 35.69 | 46.60 | 35 | 188.73 | 439.98 |
| gemma4 | free | raw | `tower` | all | 836 | 5.93 | 9.15 | 315 | 21.23 | 31.71 | 39 | 214.43 | 476.87 |
| gemma4 | free | raw | `tower` | eligible | 714 | 5.67 | 8.92 | 270 | 20.92 | 31.62 | 35 | 200.30 | 465.20 |
| gemma4 | free | raw | `merged` | all | 836 | 6.00 | 8.97 | 315 | 21.86 | 32.11 | 39 | 204.89 | 466.89 |
| gemma4 | free | raw | `projected` | all | 836 | 5.45 | 8.42 | 315 | 21.31 | 31.22 | 39 | 211.37 | 469.42 |
| gemma4 | free | matched | `tower` | all | 836 | 5.73 | 9.22 | 315 | 22.73 | 33.06 | 39 | 228.70 | 481.51 |
| gemma4 | free | matched | `tower` | eligible | 714 | 5.55 | 9.19 | 270 | 22.66 | 33.11 | 35 | 216.29 | 472.13 |
| gemma4 | free | matched | `merged` | all | 836 | 6.16 | 9.20 | 315 | 23.80 | 34.14 | 39 | 215.70 | 476.41 |
| gemma4 | free | matched | `projected` | all | 836 | 5.65 | 8.32 | 315 | 22.14 | 31.92 | 39 | 208.11 | 472.25 |
| gemma4 | exemplar | raw | `tower` | eligible | 714 | 6.04 | 9.89 | 270 | 21.52 | 32.16 | 35 | 200.59 | 459.28 |
| gemma4 | exemplar | matched | `tower` | eligible | 714 | 5.96 | 10.09 | 270 | 23.24 | 33.45 | 35 | 210.55 | 466.62 |
| glm5 | free | raw | `tower` | all | 836 | 8.29 | 12.00 | 315 | 26.03 | 34.08 | 39 | 214.83 | 471.57 |
| glm5 | free | raw | `tower` | eligible | 714 | 8.02 | 11.56 | 270 | 25.68 | 33.85 | 35 | 194.27 | 451.65 |
| glm5 | free | raw | `merged` | all | 836 | 9.73 | 13.09 | 315 | 25.77 | 35.67 | 39 | 255.43 | 503.68 |
| glm5 | free | raw | `projected` | all | 836 | 7.64 | 11.47 | 315 | 21.23 | 29.19 | 39 | 218.02 | 477.14 |
| glm5 | free | matched | `tower` | all | 836 | 7.30 | 10.35 | 315 | 26.60 | 36.41 | 39 | 219.77 | 472.57 |
| glm5 | free | matched | `tower` | eligible | 714 | 6.99 | 9.88 | 270 | 25.95 | 35.83 | 35 | 198.94 | 448.07 |
| glm5 | free | matched | `merged` | all | 836 | 6.95 | 9.91 | 315 | 22.57 | 30.65 | 39 | 214.04 | 468.51 |
| glm5 | free | matched | `projected` | all | 836 | 6.02 | 8.73 | 315 | 21.00 | 29.33 | 39 | 197.20 | 454.34 |
| glm5 | exemplar | raw | `tower` | eligible | 714 | 8.06 | 11.57 | 270 | 23.03 | 31.51 | 35 | 193.04 | 449.88 |
| glm5 | exemplar | matched | `tower` | eligible | 714 | 7.64 | 10.69 | 270 | 25.54 | 35.48 | 35 | 202.04 | 445.44 |
| kimi_k26 | free | raw | `tower` | all | 836 | 6.83 | 10.88 | 315 | 18.45 | 26.02 | 39 | 198.39 | 446.63 |
| kimi_k26 | free | raw | `tower` | eligible | 714 | 6.79 | 11.10 | 270 | 17.94 | 25.52 | 35 | 188.65 | 445.17 |
| kimi_k26 | free | raw | `merged` | all | 836 | 7.80 | 11.79 | 315 | 20.81 | 28.67 | 39 | 197.71 | 439.77 |
| kimi_k26 | free | raw | `projected` | all | 836 | 6.05 | 9.73 | 315 | 17.94 | 25.73 | 39 | 179.02 | 436.13 |
| kimi_k26 | free | matched | `tower` | all | 836 | 6.19 | 10.30 | 315 | 20.48 | 29.19 | 39 | 184.45 | 429.06 |
| kimi_k26 | free | matched | `tower` | eligible | 714 | 6.06 | 10.41 | 270 | 20.04 | 28.89 | 35 | 176.22 | 430.52 |
| kimi_k26 | free | matched | `merged` | all | 836 | 6.20 | 10.25 | 315 | 19.67 | 27.73 | 39 | 192.37 | 445.03 |
| kimi_k26 | free | matched | `projected` | all | 836 | 6.20 | 9.54 | 315 | 18.47 | 26.85 | 39 | 182.51 | 449.13 |
| kimi_k26 | exemplar | raw | `tower` | eligible | 714 | 8.52 | 13.08 | 270 | 17.76 | 24.62 | 35 | 181.90 | 435.75 |
| kimi_k26 | exemplar | matched | `tower` | eligible | 714 | 6.06 | 10.91 | 270 | 18.20 | 26.58 | 35 | 169.30 | 419.83 |
| mimo26_flash | free | raw | `tower` | all | 836 | 10.79 | 14.67 | 315 | 25.64 | 33.84 | 39 | 243.32 | 504.30 |
| mimo26_flash | free | raw | `tower` | eligible | 714 | 10.31 | 14.03 | 270 | 25.87 | 34.24 | 35 | 224.98 | 483.45 |
| mimo26_flash | free | raw | `merged` | all | 836 | 7.50 | 11.18 | 315 | 21.51 | 30.65 | 39 | 206.91 | 480.18 |
| mimo26_flash | free | raw | `projected` | all | 836 | 7.30 | 10.71 | 315 | 20.37 | 29.54 | 39 | 214.04 | 486.81 |
| mimo26_flash | free | matched | `tower` | all | 836 | 11.03 | 15.34 | 315 | 25.83 | 34.83 | 39 | 248.03 | 507.07 |
| mimo26_flash | free | matched | `tower` | eligible | 714 | 10.55 | 14.63 | 270 | 25.78 | 35.16 | 35 | 230.32 | 486.38 |
| mimo26_flash | free | matched | `merged` | all | 836 | 5.97 | 9.47 | 315 | 20.15 | 28.29 | 39 | 196.16 | 472.39 |
| mimo26_flash | free | matched | `projected` | all | 836 | 6.16 | 9.62 | 315 | 20.92 | 30.40 | 39 | 199.88 | 470.96 |
| mimo26_flash | exemplar | raw | `tower` | eligible | 714 | 11.29 | 15.51 | 270 | 24.70 | 33.02 | 35 | 222.64 | 481.38 |
| mimo26_flash | exemplar | matched | `tower` | eligible | 714 | 13.15 | 17.48 | 270 | 24.99 | 34.38 | 35 | 229.52 | 484.52 |
| mimo26_pro | free | raw | `tower` | all | 836 | 10.39 | 14.29 | 315 | 25.66 | 33.73 | 39 | 238.23 | 498.00 |
| mimo26_pro | free | raw | `tower` | eligible | 714 | 9.91 | 13.73 | 270 | 25.96 | 34.19 | 35 | 220.77 | 480.54 |
| mimo26_pro | free | raw | `merged` | all | 836 | 6.76 | 10.29 | 315 | 20.83 | 30.23 | 39 | 205.86 | 474.47 |
| mimo26_pro | free | raw | `projected` | all | 836 | 7.62 | 11.34 | 315 | 20.33 | 29.52 | 39 | 219.75 | 489.53 |
| mimo26_pro | free | matched | `tower` | all | 836 | 10.07 | 14.01 | 315 | 25.52 | 34.37 | 39 | 245.73 | 503.92 |
| mimo26_pro | free | matched | `tower` | eligible | 714 | 9.64 | 13.37 | 270 | 25.39 | 34.57 | 35 | 227.82 | 484.54 |
| mimo26_pro | free | matched | `merged` | all | 836 | 6.14 | 9.27 | 315 | 21.34 | 30.73 | 39 | 210.00 | 478.00 |
| mimo26_pro | free | matched | `projected` | all | 836 | 6.56 | 10.03 | 315 | 20.70 | 29.70 | 39 | 212.85 | 481.25 |
| mimo26_pro | exemplar | raw | `tower` | eligible | 714 | 11.16 | 15.57 | 270 | 24.31 | 32.67 | 35 | 217.91 | 478.09 |
| mimo26_pro | exemplar | matched | `tower` | eligible | 714 | 11.60 | 15.90 | 270 | 25.19 | 34.72 | 35 | 225.06 | 482.74 |
| minimax_m3 | free | raw | `tower` | all | 836 | 11.07 | 16.83 | 315 | 30.47 | 40.96 | 39 | 232.79 | 490.61 |
| minimax_m3 | free | raw | `tower` | eligible | 714 | 10.79 | 16.59 | 270 | 30.11 | 40.91 | 35 | 219.63 | 477.36 |
| minimax_m3 | free | raw | `merged` | all | 836 | 12.00 | 21.40 | 315 | 30.40 | 41.20 | 39 | 226.27 | 482.33 |
| minimax_m3 | free | raw | `projected` | all | 836 | 10.36 | 14.07 | 315 | 32.11 | 42.78 | 39 | 243.61 | 499.87 |
| minimax_m3 | free | matched | `tower` | all | 836 | 9.84 | 13.89 | 315 | 29.21 | 39.23 | 39 | 216.45 | 468.33 |
| minimax_m3 | free | matched | `tower` | eligible | 714 | 9.67 | 13.71 | 270 | 28.85 | 39.25 | 35 | 204.63 | 458.93 |
| minimax_m3 | free | matched | `merged` | all | 836 | 10.41 | 15.31 | 315 | 29.95 | 39.78 | 39 | 204.89 | 466.34 |
| minimax_m3 | free | matched | `projected` | all | 836 | 9.79 | 13.32 | 315 | 31.53 | 41.55 | 39 | 238.57 | 499.20 |
| minimax_m3 | exemplar | raw | `tower` | eligible | 714 | 11.98 | 18.14 | 270 | 28.60 | 39.35 | 35 | 217.62 | 468.60 |
| minimax_m3 | exemplar | matched | `tower` | eligible | 714 | 10.05 | 14.34 | 270 | 29.20 | 39.71 | 35 | 207.46 | 456.52 |
| moonvit_v2 | free | raw | `tower` | all | 836 | 5.84 | 9.62 | 315 | 16.65 | 24.18 | 39 | 170.94 | 450.52 |
| moonvit_v2 | free | raw | `tower` | eligible | 714 | 5.68 | 9.62 | 270 | 16.39 | 24.21 | 35 | 155.35 | 439.35 |
| moonvit_v2 | free | raw | `merged` | all | 836 | 6.02 | 9.64 | 315 | 16.33 | 23.58 | 39 | 174.22 | 445.08 |
| moonvit_v2 | free | raw | `projected` | all | 836 | 5.72 | 8.91 | 315 | 15.94 | 23.29 | 39 | 184.04 | 456.84 |
| moonvit_v2 | free | matched | `tower` | all | 836 | 5.41 | 8.46 | 315 | 17.39 | 25.31 | 39 | 168.57 | 449.51 |
| moonvit_v2 | free | matched | `tower` | eligible | 714 | 5.31 | 8.55 | 270 | 17.03 | 25.05 | 35 | 152.28 | 438.29 |
| moonvit_v2 | free | matched | `merged` | all | 836 | 5.38 | 8.43 | 315 | 16.25 | 24.44 | 39 | 168.64 | 449.42 |
| moonvit_v2 | free | matched | `projected` | all | 836 | 5.12 | 8.02 | 315 | 15.71 | 23.57 | 39 | 179.00 | 462.55 |
| moonvit_v2 | exemplar | raw | `tower` | eligible | 714 | 5.93 | 10.00 | 270 | 16.28 | 23.96 | 35 | 156.38 | 434.55 |
| moonvit_v2 | exemplar | matched | `tower` | eligible | 714 | 5.15 | 8.65 | 270 | 16.41 | 24.89 | 35 | 152.45 | 440.78 |
| muse_glimmer | free | raw | `tower` | all | 836 | 7.93 | 12.24 | 315 | 19.06 | 27.49 | 39 | 178.85 | 463.40 |
| muse_glimmer | free | raw | `tower` | eligible | 714 | 7.73 | 12.17 | 270 | 18.49 | 26.91 | 35 | 160.86 | 445.78 |
| muse_glimmer | free | raw | `merged` | all | 836 | 7.41 | 11.13 | 315 | 19.55 | 27.05 | 39 | 186.72 | 468.71 |
| muse_glimmer | free | raw | `projected` | all | 836 | 6.44 | 10.57 | 315 | 16.34 | 24.06 | 39 | 176.14 | 471.98 |
| muse_glimmer | free | matched | `tower` | all | 836 | 6.90 | 11.59 | 315 | 18.17 | 27.30 | 39 | 177.98 | 462.77 |
| muse_glimmer | free | matched | `tower` | eligible | 714 | 6.63 | 11.21 | 270 | 17.54 | 26.80 | 35 | 158.49 | 440.70 |
| muse_glimmer | free | matched | `merged` | all | 836 | 7.32 | 12.08 | 315 | 18.47 | 26.89 | 39 | 188.15 | 471.12 |
| muse_glimmer | free | matched | `projected` | all | 836 | 6.48 | 10.66 | 315 | 16.07 | 23.77 | 39 | 184.07 | 470.05 |
| muse_glimmer | exemplar | raw | `tower` | eligible | 714 | 7.70 | 12.19 | 270 | 17.81 | 26.04 | 35 | 167.81 | 444.87 |
| muse_glimmer | exemplar | matched | `tower` | eligible | 714 | 7.03 | 11.81 | 270 | 17.16 | 26.27 | 35 | 158.74 | 435.97 |
| nemotron35_bolt | free | raw | `tower` | all | 836 | 7.54 | 11.10 | 315 | 21.44 | 31.25 | 39 | 211.21 | 466.90 |
| nemotron35_bolt | free | raw | `tower` | eligible | 714 | 7.35 | 10.95 | 270 | 21.51 | 31.29 | 35 | 196.82 | 462.19 |
| nemotron35_bolt | free | raw | `merged` | all | 836 | 8.71 | 12.38 | 315 | 21.92 | 31.58 | 39 | 206.96 | 460.45 |
| nemotron35_bolt | free | raw | `projected` | all | 836 | 5.86 | 9.05 | 315 | 19.99 | 28.75 | 39 | 201.11 | 458.66 |
| nemotron35_bolt | free | matched | `tower` | all | 836 | 7.34 | 11.96 | 315 | 21.29 | 31.09 | 39 | 181.70 | 447.64 |
| nemotron35_bolt | free | matched | `tower` | eligible | 714 | 7.13 | 11.77 | 270 | 20.75 | 30.62 | 35 | 166.70 | 442.88 |
| nemotron35_bolt | free | matched | `merged` | all | 836 | 7.46 | 11.60 | 315 | 20.26 | 28.40 | 39 | 190.68 | 467.93 |
| nemotron35_bolt | free | matched | `projected` | all | 836 | 5.89 | 9.03 | 315 | 18.31 | 26.69 | 39 | 197.27 | 463.51 |
| nemotron35_bolt | exemplar | raw | `tower` | eligible | 714 | 9.05 | 13.47 | 270 | 20.87 | 29.40 | 35 | 176.63 | 443.01 |
| nemotron35_bolt | exemplar | matched | `tower` | eligible | 714 | 7.85 | 13.08 | 270 | 21.95 | 32.21 | 35 | 165.14 | 435.97 |
| nemotron35_super | free | raw | `tower` | all | 836 | 7.72 | 11.32 | 315 | 21.94 | 31.62 | 39 | 213.82 | 469.10 |
| nemotron35_super | free | raw | `tower` | eligible | 714 | 7.54 | 11.17 | 270 | 22.03 | 31.76 | 35 | 199.16 | 463.19 |
| nemotron35_super | free | raw | `merged` | all | 836 | 8.36 | 12.13 | 315 | 22.70 | 32.57 | 39 | 215.85 | 464.41 |
| nemotron35_super | free | raw | `projected` | all | 836 | 5.82 | 9.16 | 315 | 17.84 | 26.27 | 39 | 196.85 | 457.32 |
| nemotron35_super | free | matched | `tower` | all | 836 | 7.61 | 12.25 | 315 | 21.28 | 31.16 | 39 | 183.11 | 452.64 |
| nemotron35_super | free | matched | `tower` | eligible | 714 | 7.42 | 12.08 | 270 | 20.74 | 30.75 | 35 | 166.38 | 443.92 |
| nemotron35_super | free | matched | `merged` | all | 836 | 6.22 | 9.63 | 315 | 19.56 | 27.57 | 39 | 182.36 | 454.79 |
| nemotron35_super | free | matched | `projected` | all | 836 | 5.89 | 9.07 | 315 | 20.34 | 29.69 | 39 | 202.48 | 460.35 |
| nemotron35_super | exemplar | raw | `tower` | eligible | 714 | 9.15 | 13.72 | 270 | 21.74 | 30.97 | 35 | 185.06 | 449.41 |
| nemotron35_super | exemplar | matched | `tower` | eligible | 714 | 7.82 | 12.98 | 270 | 21.45 | 31.45 | 35 | 165.24 | 435.36 |
| nemotron_omni | free | raw | `tower` | all | 836 | 8.84 | 13.25 | 315 | 26.20 | 35.94 | 39 | 204.82 | 466.17 |
| nemotron_omni | free | raw | `tower` | eligible | 714 | 8.55 | 13.00 | 270 | 25.84 | 35.49 | 35 | 187.78 | 452.49 |
| nemotron_omni | free | raw | `merged` | all | 836 | 8.70 | 12.93 | 315 | 25.12 | 35.01 | 39 | 203.64 | 458.47 |
| nemotron_omni | free | raw | `projected` | all | 836 | 6.90 | 10.28 | 315 | 23.10 | 32.60 | 39 | 228.70 | 487.15 |
| nemotron_omni | free | matched | `tower` | all | 836 | 6.57 | 10.03 | 315 | 23.09 | 33.74 | 39 | 219.05 | 482.03 |
| nemotron_omni | free | matched | `tower` | eligible | 714 | 6.21 | 9.57 | 270 | 22.80 | 33.52 | 35 | 199.26 | 459.27 |
| nemotron_omni | free | matched | `merged` | all | 836 | 8.15 | 12.53 | 315 | 24.88 | 35.22 | 39 | 228.50 | 479.25 |
| nemotron_omni | free | matched | `projected` | all | 836 | 6.83 | 10.34 | 315 | 24.69 | 35.02 | 39 | 225.15 | 476.81 |
| nemotron_omni | exemplar | raw | `tower` | eligible | 714 | 9.28 | 14.46 | 270 | 26.85 | 36.74 | 35 | 182.91 | 441.52 |
| nemotron_omni | exemplar | matched | `tower` | eligible | 714 | 7.23 | 11.32 | 270 | 24.88 | 35.71 | 35 | 196.95 | 451.84 |
| qwen3_5 | free | raw | `tower` | all | 836 | 8.25 | 11.31 | 315 | 27.09 | 36.78 | 39 | 213.41 | 479.32 |
| qwen3_5 | free | raw | `tower` | eligible | 714 | 7.88 | 10.82 | 270 | 26.83 | 36.54 | 35 | 192.05 | 452.91 |
| qwen3_5 | free | raw | `merged` | all | 836 | 8.12 | 10.82 | 315 | 28.94 | 38.32 | 39 | 225.86 | 497.65 |
| qwen3_5 | free | raw | `projected` | all | 836 | 6.70 | 10.02 | 315 | 23.72 | 32.55 | 39 | 204.84 | 480.11 |
| qwen3_5 | free | matched | `tower` | all | 836 | 7.92 | 11.57 | 315 | 28.38 | 37.72 | 39 | 220.93 | 473.71 |
| qwen3_5 | free | matched | `tower` | eligible | 714 | 7.62 | 11.23 | 270 | 27.89 | 37.42 | 35 | 199.06 | 445.86 |
| qwen3_5 | free | matched | `merged` | all | 836 | 8.20 | 11.50 | 315 | 30.81 | 40.13 | 39 | 231.22 | 478.13 |
| qwen3_5 | free | matched | `projected` | all | 836 | 6.82 | 9.61 | 315 | 24.28 | 33.83 | 39 | 212.88 | 480.75 |
| qwen3_5 | exemplar | raw | `tower` | eligible | 714 | 8.37 | 11.49 | 270 | 25.64 | 34.75 | 35 | 187.61 | 441.78 |
| qwen3_5 | exemplar | matched | `tower` | eligible | 714 | 7.63 | 11.79 | 270 | 26.20 | 35.82 | 35 | 191.09 | 426.02 |
| siglip2 | free | raw | `tower` | all | 836 | 9.48 | 14.18 | 315 | 27.70 | 36.33 | 39 | 244.07 | 486.72 |
| siglip2 | free | raw | `tower` | eligible | 714 | 9.16 | 13.72 | 270 | 27.23 | 35.63 | 35 | 220.72 | 450.00 |
| siglip2 | free | matched | `tower` | all | 836 | 9.96 | 15.22 | 315 | 34.88 | 44.04 | 39 | 274.91 | 508.36 |
| siglip2 | free | matched | `tower` | eligible | 714 | 9.66 | 14.82 | 270 | 34.87 | 43.88 | 35 | 252.29 | 478.48 |
| siglip2 | exemplar | raw | `tower` | eligible | 714 | 9.46 | 13.84 | 270 | 26.14 | 34.83 | 35 | 214.62 | 444.54 |
| siglip2 | exemplar | matched | `tower` | eligible | 714 | 9.73 | 15.46 | 270 | 33.24 | 42.70 | 35 | 245.50 | 467.35 |

### Diagnostic: one-exemplar FSC-147 validation

Transfer diagnostic only, never used for selection: the three-box exemplar head read with release box 0 alone on FSC-147 validation (`one_exemplar_val`), beside the same heads' three-box validation error from the cell's `.val.tsv`.

| Model | Capacity | Stage | val n | 3-box val MAE | 1-box val MAE ± sd | 3-box val RMSE | 1-box val RMSE |
|---|---|---|---|---|---|---|---|
| deepseek_v41 | raw | `tower` | 1068 | 20.86 | 20.87 ± 0.14 | 77.64 | 77.76 |
| deepseek_v41 | matched | `tower` | 1068 | 21.63 | 21.65 ± 0.24 | 78.12 | 78.26 |
| dinov2 | raw | `tower` | 1068 | 30.45 | 30.41 ± 0.32 | 83.97 | 84.18 |
| dinov2 | matched | `tower` | 1068 | 30.17 | 30.18 ± 0.52 | 83.26 | 83.51 |
| dinov3 | raw | `tower` | 1068 | 31.86 | 31.66 ± 0.67 | 79.59 | 79.84 |
| dinov3 | matched | `tower` | 1068 | 32.22 | 32.33 ± 0.44 | 81.82 | 81.97 |
| gemma4 | raw | `tower` | 1068 | 24.09 | 24.09 ± 0.07 | 82.06 | 82.07 |
| gemma4 | matched | `tower` | 1068 | 24.38 | 24.38 ± 0.02 | 85.27 | 85.29 |
| glm5 | raw | `tower` | 1068 | 24.41 | 24.42 ± 0.56 | 78.34 | 78.35 |
| glm5 | matched | `tower` | 1068 | 25.92 | 25.95 ± 0.29 | 81.52 | 81.55 |
| kimi_k26 | raw | `tower` | 1068 | 23.23 | 23.21 ± 0.47 | 75.55 | 74.37 |
| kimi_k26 | matched | `tower` | 1068 | 22.97 | 22.79 ± 0.34 | 75.00 | 73.09 |
| mimo26_flash | raw | `tower` | 1068 | 27.17 | 27.17 ± 0.08 | 87.41 | 87.41 |
| mimo26_flash | matched | `tower` | 1068 | 28.24 | 28.24 ± 0.96 | 88.20 | 88.21 |
| mimo26_pro | raw | `tower` | 1068 | 26.93 | 26.93 ± 0.08 | 86.44 | 86.44 |
| mimo26_pro | matched | `tower` | 1068 | 27.66 | 27.66 ± 0.55 | 88.01 | 88.01 |
| minimax_m3 | raw | `tower` | 1068 | 28.37 | 28.36 ± 0.25 | 79.16 | 79.15 |
| minimax_m3 | matched | `tower` | 1068 | 27.39 | 27.47 ± 0.30 | 80.36 | 80.61 |
| moonvit_v2 | raw | `tower` | 1068 | 20.13 | 19.93 ± 0.38 | 73.68 | 72.30 |
| moonvit_v2 | matched | `tower` | 1068 | 19.50 | 19.42 ± 0.09 | 72.43 | 71.48 |
| muse_glimmer | raw | `tower` | 1068 | 21.14 | 20.98 ± 0.82 | 77.62 | 77.05 |
| muse_glimmer | matched | `tower` | 1068 | 20.77 | 20.54 ± 0.28 | 78.08 | 77.55 |
| nemotron35_bolt | raw | `tower` | 1068 | 21.66 | 21.66 ± 0.53 | 70.83 | 70.84 |
| nemotron35_bolt | matched | `tower` | 1068 | 22.13 | 22.13 ± 0.67 | 74.39 | 74.40 |
| nemotron35_super | raw | `tower` | 1068 | 22.36 | 22.36 ± 0.22 | 73.54 | 73.55 |
| nemotron35_super | matched | `tower` | 1068 | 22.18 | 22.18 ± 0.56 | 75.55 | 75.56 |
| nemotron_omni | raw | `tower` | 1068 | 22.01 | 22.01 ± 0.31 | 73.83 | 73.85 |
| nemotron_omni | matched | `tower` | 1068 | 23.78 | 23.77 ± 0.28 | 76.37 | 76.41 |
| qwen3_5 | raw | `tower` | 1068 | 25.45 | 25.44 ± 0.65 | 78.18 | 78.19 |
| qwen3_5 | matched | `tower` | 1068 | 25.37 | 25.36 ± 0.31 | 78.82 | 78.83 |
| siglip2 | raw | `tower` | 1068 | 26.53 | 26.66 ± 0.23 | 82.58 | 82.75 |
| siglip2 | matched | `tower` | 1068 | 26.14 | 26.37 ± 0.36 | 84.65 | 84.97 |

## CountQA-derived transfer (experimental, supplementary; frozen FSC-147 heads)

Experimental, supplementary: exemplar-based frozen-feature counting transfer on a curated CountQA-derived subset. These are not official CountQA benchmark scores, not MLLM question answering, and not comparable to published CountQA MLLM results. Results with AI-authored exemplars (not human-reviewed) carry that provenance in their JSON. Exact match: floor(pred + 0.5) == answer.

| Model | Arm | Capacity | Stage | L | Test IDs | n | MAE ± sd | RMSE ± sd | pooled RMSE | bias | exact | LR | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek_v41 | free | raw | `tower` | 32 | all | 439 | 5.66 ± 0.12 | 12.33 ± 0.32 | 12.34 | 2.27 | 0.101 | n/a | complete |
| deepseek_v41 | free | raw | `merged` | 32 | all | 439 | 6.02 ± 0.39 | 13.09 ± 3.50 | 13.55 | 1.76 | 0.072 | n/a | complete |
| deepseek_v41 | free | raw | `projected` | 32 | all | 439 | 7.37 ± 0.52 | 12.62 ± 1.27 | 12.69 | 4.22 | 0.076 | n/a | complete |
| deepseek_v41 | free | matched | `tower` | 32 | all | 439 | 5.73 ± 0.09 | 11.00 ± 0.43 | 11.01 | 2.33 | 0.103 | n/a | complete |
| deepseek_v41 | free | matched | `merged` | 32 | all | 439 | 6.01 ± 0.03 | 13.45 ± 0.14 | 13.45 | 2.29 | 0.077 | n/a | complete |
| deepseek_v41 | free | matched | `projected` | 32 | all | 439 | 6.82 ± 0.16 | 16.03 ± 1.23 | 16.08 | 2.86 | 0.077 | n/a | complete |
| dinov2 | free | raw | `tower` | 24 | all | 439 | 7.18 ± 0.15 | 21.04 ± 0.15 | 21.04 | -1.95 | 0.064 | n/a | complete |
| dinov2 | free | matched | `tower` | 24 | all | 439 | 7.52 ± 0.05 | 21.29 ± 0.05 | 21.29 | -2.03 | 0.055 | n/a | complete |
| dinov3 | free | raw | `tower` | 24 | all | 439 | 8.43 ± 0.52 | 22.32 ± 0.91 | 22.34 | -1.20 | 0.055 | n/a | complete |
| dinov3 | free | matched | `tower` | 24 | all | 439 | 7.84 ± 0.24 | 21.58 ± 0.12 | 21.58 | -3.78 | 0.051 | n/a | complete |
| gemma4 | free | raw | `tower` | 27 | all | 439 | 8.14 ± 0.11 | 22.45 ± 0.16 | 22.46 | 1.63 | 0.066 | n/a | complete |
| gemma4 | free | raw | `merged` | 27 | all | 439 | 9.16 ± 0.30 | 23.08 ± 0.35 | 23.08 | 2.37 | 0.068 | n/a | complete |
| gemma4 | free | raw | `projected` | 27 | all | 439 | 8.43 ± 0.12 | 22.29 ± 0.48 | 22.29 | 2.17 | 0.070 | n/a | complete |
| gemma4 | free | matched | `tower` | 27 | all | 439 | 7.85 ± 0.08 | 21.69 ± 0.19 | 21.69 | 0.83 | 0.056 | n/a | complete |
| gemma4 | free | matched | `merged` | 27 | all | 439 | 9.64 ± 0.22 | 23.17 ± 0.04 | 23.17 | 2.91 | 0.053 | n/a | complete |
| gemma4 | free | matched | `projected` | 27 | all | 439 | 9.05 ± 0.29 | 23.14 ± 0.23 | 23.14 | 2.05 | 0.072 | n/a | complete |
| glm5 | free | raw | `tower` | 24 | all | 439 | 7.05 ± 0.44 | 18.59 ± 0.30 | 18.59 | 0.52 | 0.068 | n/a | complete |
| glm5 | free | raw | `merged` | 24 | all | 439 | 7.70 ± 0.19 | 20.81 ± 0.14 | 20.81 | 1.77 | 0.056 | n/a | complete |
| glm5 | free | raw | `projected` | 24 | all | 439 | 6.70 ± 0.41 | 14.60 ± 0.35 | 14.61 | 0.48 | 0.068 | n/a | complete |
| glm5 | free | matched | `tower` | 24 | all | 439 | 6.52 ± 0.10 | 18.72 ± 0.15 | 18.72 | -0.39 | 0.062 | n/a | complete |
| glm5 | free | matched | `merged` | 24 | all | 439 | 6.69 ± 0.10 | 19.36 ± 0.24 | 19.36 | -1.65 | 0.060 | n/a | complete |
| glm5 | free | matched | `projected` | 24 | all | 439 | 5.94 ± 0.03 | 15.91 ± 0.66 | 15.92 | -0.89 | 0.078 | n/a | complete |
| kimi_k26 | free | raw | `tower` | 27 | all | 439 | 6.30 ± 0.23 | 17.05 ± 0.07 | 17.05 | 1.98 | 0.102 | n/a | complete |
| kimi_k26 | free | raw | `merged` | 27 | all | 439 | 7.54 ± 0.34 | 16.47 ± 0.34 | 16.48 | 3.61 | 0.071 | n/a | complete |
| kimi_k26 | free | raw | `projected` | 27 | all | 439 | 6.42 ± 0.17 | 14.47 ± 0.89 | 14.50 | 1.89 | 0.093 | n/a | complete |
| kimi_k26 | free | matched | `tower` | 27 | all | 439 | 5.97 ± 0.03 | 15.35 ± 0.53 | 15.36 | 0.67 | 0.095 | n/a | complete |
| kimi_k26 | free | matched | `merged` | 27 | all | 439 | 6.10 ± 0.21 | 14.24 ± 0.67 | 14.25 | 0.98 | 0.066 | n/a | complete |
| kimi_k26 | free | matched | `projected` | 27 | all | 439 | 6.02 ± 0.18 | 14.62 ± 0.67 | 14.64 | -0.23 | 0.076 | n/a | complete |
| mimo26_flash | free | raw | `tower` | 28 | all | 439 | 11.81 ± 0.80 | 23.71 ± 0.65 | 23.72 | 8.23 | 0.040 | n/a | complete |
| mimo26_flash | free | raw | `merged` | 28 | all | 439 | 10.44 ± 0.49 | 23.31 ± 0.76 | 23.32 | 6.63 | 0.052 | n/a | complete |
| mimo26_flash | free | raw | `projected` | 28 | all | 439 | 9.74 ± 0.56 | 21.46 ± 0.78 | 21.47 | 6.63 | 0.056 | n/a | complete |
| mimo26_flash | free | matched | `tower` | 28 | all | 439 | 10.76 ± 0.45 | 23.44 ± 0.46 | 23.44 | 7.41 | 0.041 | n/a | complete |
| mimo26_flash | free | matched | `merged` | 28 | all | 439 | 7.42 ± 0.14 | 21.56 ± 0.20 | 21.57 | 2.70 | 0.097 | n/a | complete |
| mimo26_flash | free | matched | `projected` | 28 | all | 439 | 7.28 ± 0.14 | 20.62 ± 0.68 | 20.63 | 3.58 | 0.088 | n/a | complete |
| mimo26_pro | free | raw | `tower` | 28 | all | 439 | 11.74 ± 0.28 | 23.59 ± 0.24 | 23.59 | 8.02 | 0.046 | n/a | complete |
| mimo26_pro | free | raw | `merged` | 28 | all | 439 | 9.72 ± 0.48 | 21.84 ± 0.67 | 21.85 | 6.04 | 0.054 | n/a | complete |
| mimo26_pro | free | raw | `projected` | 28 | all | 439 | 9.84 ± 0.23 | 20.87 ± 0.54 | 20.88 | 6.68 | 0.058 | n/a | complete |
| mimo26_pro | free | matched | `tower` | 28 | all | 439 | 10.94 ± 0.27 | 23.57 ± 0.25 | 23.57 | 7.31 | 0.055 | n/a | complete |
| mimo26_pro | free | matched | `merged` | 28 | all | 439 | 8.16 ± 0.26 | 22.49 ± 0.51 | 22.49 | 3.89 | 0.066 | n/a | complete |
| mimo26_pro | free | matched | `projected` | 28 | all | 439 | 8.10 ± 0.60 | 20.20 ± 0.43 | 20.20 | 3.93 | 0.085 | n/a | complete |
| minimax_m3 | free | raw | `tower` | 32 | all | 439 | 19.27 ± 0.53 | 42.18 ± 0.78 | 42.19 | 17.25 | 0.017 | n/a | complete |
| minimax_m3 | free | raw | `merged` | 32 | all | 439 | 19.58 ± 1.73 | 39.71 ± 3.27 | 39.84 | 17.14 | 0.023 | n/a | complete |
| minimax_m3 | free | raw | `projected` | 32 | all | 439 | 11.04 ± 0.20 | 28.27 ± 0.74 | 28.28 | 6.33 | 0.039 | n/a | complete |
| minimax_m3 | free | matched | `tower` | 32 | all | 439 | 23.92 ± 0.91 | 51.73 ± 0.77 | 51.74 | 21.65 | 0.022 | n/a | complete |
| minimax_m3 | free | matched | `merged` | 32 | all | 439 | 26.46 ± 1.39 | 54.12 ± 2.39 | 54.18 | 24.33 | 0.015 | n/a | complete |
| minimax_m3 | free | matched | `projected` | 32 | all | 439 | 9.83 ± 0.25 | 26.02 ± 0.85 | 26.04 | 3.99 | 0.058 | n/a | complete |
| moonvit_v2 | free | raw | `tower` | 27 | all | 439 | 7.24 ± 0.12 | 19.31 ± 0.19 | 19.31 | 2.38 | 0.085 | n/a | complete |
| moonvit_v2 | free | raw | `merged` | 27 | all | 439 | 8.46 ± 0.59 | 21.07 ± 1.38 | 21.12 | 4.44 | 0.077 | n/a | complete |
| moonvit_v2 | free | raw | `projected` | 27 | all | 439 | 9.07 ± 1.30 | 22.31 ± 1.88 | 22.39 | 5.30 | 0.068 | n/a | complete |
| moonvit_v2 | free | matched | `tower` | 27 | all | 439 | 7.56 ± 0.20 | 18.60 ± 0.99 | 18.62 | 4.36 | 0.080 | n/a | complete |
| moonvit_v2 | free | matched | `merged` | 27 | all | 439 | 7.12 ± 0.44 | 20.15 ± 0.44 | 20.15 | 3.12 | 0.084 | n/a | complete |
| moonvit_v2 | free | matched | `projected` | 27 | all | 439 | 7.27 ± 0.32 | 19.03 ± 1.10 | 19.07 | 2.86 | 0.077 | n/a | complete |
| muse_glimmer | free | raw | `tower` | 50 | all | 439 | 7.26 ± 0.36 | 20.61 ± 0.38 | 20.62 | 0.64 | 0.085 | n/a | complete |
| muse_glimmer | free | raw | `merged` | 50 | all | 439 | 7.67 ± 0.19 | 21.69 ± 0.57 | 21.70 | 0.66 | 0.090 | n/a | complete |
| muse_glimmer | free | raw | `projected` | 50 | all | 439 | 5.87 ± 0.12 | 18.12 ± 0.22 | 18.12 | -0.18 | 0.093 | n/a | complete |
| muse_glimmer | free | matched | `tower` | 50 | all | 439 | 6.51 ± 0.23 | 19.12 ± 0.20 | 19.12 | 0.46 | 0.082 | n/a | complete |
| muse_glimmer | free | matched | `merged` | 50 | all | 439 | 6.52 ± 0.12 | 19.71 ± 0.14 | 19.71 | -0.25 | 0.080 | n/a | complete |
| muse_glimmer | free | matched | `projected` | 50 | all | 439 | 5.54 ± 0.07 | 18.18 ± 0.75 | 18.19 | -1.81 | 0.094 | n/a | complete |
| nemotron35_bolt | free | raw | `tower` | 32 | all | 439 | 6.09 ± 0.09 | 16.60 ± 0.14 | 16.60 | -1.51 | 0.071 | n/a | complete |
| nemotron35_bolt | free | raw | `merged` | 32 | all | 439 | 7.07 ± 0.02 | 17.30 ± 0.38 | 17.31 | -2.45 | 0.048 | n/a | complete |
| nemotron35_bolt | free | raw | `projected` | 32 | all | 439 | 7.06 ± 0.18 | 17.61 ± 0.34 | 17.62 | 1.21 | 0.078 | n/a | complete |
| nemotron35_bolt | free | matched | `tower` | 32 | all | 439 | 6.11 ± 0.11 | 16.61 ± 0.05 | 16.61 | -2.49 | 0.070 | n/a | complete |
| nemotron35_bolt | free | matched | `merged` | 32 | all | 439 | 6.36 ± 0.12 | 15.74 ± 0.36 | 15.75 | -1.54 | 0.068 | n/a | complete |
| nemotron35_bolt | free | matched | `projected` | 32 | all | 439 | 6.07 ± 0.25 | 18.08 ± 0.39 | 18.08 | 0.13 | 0.074 | n/a | complete |
| nemotron35_super | free | raw | `tower` | 32 | all | 439 | 6.23 ± 0.24 | 16.59 ± 0.45 | 16.60 | -2.51 | 0.068 | n/a | complete |
| nemotron35_super | free | raw | `merged` | 32 | all | 439 | 6.62 ± 0.13 | 17.01 ± 0.08 | 17.01 | -2.21 | 0.055 | n/a | complete |
| nemotron35_super | free | raw | `projected` | 32 | all | 439 | 7.44 ± 0.27 | 17.50 ± 0.15 | 17.50 | 2.41 | 0.073 | n/a | complete |
| nemotron35_super | free | matched | `tower` | 32 | all | 439 | 6.02 ± 0.13 | 16.70 ± 0.13 | 16.71 | -2.27 | 0.073 | n/a | complete |
| nemotron35_super | free | matched | `merged` | 32 | all | 439 | 6.23 ± 0.13 | 14.36 ± 0.98 | 14.39 | -2.80 | 0.054 | n/a | complete |
| nemotron35_super | free | matched | `projected` | 32 | all | 439 | 7.24 ± 0.24 | 18.53 ± 0.43 | 18.54 | 2.62 | 0.077 | n/a | complete |
| nemotron_omni | free | raw | `tower` | 32 | all | 439 | 7.71 ± 0.76 | 14.96 ± 0.81 | 14.99 | 3.36 | 0.061 | n/a | complete |
| nemotron_omni | free | raw | `merged` | 32 | all | 439 | 9.26 ± 1.37 | 16.85 ± 1.36 | 16.91 | 5.02 | 0.052 | n/a | complete |
| nemotron_omni | free | raw | `projected` | 32 | all | 439 | 6.70 ± 0.28 | 14.99 ± 0.42 | 15.00 | 2.64 | 0.083 | n/a | complete |
| nemotron_omni | free | matched | `tower` | 32 | all | 439 | 6.50 ± 0.07 | 16.31 ± 0.46 | 16.32 | -0.32 | 0.068 | n/a | complete |
| nemotron_omni | free | matched | `merged` | 32 | all | 439 | 6.99 ± 0.39 | 16.82 ± 0.43 | 16.82 | 1.11 | 0.073 | n/a | complete |
| nemotron_omni | free | matched | `projected` | 32 | all | 439 | 6.66 ± 0.31 | 17.30 ± 0.25 | 17.30 | 1.28 | 0.080 | n/a | complete |
| qwen3_5 | free | raw | `tower` | 27 | all | 439 | 9.52 ± 0.51 | 21.67 ± 0.28 | 21.67 | 5.12 | 0.046 | n/a | complete |
| qwen3_5 | free | raw | `merged` | 27 | all | 439 | 10.40 ± 0.27 | 23.13 ± 0.26 | 23.13 | 4.80 | 0.052 | n/a | complete |
| qwen3_5 | free | raw | `projected` | 27 | all | 439 | 6.94 ± 0.26 | 18.62 ± 0.37 | 18.63 | 0.58 | 0.062 | n/a | complete |
| qwen3_5 | free | matched | `tower` | 27 | all | 439 | 6.66 ± 0.12 | 15.72 ± 0.35 | 15.72 | 0.16 | 0.073 | n/a | complete |
| qwen3_5 | free | matched | `merged` | 27 | all | 439 | 8.38 ± 0.11 | 19.44 ± 0.01 | 19.44 | 1.83 | 0.051 | n/a | complete |
| qwen3_5 | free | matched | `projected` | 27 | all | 439 | 6.34 ± 0.22 | 17.83 ± 0.75 | 17.84 | -0.20 | 0.078 | n/a | complete |
| siglip2 | free | raw | `tower` | 27 | all | 439 | 7.99 ± 0.10 | 18.27 ± 0.55 | 18.27 | 0.51 | 0.067 | n/a | complete |
| siglip2 | free | matched | `tower` | 27 | all | 439 | 7.90 ± 0.15 | 19.00 ± 0.17 | 19.00 | 1.30 | 0.050 | n/a | complete |

### Per count bin (true answer; † marks n < 30)

| Model | Arm | Capacity | Stage | Test IDs | le50 n | le50 MAE | le50 RMSE | 50to200 n | 50to200 MAE | 50to200 RMSE | gt200 n | gt200 MAE | gt200 RMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek_v41 | free | raw | `tower` | all | 435 | 5.04 | 8.57 | 3† | 43.69 | 54.01 | 1† | 160.92 | 160.92 |
| deepseek_v41 | free | raw | `merged` | all | 435 | 5.37 | 8.48 | 3† | 46.01 | 56.60 | 1† | 171.03 | 171.03 |
| deepseek_v41 | free | raw | `projected` | all | 435 | 6.99 | 11.30 | 3† | 37.48 | 45.22 | 1† | 84.31 | 84.31 |
| deepseek_v41 | free | matched | `tower` | all | 435 | 5.25 | 8.43 | 3† | 34.17 | 42.54 | 1† | 129.44 | 129.44 |
| deepseek_v41 | free | matched | `merged` | all | 435 | 5.30 | 8.24 | 3† | 42.56 | 51.46 | 1† | 204.60 | 204.60 |
| deepseek_v41 | free | matched | `projected` | all | 435 | 6.04 | 9.61 | 3† | 37.19 | 47.82 | 1† | 255.91 | 255.91 |
| dinov2 | free | raw | `tower` | all | 435 | 5.97 | 8.28 | 3† | 55.11 | 67.82 | 1† | 388.14 | 388.14 |
| dinov2 | free | matched | `tower` | all | 435 | 6.33 | 8.68 | 3† | 52.35 | 62.19 | 1† | 393.21 | 393.21 |
| dinov3 | free | raw | `tower` | all | 435 | 7.25 | 12.47 | 3† | 58.65 | 67.79 | 1† | 369.78 | 369.78 |
| dinov3 | free | matched | `tower` | all | 435 | 6.62 | 9.67 | 3† | 59.63 | 68.87 | 1† | 386.55 | 386.55 |
| gemma4 | free | raw | `tower` | all | 435 | 7.05 | 12.22 | 3† | 38.98 | 49.75 | 1† | 385.97 | 385.97 |
| gemma4 | free | raw | `merged` | all | 435 | 8.04 | 13.81 | 3† | 48.71 | 56.65 | 1† | 375.68 | 375.68 |
| gemma4 | free | raw | `projected` | all | 435 | 7.31 | 12.86 | 3† | 50.72 | 59.45 | 1† | 367.91 | 367.91 |
| gemma4 | free | matched | `tower` | all | 435 | 6.79 | 11.36 | 3† | 37.11 | 47.36 | 1† | 378.85 | 378.85 |
| gemma4 | free | matched | `merged` | all | 435 | 8.54 | 14.35 | 3† | 49.75 | 59.14 | 1† | 368.00 | 368.00 |
| gemma4 | free | matched | `projected` | all | 435 | 7.92 | 14.02 | 3† | 52.36 | 59.54 | 1† | 372.38 | 372.38 |
| glm5 | free | raw | `tower` | all | 435 | 6.23 | 10.18 | 3† | 20.93 | 22.84 | 1† | 323.78 | 323.78 |
| glm5 | free | raw | `merged` | all | 435 | 6.79 | 10.75 | 3† | 18.58 | 22.51 | 1† | 371.76 | 371.76 |
| glm5 | free | raw | `projected` | all | 435 | 6.05 | 9.63 | 3† | 30.80 | 37.94 | 1† | 218.75 | 218.75 |
| glm5 | free | matched | `tower` | all | 435 | 5.55 | 8.13 | 3† | 34.04 | 44.19 | 1† | 345.36 | 345.36 |
| glm5 | free | matched | `merged` | all | 435 | 5.66 | 8.40 | 3† | 39.38 | 49.85 | 1† | 355.40 | 355.40 |
| glm5 | free | matched | `projected` | all | 435 | 5.11 | 7.48 | 3† | 33.50 | 45.21 | 1† | 283.53 | 283.53 |
| kimi_k26 | free | raw | `tower` | all | 435 | 5.70 | 13.68 | 3† | 28.00 | 36.15 | 1† | 205.03 | 205.03 |
| kimi_k26 | free | raw | `merged` | all | 435 | 7.11 | 14.96 | 3† | 27.44 | 34.00 | 1† | 135.38 | 135.38 |
| kimi_k26 | free | raw | `projected` | all | 435 | 6.05 | 12.96 | 3† | 22.01 | 30.97 | 1† | 119.87 | 119.87 |
| kimi_k26 | free | matched | `tower` | all | 435 | 5.33 | 11.11 | 3† | 30.42 | 37.38 | 1† | 212.99 | 212.99 |
| kimi_k26 | free | matched | `merged` | all | 435 | 5.54 | 11.03 | 3† | 31.89 | 40.17 | 1† | 175.81 | 175.81 |
| kimi_k26 | free | matched | `projected` | all | 435 | 5.39 | 10.89 | 3† | 35.54 | 41.57 | 1† | 191.42 | 191.42 |
| mimo26_flash | free | raw | `tower` | all | 435 | 10.99 | 16.62 | 3† | 17.49 | 23.94 | 1† | 352.92 | 352.92 |
| mimo26_flash | free | raw | `merged` | all | 435 | 9.76 | 19.42 | 3† | 24.25 | 28.76 | 1† | 267.54 | 267.54 |
| mimo26_flash | free | raw | `projected` | all | 435 | 9.07 | 16.97 | 3† | 18.70 | 23.19 | 1† | 272.85 | 272.85 |
| mimo26_flash | free | matched | `tower` | all | 435 | 9.88 | 15.54 | 3† | 20.42 | 27.65 | 1† | 365.81 | 365.81 |
| mimo26_flash | free | matched | `merged` | all | 435 | 6.50 | 14.58 | 3† | 34.79 | 42.43 | 1† | 326.05 | 326.05 |
| mimo26_flash | free | matched | `projected` | all | 435 | 6.50 | 15.07 | 3† | 26.56 | 32.99 | 1† | 290.84 | 290.84 |
| mimo26_pro | free | raw | `tower` | all | 435 | 10.95 | 16.83 | 3† | 14.81 | 21.63 | 1† | 345.99 | 345.99 |
| mimo26_pro | free | raw | `merged` | all | 435 | 9.02 | 17.62 | 3† | 25.63 | 33.72 | 1† | 266.12 | 266.12 |
| mimo26_pro | free | raw | `projected` | all | 435 | 9.19 | 16.77 | 3† | 22.06 | 33.91 | 1† | 255.64 | 255.64 |
| mimo26_pro | free | matched | `tower` | all | 435 | 10.06 | 15.88 | 3† | 20.80 | 29.21 | 1† | 362.69 | 362.69 |
| mimo26_pro | free | matched | `merged` | all | 435 | 7.23 | 15.23 | 3† | 32.54 | 42.83 | 1† | 340.04 | 340.04 |
| mimo26_pro | free | matched | `projected` | all | 435 | 7.28 | 14.43 | 3† | 34.29 | 44.78 | 1† | 286.58 | 286.58 |
| minimax_m3 | free | raw | `tower` | all | 435 | 18.96 | 41.80 | 3† | 24.72 | 27.79 | 1† | 136.30 | 136.30 |
| minimax_m3 | free | raw | `merged` | all | 435 | 19.14 | 38.88 | 3† | 32.13 | 33.05 | 1† | 171.61 | 171.61 |
| minimax_m3 | free | raw | `projected` | all | 435 | 10.31 | 25.39 | 3† | 34.68 | 38.17 | 1† | 256.18 | 256.18 |
| minimax_m3 | free | matched | `tower` | all | 435 | 23.65 | 51.62 | 3† | 37.30 | 41.10 | 1† | 104.08 | 104.08 |
| minimax_m3 | free | matched | `merged` | all | 435 | 26.16 | 53.76 | 3† | 25.29 | 34.43 | 1† | 157.54 | 157.54 |
| minimax_m3 | free | matched | `projected` | all | 435 | 9.03 | 21.93 | 3† | 32.13 | 33.06 | 1† | 290.08 | 290.08 |
| moonvit_v2 | free | raw | `tower` | all | 435 | 6.57 | 14.72 | 3† | 21.43 | 26.68 | 1† | 256.54 | 256.54 |
| moonvit_v2 | free | raw | `merged` | all | 435 | 7.85 | 17.94 | 3† | 25.23 | 33.95 | 1† | 225.09 | 225.09 |
| moonvit_v2 | free | raw | `projected` | all | 435 | 8.35 | 18.33 | 3† | 29.91 | 36.50 | 1† | 258.06 | 258.06 |
| moonvit_v2 | free | matched | `tower` | all | 435 | 6.90 | 14.55 | 3† | 26.63 | 35.35 | 1† | 235.61 | 235.61 |
| moonvit_v2 | free | matched | `merged` | all | 435 | 6.35 | 14.48 | 3† | 26.93 | 37.76 | 1† | 286.21 | 286.21 |
| moonvit_v2 | free | matched | `projected` | all | 435 | 6.48 | 13.42 | 3† | 32.15 | 40.58 | 1† | 274.68 | 274.68 |
| muse_glimmer | free | raw | `tower` | all | 435 | 6.34 | 11.51 | 3† | 23.69 | 28.28 | 1† | 355.46 | 355.46 |
| muse_glimmer | free | raw | `merged` | all | 435 | 6.71 | 12.57 | 3† | 27.37 | 33.03 | 1† | 366.12 | 366.12 |
| muse_glimmer | free | raw | `projected` | all | 435 | 4.98 | 9.15 | 3† | 28.29 | 34.43 | 1† | 322.58 | 322.58 |
| muse_glimmer | free | matched | `tower` | all | 435 | 5.60 | 9.60 | 3† | 25.79 | 30.07 | 1† | 342.88 | 342.88 |
| muse_glimmer | free | matched | `merged` | all | 435 | 5.55 | 9.74 | 3† | 30.71 | 36.51 | 1† | 353.66 | 353.66 |
| muse_glimmer | free | matched | `projected` | all | 435 | 4.62 | 8.65 | 3† | 32.14 | 39.57 | 1† | 328.08 | 328.08 |
| nemotron35_bolt | free | raw | `tower` | all | 435 | 5.20 | 8.21 | 3† | 41.95 | 51.41 | 1† | 289.45 | 289.45 |
| nemotron35_bolt | free | raw | `merged` | all | 435 | 6.18 | 9.55 | 3† | 41.65 | 50.38 | 1† | 289.75 | 289.75 |
| nemotron35_bolt | free | raw | `projected` | all | 435 | 6.25 | 10.99 | 3† | 33.87 | 44.39 | 1† | 278.38 | 278.38 |
| nemotron35_bolt | free | matched | `tower` | all | 435 | 5.22 | 8.28 | 3† | 39.88 | 48.66 | 1† | 290.09 | 290.09 |
| nemotron35_bolt | free | matched | `merged` | all | 435 | 5.60 | 9.69 | 3† | 37.77 | 47.98 | 1† | 245.65 | 245.65 |
| nemotron35_bolt | free | matched | `projected` | all | 435 | 5.13 | 8.44 | 3† | 36.12 | 46.36 | 1† | 325.55 | 325.55 |
| nemotron35_super | free | raw | `tower` | all | 435 | 5.32 | 8.00 | 3† | 44.73 | 55.18 | 1† | 289.60 | 289.60 |
| nemotron35_super | free | raw | `merged` | all | 435 | 5.72 | 8.62 | 3† | 42.49 | 51.13 | 1† | 294.61 | 294.61 |
| nemotron35_super | free | raw | `projected` | all | 435 | 6.62 | 11.20 | 3† | 40.89 | 52.80 | 1† | 266.54 | 266.54 |
| nemotron35_super | free | matched | `tower` | all | 435 | 5.12 | 8.07 | 3† | 39.56 | 48.90 | 1† | 294.85 | 294.85 |
| nemotron35_super | free | matched | `merged` | all | 435 | 5.49 | 8.33 | 3† | 40.52 | 50.18 | 1† | 228.72 | 228.72 |
| nemotron35_super | free | matched | `projected` | all | 435 | 6.33 | 11.07 | 3† | 43.24 | 55.17 | 1† | 296.94 | 296.94 |
| nemotron_omni | free | raw | `tower` | all | 435 | 7.09 | 10.95 | 3† | 33.59 | 45.47 | 1† | 197.94 | 197.94 |
| nemotron_omni | free | raw | `merged` | all | 435 | 8.68 | 13.53 | 3† | 32.10 | 45.38 | 1† | 193.92 | 193.92 |
| nemotron_omni | free | raw | `projected` | all | 435 | 5.96 | 9.19 | 3† | 38.17 | 47.47 | 1† | 234.69 | 234.69 |
| nemotron_omni | free | matched | `tower` | all | 435 | 5.66 | 8.21 | 3† | 35.40 | 47.85 | 1† | 283.89 | 283.89 |
| nemotron_omni | free | matched | `merged` | all | 435 | 6.14 | 9.33 | 3† | 40.64 | 52.03 | 1† | 279.26 | 279.26 |
| nemotron_omni | free | matched | `projected` | all | 435 | 5.81 | 9.57 | 3† | 36.13 | 48.14 | 1† | 290.46 | 290.46 |
| qwen3_5 | free | raw | `tower` | all | 435 | 8.77 | 15.25 | 3† | 14.64 | 16.58 | 1† | 321.93 | 321.93 |
| qwen3_5 | free | raw | `merged` | all | 435 | 9.59 | 16.04 | 3† | 14.93 | 20.10 | 1† | 348.70 | 348.70 |
| qwen3_5 | free | raw | `projected` | all | 435 | 6.03 | 11.05 | 3† | 41.47 | 51.55 | 1† | 301.63 | 301.63 |
| qwen3_5 | free | matched | `tower` | all | 435 | 5.96 | 11.20 | 3† | 36.93 | 44.43 | 1† | 218.78 | 218.78 |
| qwen3_5 | free | matched | `merged` | all | 435 | 7.56 | 13.71 | 3† | 36.29 | 41.58 | 1† | 281.05 | 281.05 |
| qwen3_5 | free | matched | `projected` | all | 435 | 5.40 | 9.30 | 3† | 43.82 | 56.09 | 1† | 302.75 | 302.75 |
| siglip2 | free | raw | `tower` | all | 435 | 7.10 | 11.39 | 3† | 45.87 | 56.68 | 1† | 283.00 | 283.00 |
| siglip2 | free | matched | `tower` | all | 435 | 6.97 | 11.03 | 3† | 41.78 | 52.15 | 1† | 312.08 | 312.08 |

