# All 102 paired comparisons

Every interval below was replayed exactly from its actual paired rows. Each source retains its own point estimate. No matrix point is substituted.

Positive values favor the first side. Normal-error comparisons report second-minus-first error in degrees; all other rows report first-minus-second score in percentage points. Intervals are individual 95% intervals with 10,000 resamples and seed 0, without multiplicity adjustment.

SPair retains all 3,600 source selection IDs; only the 3,408 nonempty pairs enter its paired intervals. The source-backed exclusion union lists all 192 empty-pair IDs and reasons. Class-macro intervals appear separately in the same three sibling-comparison rows.

Production official vLLM TORCH_SDPA BF16 agreement remains failed. Accepted conversion parity uses the independent mathematical reference.

| ID | Task / arm / condition | Comparison | Stages | N | First-side advantage [95% interval] | Class-macro interval | Source |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | recognition/matched/attention | Pro vs K3 | tower:3 vs tower:3 | 1950 | -0.308 [-1.197, +0.564] |  | [JSON](../../bootstrap/000-recognition-kimi_k31_pro-minus-moonvit_v2-tower-L3.json) |
| 1 | recognition/matched/attention | Pro vs K3 | tower:7 vs tower:7 | 1950 | -0.530 [-1.470, +0.427] |  | [JSON](../../bootstrap/001-recognition-kimi_k31_pro-minus-moonvit_v2-tower-L7.json) |
| 2 | recognition/matched/attention | Pro vs K3 | tower:10 vs tower:10 | 1950 | -0.410 [-1.299, +0.462] |  | [JSON](../../bootstrap/002-recognition-kimi_k31_pro-minus-moonvit_v2-tower-L10.json) |
| 3 | recognition/matched/attention | Pro vs K3 | tower:14 vs tower:14 | 1950 | +0.769 [-0.085, +1.658] |  | [JSON](../../bootstrap/003-recognition-kimi_k31_pro-minus-moonvit_v2-tower-L14.json) |
| 4 | recognition/matched/attention | Pro vs K3 | tower:17 vs tower:17 | 1950 | +0.615 [-0.154, +1.385] |  | [JSON](../../bootstrap/004-recognition-kimi_k31_pro-minus-moonvit_v2-tower-L17.json) |
| 5 | recognition/matched/attention | Pro vs K3 | tower:20 vs tower:20 | 1950 | +0.530 [-0.256, +1.316] |  | [JSON](../../bootstrap/005-recognition-kimi_k31_pro-minus-moonvit_v2-tower-L20.json) |
| 6 | recognition/matched/attention | Pro vs K3 | tower:24 vs tower:24 | 1950 | +0.291 [-0.496, +1.094] |  | [JSON](../../bootstrap/006-recognition-kimi_k31_pro-minus-moonvit_v2-tower-L24.json) |
| 7 | recognition/matched/attention | Pro vs K3 | tower:27 vs tower:27 | 1950 | +0.188 [-0.530, +0.906] |  | [JSON](../../bootstrap/007-recognition-kimi_k31_pro-minus-moonvit_v2-tower-L27.json) |
| 8 | recognition/matched/attention | Pro vs K3 | merged:27 vs merged:27 | 1950 | +1.521 [+0.650, +2.410] |  | [JSON](../../bootstrap/008-recognition-kimi_k31_pro-minus-moonvit_v2-merged-L27.json) |
| 9 | recognition/matched/attention | Pro vs K3 | projected:27 vs projected:27 | 1950 | +0.513 [-0.171, +1.214] |  | [JSON](../../bootstrap/009-recognition-kimi_k31_pro-minus-moonvit_v2-projected-L27.json) |
| 10 | recognition/matched/attention | Pro projected vs Tower | projected:27 vs tower:27 | 1950 | +0.786 [+0.103, +1.487] |  | [JSON](../../bootstrap/semantic-kimi_k31_pro-projected-vs-tower-attention-matched.json) |
| 11 | depth/matched | Pro vs K3 | tower:3 vs tower:3 | 115 | -0.716 [-1.212, -0.234] |  | [JSON](../../bootstrap/011-depth-kimi_k31_pro-minus-moonvit_v2-tower-L3.json) |
| 12 | depth/matched | Pro vs K3 | tower:7 vs tower:7 | 115 | +0.135 [-0.468, +0.723] |  | [JSON](../../bootstrap/012-depth-kimi_k31_pro-minus-moonvit_v2-tower-L7.json) |
| 13 | depth/matched | Pro vs K3 | tower:10 vs tower:10 | 115 | +0.053 [-0.503, +0.578] |  | [JSON](../../bootstrap/013-depth-kimi_k31_pro-minus-moonvit_v2-tower-L10.json) |
| 14 | depth/matched | Pro vs K3 | tower:14 vs tower:14 | 115 | +0.545 [+0.155, +0.936] |  | [JSON](../../bootstrap/014-depth-kimi_k31_pro-minus-moonvit_v2-tower-L14.json) |
| 15 | depth/matched | Pro vs K3 | tower:17 vs tower:17 | 115 | +0.364 [-0.029, +0.777] |  | [JSON](../../bootstrap/015-depth-kimi_k31_pro-minus-moonvit_v2-tower-L17.json) |
| 16 | depth/matched | Pro vs K3 | tower:20 vs tower:20 | 115 | +0.014 [-0.396, +0.414] |  | [JSON](../../bootstrap/016-depth-kimi_k31_pro-minus-moonvit_v2-tower-L20.json) |
| 17 | depth/matched | Pro vs K3 | tower:24 vs tower:24 | 115 | -1.028 [-1.648, -0.438] |  | [JSON](../../bootstrap/017-depth-kimi_k31_pro-minus-moonvit_v2-tower-L24.json) |
| 18 | depth/matched | Pro vs K3 | tower:27 vs tower:27 | 115 | +0.637 [+0.077, +1.195] |  | [JSON](../../bootstrap/018-depth-kimi_k31_pro-minus-moonvit_v2-tower-L27.json) |
| 19 | depth/matched | Pro vs K3 | merged:27 vs merged:27 | 115 | -0.014 [-0.630, +0.594] |  | [JSON](../../bootstrap/019-depth-kimi_k31_pro-minus-moonvit_v2-merged-L27.json) |
| 20 | depth/matched | Pro vs K3 | projected:27 vs projected:27 | 115 | +0.922 [+0.296, +1.582] |  | [JSON](../../bootstrap/020-depth-kimi_k31_pro-minus-moonvit_v2-projected-L27.json) |
| 21 | depth/matched | Pro projected vs Tower | projected:27 vs tower:27 | 115 | +7.948 [+6.290, +9.714] |  | [JSON](../../bootstrap/geometry-depth-matched-kimi_k31_pro-projected-vs-tower.json) |
| 22 | normal/matched | Pro vs K3 | tower:3 vs tower:3 | 115 | +0.047 [-0.029, +0.128] |  | [JSON](../../bootstrap/022-normal-kimi_k31_pro-minus-moonvit_v2-tower-L3.json) |
| 23 | normal/matched | Pro vs K3 | tower:7 vs tower:7 | 115 | -0.131 [-0.271, -0.010] |  | [JSON](../../bootstrap/023-normal-kimi_k31_pro-minus-moonvit_v2-tower-L7.json) |
| 24 | normal/matched | Pro vs K3 | tower:10 vs tower:10 | 115 | +0.185 [+0.065, +0.318] |  | [JSON](../../bootstrap/024-normal-kimi_k31_pro-minus-moonvit_v2-tower-L10.json) |
| 25 | normal/matched | Pro vs K3 | tower:14 vs tower:14 | 115 | -0.023 [-0.128, +0.088] |  | [JSON](../../bootstrap/025-normal-kimi_k31_pro-minus-moonvit_v2-tower-L14.json) |
| 26 | normal/matched | Pro vs K3 | tower:17 vs tower:17 | 115 | -1.159 [-1.462, -0.860] |  | [JSON](../../bootstrap/026-normal-kimi_k31_pro-minus-moonvit_v2-tower-L17.json) |
| 27 | normal/matched | Pro vs K3 | tower:20 vs tower:20 | 115 | -0.087 [-0.194, +0.020] |  | [JSON](../../bootstrap/027-normal-kimi_k31_pro-minus-moonvit_v2-tower-L20.json) |
| 28 | normal/matched | Pro vs K3 | tower:24 vs tower:24 | 115 | -0.185 [-0.308, -0.068] |  | [JSON](../../bootstrap/028-normal-kimi_k31_pro-minus-moonvit_v2-tower-L24.json) |
| 29 | normal/matched | Pro vs K3 | tower:27 vs tower:27 | 115 | +1.046 [+0.604, +1.487] |  | [JSON](../../bootstrap/029-normal-kimi_k31_pro-minus-moonvit_v2-tower-L27.json) |
| 30 | normal/matched | Pro vs K3 | merged:27 vs merged:27 | 115 | +0.380 [+0.196, +0.564] |  | [JSON](../../bootstrap/030-normal-kimi_k31_pro-minus-moonvit_v2-merged-L27.json) |
| 31 | normal/matched | Pro vs K3 | projected:27 vs projected:27 | 115 | +0.466 [+0.282, +0.651] |  | [JSON](../../bootstrap/031-normal-kimi_k31_pro-minus-moonvit_v2-projected-L27.json) |
| 32 | normal/matched | Pro projected vs Tower | projected:27 vs tower:27 | 115 | +2.646 [+2.113, +3.194] |  | [JSON](../../bootstrap/geometry-normal-matched-kimi_k31_pro-projected-vs-tower.json) |
| 33 | navi | Pro vs K3 | tower:27 vs tower:27 | 555 | +0.900 [+0.679, +1.117] |  | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-navi-tower.json) |
| 34 | navi | Pro vs K3 | merged:27 vs merged:27 | 555 | +0.309 [+0.085, +0.536] |  | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-navi-merged.json) |
| 35 | navi | Pro vs K3 | projected:27 vs projected:27 | 555 | +0.410 [+0.199, +0.625] |  | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-navi-projected.json) |
| 36 | navi | Pro projected vs Tower | projected:27 vs tower:27 | 555 | +5.599 [+5.114, +6.084] |  | [JSON](../../bootstrap/correspondence-navi-kimi_k31_pro-projected-vs-tower.json) |
| 37 | scannet | Pro vs K3 | tower:27 vs tower:27 | 1500 | +0.182 [+0.149, +0.216] |  | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-scannet-tower.json) |
| 38 | scannet | Pro vs K3 | merged:27 vs merged:27 | 1500 | +0.207 [+0.154, +0.260] |  | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-scannet-merged.json) |
| 39 | scannet | Pro vs K3 | projected:27 vs projected:27 | 1500 | +0.226 [+0.171, +0.281] |  | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-scannet-projected.json) |
| 40 | scannet | Pro projected vs Tower | projected:27 vs tower:27 | 1500 | +2.188 [+2.066, +2.311] |  | [JSON](../../bootstrap/correspondence-scannet-kimi_k31_pro-projected-vs-tower.json) |
| 41 | spair | Pro vs K3 | tower:27 vs tower:27 | 3408 | +0.955 [+0.408, +1.515] | +0.932 [+0.368, +1.496] | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-spair-tower.json) |
| 42 | spair | Pro vs K3 | merged:27 vs merged:27 | 3408 | +1.446 [+0.857, +2.032] | +1.417 [+0.840, +2.011] | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-spair-merged.json) |
| 43 | spair | Pro vs K3 | projected:27 vs projected:27 | 3408 | +1.343 [+0.764, +1.925] | +1.314 [+0.744, +1.888] | [JSON](../../bootstrap/correspondence-kimi_k31_pro-vs-moonvit_v2-spair-projected.json) |
| 44 | spair | Pro projected vs Tower | projected:27 vs tower:27 | 3408 | -0.804 [-1.540, -0.052] |  | [JSON](../../bootstrap/correspondence-spair-kimi_k31_pro-projected-vs-tower.json) |
| 45 | perturbation/identity | Pro vs K3 | tower:27 vs tower:27 | 300 | +4.000 [+1.222, +6.889] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 46 | perturbation/identity | Pro vs K3 | merged:27 vs merged:27 | 300 | +2.778 [+0.111, +5.556] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 47 | perturbation/identity | Pro vs K3 | projected:27 vs projected:27 | 300 | -1.444 [-4.000, +1.222] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 48 | perturbation/identity | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -2.333 [-4.889, +0.333] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 49 | perturbation/scale_l1 | Pro vs K3 | tower:27 vs tower:27 | 300 | +3.778 [+0.889, +6.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 50 | perturbation/scale_l1 | Pro vs K3 | merged:27 vs merged:27 | 300 | +3.667 [+1.000, +6.444] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 51 | perturbation/scale_l1 | Pro vs K3 | projected:27 vs projected:27 | 300 | -1.000 [-3.667, +1.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 52 | perturbation/scale_l1 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -2.556 [-5.000, -0.111] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 53 | perturbation/scale_l2 | Pro vs K3 | tower:27 vs tower:27 | 300 | +2.889 [+0.222, +5.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 54 | perturbation/scale_l2 | Pro vs K3 | merged:27 vs merged:27 | 300 | +4.556 [+1.778, +7.333] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 55 | perturbation/scale_l2 | Pro vs K3 | projected:27 vs projected:27 | 300 | -0.556 [-3.333, +2.222] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 56 | perturbation/scale_l2 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -0.667 [-3.333, +1.889] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 57 | perturbation/scale_l3 | Pro vs K3 | tower:27 vs tower:27 | 300 | +4.778 [+1.778, +7.778] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 58 | perturbation/scale_l3 | Pro vs K3 | merged:27 vs merged:27 | 300 | +4.556 [+1.667, +7.444] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 59 | perturbation/scale_l3 | Pro vs K3 | projected:27 vs projected:27 | 300 | +1.333 [-1.667, +4.333] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 60 | perturbation/scale_l3 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -0.222 [-2.556, +2.222] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 61 | perturbation/scale_l4 | Pro vs K3 | tower:27 vs tower:27 | 300 | +5.000 [+2.333, +7.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 62 | perturbation/scale_l4 | Pro vs K3 | merged:27 vs merged:27 | 300 | +1.667 [-1.333, +4.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 63 | perturbation/scale_l4 | Pro vs K3 | projected:27 vs projected:27 | 300 | -0.111 [-2.667, +2.556] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 64 | perturbation/scale_l4 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -1.889 [-4.556, +0.778] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 65 | perturbation/occlusion_l1 | Pro vs K3 | tower:27 vs tower:27 | 300 | +6.222 [+3.444, +9.111] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 66 | perturbation/occlusion_l1 | Pro vs K3 | merged:27 vs merged:27 | 300 | +4.000 [+1.222, +6.889] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 67 | perturbation/occlusion_l1 | Pro vs K3 | projected:27 vs projected:27 | 300 | -3.333 [-6.222, -0.444] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 68 | perturbation/occlusion_l1 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -3.889 [-6.333, -1.444] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 69 | perturbation/occlusion_l2 | Pro vs K3 | tower:27 vs tower:27 | 300 | +4.111 [+1.333, +6.889] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 70 | perturbation/occlusion_l2 | Pro vs K3 | merged:27 vs merged:27 | 300 | +1.667 [-1.444, +4.778] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 71 | perturbation/occlusion_l2 | Pro vs K3 | projected:27 vs projected:27 | 300 | -2.333 [-5.336, +0.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 72 | perturbation/occlusion_l2 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -2.778 [-5.667, +0.111] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 73 | perturbation/occlusion_l3 | Pro vs K3 | tower:27 vs tower:27 | 300 | +7.333 [+4.556, +10.222] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 74 | perturbation/occlusion_l3 | Pro vs K3 | merged:27 vs merged:27 | 300 | +2.778 [-0.667, +6.111] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 75 | perturbation/occlusion_l3 | Pro vs K3 | projected:27 vs projected:27 | 300 | -3.889 [-7.000, -0.778] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 76 | perturbation/occlusion_l3 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -4.556 [-7.222, -2.000] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 77 | perturbation/occlusion_l4 | Pro vs K3 | tower:27 vs tower:27 | 300 | +3.889 [+1.556, +6.222] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 78 | perturbation/occlusion_l4 | Pro vs K3 | merged:27 vs merged:27 | 300 | +4.222 [+1.889, +6.556] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 79 | perturbation/occlusion_l4 | Pro vs K3 | projected:27 vs projected:27 | 300 | -3.667 [-6.333, -1.222] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 80 | perturbation/occlusion_l4 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -1.222 [-3.333, +0.889] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 81 | perturbation/motion_blur_l1 | Pro vs K3 | tower:27 vs tower:27 | 300 | +5.000 [+2.444, +7.558] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 82 | perturbation/motion_blur_l1 | Pro vs K3 | merged:27 vs merged:27 | 300 | +2.000 [-0.778, +4.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 83 | perturbation/motion_blur_l1 | Pro vs K3 | projected:27 vs projected:27 | 300 | -2.111 [-5.000, +0.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 84 | perturbation/motion_blur_l1 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -1.667 [-4.111, +0.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 85 | perturbation/motion_blur_l2 | Pro vs K3 | tower:27 vs tower:27 | 300 | +7.222 [+4.000, +10.556] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 86 | perturbation/motion_blur_l2 | Pro vs K3 | merged:27 vs merged:27 | 300 | +4.556 [+1.444, +7.778] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 87 | perturbation/motion_blur_l2 | Pro vs K3 | projected:27 vs projected:27 | 300 | -1.778 [-4.778, +1.222] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 88 | perturbation/motion_blur_l2 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -4.778 [-7.556, -2.111] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 89 | perturbation/motion_blur_l3 | Pro vs K3 | tower:27 vs tower:27 | 300 | +5.333 [+2.444, +8.333] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 90 | perturbation/motion_blur_l3 | Pro vs K3 | merged:27 vs merged:27 | 300 | +4.667 [+1.667, +7.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 91 | perturbation/motion_blur_l3 | Pro vs K3 | projected:27 vs projected:27 | 300 | -3.778 [-6.778, -0.778] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 92 | perturbation/motion_blur_l3 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -4.444 [-7.111, -1.889] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 93 | perturbation/motion_blur_l4 | Pro vs K3 | tower:27 vs tower:27 | 300 | +4.222 [+1.889, +6.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-tower.json) |
| 94 | perturbation/motion_blur_l4 | Pro vs K3 | merged:27 vs merged:27 | 300 | +4.778 [+1.667, +8.000] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-merged.json) |
| 95 | perturbation/motion_blur_l4 | Pro vs K3 | projected:27 vs projected:27 | 300 | -1.778 [-4.333, +0.667] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-vs-moonvit_v2-projected.json) |
| 96 | perturbation/motion_blur_l4 | Pro projected vs Tower | projected:27 vs tower:27 | 300 | -2.222 [-4.889, +0.333] |  | [JSON](../../bootstrap/perturbation-kimi_k31_pro-projected-vs-tower-all-conditions.json) |
| 97 | kitti | Pro vs K3 | tower:27 vs tower:27 | 150 | -0.028 [-0.198, +0.128] |  | [JSON](../../bootstrap/097-kitti-kimi_k31_pro-minus-moonvit_v2-tower-L27.json) |
| 98 | kitti | Pro vs K3 | projected:27 vs projected:27 | 150 | +0.140 [-0.048, +0.328] |  | [JSON](../../bootstrap/098-kitti-kimi_k31_pro-minus-moonvit_v2-projected-L27.json) |
| 99 | kitti | Pro projected vs Tower | projected:27 vs tower:27 | 150 | +0.379 [+0.122, +0.650] |  | [JSON](../../bootstrap/geometry-depth-kitti-matched-kimi_k31_pro-projected-vs-tower.json) |
| 100 | depth/raw | Pro projected vs Tower | projected:27 vs tower:27 | 115 | +1.644 [+0.229, +3.050] |  | [JSON](../../bootstrap/geometry-depth-raw-kimi_k31_pro-projected-vs-tower.json) |
| 101 | normal/raw | Pro projected vs Tower | projected:27 vs tower:27 | 115 | +2.908 [+2.234, +3.611] |  | [JSON](../../bootstrap/geometry-normal-raw-kimi_k31_pro-projected-vs-tower.json) |
