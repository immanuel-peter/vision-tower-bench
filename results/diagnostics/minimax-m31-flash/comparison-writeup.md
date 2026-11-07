# MiniMax M3.1 Flash versus compatible M3 controls

Same architecture, weights pinned separately, identical isolated image inference at 448. Every point and interval comes from the same paired run.

Flash minus M3; negative is better for normal error, positive is better for accuracy and recall.

## Main paired measurements

| Task | Stage | Flash | M3 | Delta | 95% interval |
|---|---|---:|---:|---:|---|
| Recognition, matched attention | tower | 57.299 | 57.880 | -0.581 | [-1.983, +0.821] |
| DIODE depth d1, matched | tower | 58.118 | 58.772 | -0.654 | [-2.102, +0.765] |
| DIODE normals mean error, matched | tower | 31.215 | 30.878 | +0.336 | [-0.356, +1.042] |
| DIODE depth d1, raw | tower | 41.853 | 39.814 | +2.040 | [-0.213, +4.405] |
| DIODE normals mean error, raw | tower | 34.382 | 34.651 | -0.268 | [-0.636, +0.112] |
| NAVI recall | tower | 22.677 | 23.087 | -0.410 | [-0.748, -0.067] |
| SCANNET recall | tower | 3.607 | 4.023 | -0.416 | [-0.478, -0.359] |
| SPAIR class-macro PCK | tower | 9.948 | 9.659 | +0.288 | [-0.375, +0.933] |
| Recognition, matched attention | merged | 56.991 | 57.316 | -0.325 | [-1.744, +1.026] |
| DIODE depth d1, matched | merged | 54.266 | 55.660 | -1.394 | [-3.272, +0.491] |
| DIODE normals mean error, matched | merged | 32.016 | 31.475 | +0.541 | [-0.070, +1.152] |
| DIODE depth d1, raw | merged | 42.803 | 36.271 | +6.532 | [+3.367, +9.799] |
| DIODE normals mean error, raw | merged | 34.677 | 34.965 | -0.289 | [-0.667, +0.104] |
| NAVI recall | merged | 20.374 | 21.572 | -1.198 | [-1.655, -0.731] |
| SCANNET recall | merged | 1.606 | 1.859 | -0.252 | [-0.305, -0.202] |
| SPAIR class-macro PCK | merged | 7.802 | 6.867 | +0.935 | [+0.363, +1.525] |
| Recognition, matched attention | projected | 58.479 | 58.906 | -0.427 | [-1.829, +1.026] |
| DIODE depth d1, matched | projected | 50.331 | 48.574 | +1.757 | [-0.406, +3.814] |
| DIODE normals mean error, matched | projected | 33.613 | 34.412 | -0.800 | [-1.302, -0.307] |
| DIODE depth d1, raw | projected | 21.157 | 31.627 | -10.470 | [-13.819, -7.233] |
| DIODE normals mean error, raw | projected | 34.098 | 34.432 | -0.334 | [-0.835, +0.143] |
| NAVI recall | projected | 23.317 | 22.693 | +0.624 | [+0.125, +1.102] |
| SCANNET recall | projected | 2.185 | 1.648 | +0.537 | [+0.477, +0.598] |
| SPAIR class-macro PCK | projected | 10.745 | 9.435 | +1.310 | [+0.669, +1.960] |
| KITTI depth d1 | tower | 89.094 | 89.626 | -0.532 | [-1.034, -0.025] |
| KITTI depth d1 | projected | 82.299 | 77.948 | +4.351 | [+3.176, +5.510] |

Accuracy, recall and d1 values are percentages; their deltas are percentage points. Normal errors and deltas are degrees. An interval spanning zero does not resolve the ordering. SPair uses class-macro paired intervals here; the standard within-model figure retains its historical per-pair protocol and reports a class-macro supplement.

## All perturbation conditions

Identity rows give the clean baselines on the same 300 test images. All conditions use readouts trained on 1,400 clean images, with the original primary reducer. Interpret condition differences relative to these clean readout baselines. The standardized full-recognition label-budget measurements are separate diagnostics.

| Condition | Stage | Flash | M3 | Delta, pp | 95% interval, pp |
|---|---|---:|---:|---:|---|
| identity | tower | 32.667 | 34.556 | -1.889 | [-5.556, +1.667] |
| motion_blur_l1 | tower | 25.444 | 21.778 | +3.667 | [-0.111, +7.444] |
| motion_blur_l2 | tower | 16.000 | 15.556 | +0.444 | [-2.889, +3.778] |
| motion_blur_l3 | tower | 8.889 | 10.111 | -1.222 | [-4.000, +1.444] |
| motion_blur_l4 | tower | 6.667 | 8.667 | -2.000 | [-4.333, +0.222] |
| occlusion_l1 | tower | 22.222 | 25.667 | -3.444 | [-7.111, +0.222] |
| occlusion_l2 | tower | 15.333 | 16.444 | -1.111 | [-4.000, +1.889] |
| occlusion_l3 | tower | 8.333 | 9.222 | -0.889 | [-4.000, +2.111] |
| occlusion_l4 | tower | 5.778 | 4.444 | +1.333 | [-0.556, +3.222] |
| scale_l1 | tower | 34.889 | 33.889 | +1.000 | [-2.556, +4.444] |
| scale_l2 | tower | 33.556 | 32.222 | +1.333 | [-2.222, +5.000] |
| scale_l3 | tower | 27.778 | 27.889 | -0.111 | [-4.000, +3.667] |
| scale_l4 | tower | 21.667 | 21.000 | +0.667 | [-2.889, +4.222] |
| identity | merged | 31.333 | 32.556 | -1.222 | [-5.111, +2.556] |
| motion_blur_l1 | merged | 23.444 | 23.667 | -0.222 | [-3.667, +3.222] |
| motion_blur_l2 | merged | 16.222 | 16.778 | -0.556 | [-3.889, +2.778] |
| motion_blur_l3 | merged | 9.333 | 11.000 | -1.667 | [-4.444, +1.000] |
| motion_blur_l4 | merged | 7.444 | 8.222 | -0.778 | [-3.114, +1.556] |
| occlusion_l1 | merged | 20.444 | 24.333 | -3.889 | [-7.667, -0.111] |
| occlusion_l2 | merged | 16.111 | 17.000 | -0.889 | [-4.000, +2.222] |
| occlusion_l3 | merged | 8.333 | 9.444 | -1.111 | [-3.889, +1.556] |
| occlusion_l4 | merged | 4.667 | 4.556 | +0.111 | [-1.556, +1.778] |
| scale_l1 | merged | 35.333 | 33.556 | +1.778 | [-2.111, +5.667] |
| scale_l2 | merged | 32.333 | 32.111 | +0.222 | [-3.222, +3.889] |
| scale_l3 | merged | 26.556 | 29.111 | -2.556 | [-6.111, +1.000] |
| scale_l4 | merged | 21.111 | 22.222 | -1.111 | [-4.889, +2.667] |
| identity | projected | 37.889 | 36.667 | +1.222 | [-2.333, +4.778] |
| motion_blur_l1 | projected | 26.333 | 24.222 | +2.111 | [-1.667, +5.889] |
| motion_blur_l2 | projected | 16.444 | 16.111 | +0.333 | [-3.000, +3.667] |
| motion_blur_l3 | projected | 10.556 | 12.667 | -2.111 | [-5.003, +0.667] |
| motion_blur_l4 | projected | 7.333 | 9.111 | -1.778 | [-4.111, +0.556] |
| occlusion_l1 | projected | 21.333 | 25.889 | -4.556 | [-8.444, -0.778] |
| occlusion_l2 | projected | 18.111 | 16.667 | +1.444 | [-1.778, +4.889] |
| occlusion_l3 | projected | 10.667 | 11.333 | -0.667 | [-3.667, +2.444] |
| occlusion_l4 | projected | 5.667 | 6.222 | -0.556 | [-2.556, +1.444] |
| scale_l1 | projected | 36.222 | 36.222 | -0.000 | [-4.000, +3.889] |
| scale_l2 | projected | 35.778 | 34.556 | +1.222 | [-2.667, +5.111] |
| scale_l3 | projected | 30.889 | 28.111 | +2.778 | [-1.222, +6.669] |
| scale_l4 | projected | 25.333 | 22.444 | +2.889 | [-0.667, +6.333] |

The complete JSON records each source payload. These paired estimates are not attached to points from a separate primary matrix. See image-isolation.md for the inference correction and source-reference limits.

## Paper note

MiniMax M3 and M3.1 Flash share the evaluated Tower architecture, preprocessing and Stage definitions. Their nominal image-size field does not affect these 448-pixel forwards. With identical image-independent inference and pinned weights, the paired per-Stage differences above isolate checkpoint training effects under the fixed probing protocols. The native packed attention path couples unrelated images, so this comparison uses fresh compatible M3 controls and preserves historical M3 payloads separately. Weight drift is reported for every Tower tensor and Projector tensor; concatenated layer drift is not monotonic and can be dominated by large-norm parameters.
