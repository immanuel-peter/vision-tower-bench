# Bolt final Tower sanity check, October 18 2026

Bolt's matrices cover only the `merged` and `projected` Stages. This single
cell checks its final Tower against Super's on the identical protocol: all
13000 ImageNet-100 validation images (9100 train), 4x4 pooled tokens, final
Tower layer 32, mean readout, raw capacity. Eleven-rate grid `[1e-5 ... 1.0]`,
20 epochs, three seeds.

| Tower | test accuracy | std | rate | source |
|---|---|---|---|---|
| nemotron35_super | 0.8807 | 0.0005 | 0.003 | `tower:32` in [../../recognition/nemotron35_super_probe_mean_raw.json](../../recognition/nemotron35_super_probe_mean_raw.json) |
| nemotron35_bolt | 0.8827 | 0.0009 | 0.003 | [nemotron35_bolt_tower32_mean_raw.json](nemotron35_bolt_tower32_mean_raw.json) |

Bolt is 0.0020 above Super. The Bolt cache holds only the final Tower layer.
Reproduce it with `vtb.extract --model nemotron35_bolt --pool 4 --stages tower
--final-stages tower`, then `vtb.probe_run --readout mean --stages tower
--depth-points 32`.
