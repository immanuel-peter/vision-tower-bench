# DINOv3 equal-pixel check, October 18 2026

The primary DINOv3 run uses 512 px, so its 16 px patches form the same 32 by 32
grid as DINOv2 at 448 px. This check reruns DINOv3 at 448 px, the same input
pixels as DINOv2, which gives a 28 by 28 grid. It covers only the final Tower
layer (24) at matched capacity, under the primary protocols:

- ImageNet-100 attention readout on 4x4 pooled tokens: 13,000 images, eleven rates, 20 epochs, three seeds.
- DIODE depth and normals on unpooled tokens: 771 images, six rates, ten epochs, three seeds.

Image identities and order match the 512 px caches.

| Task | Metric | 448 px | std | rate | 512 px primary | std | rate | 448 px source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Recognition, attention | top-1 | 0.9176 | 0.0031 | 1e-4 | 0.9137 | 0.0034 | 1e-4 | [dinov3_probe_attention_matched.json](dinov3_probe_attention_matched.json) |
| Depth | delta-1 | 0.7661 | 0.0039 | 0.01 | 0.7677 | 0.0031 | 0.01 | [dinov3_geometry_depth_matched.json](dinov3_geometry_depth_matched.json) |
| Normals | mean error (deg) | 21.4612 | 0.4287 | 0.001 | 21.1196 | 0.4688 | 0.001 | [dinov3_geometry_normal_matched.json](dinov3_geometry_normal_matched.json) |

The 512 px values are layer 24 in
[recognition/dinov3_probe_attention_matched.json](../../recognition/dinov3_probe_attention_matched.json)
and `geometry/dinov3_geometry_{depth,normal}_matched.json`. Each pair differs by
about one seed standard deviation or less: recognition +0.0039 at 448 px, depth
-0.0016, normals 0.34 degrees worse. The equal-grid choice does not drive the
DINOv3 results. These are separate runs with point estimates only. The paired
DINOv3-versus-DINOv2 intervals in `bootstrap/` use 512 px and do not apply here.

Reproduce with `vtb.extract --model dinov3 --resolution 448` (`--pool 4` for
recognition, unpooled for DIODE), then `vtb.probe_run --readout attention
--depth-points 24 --match-capacity` and `vtb.geometry_run --task {depth,normal}
--stages tower --deepest-only --match-capacity`.
