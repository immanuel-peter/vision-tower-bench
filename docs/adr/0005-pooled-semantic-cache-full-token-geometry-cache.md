# Pooled feature cache for semantics, full patch tokens for geometry

A DINOv2 ViT-L/14 extraction over 2,000 COCO photos at 448 square writes 2.00 MB per
image at each Relative Depth point. Across all eight points, that is 16.00 MB per image
with no measurable safetensors overhead. The measurements are in
`docs/measurements/dinov2-448-cache-size.json`.

Caching every patch token would cost 2.18 TB per model for ImageNet-100 and 13.1 TB for
the full roster. ImageNet-1K would cost 129 TB. Those caches do not fit the M4 Max, the
bundled disks on burst instances, or the $150 ceiling in ADR-0002.

Semantic probes cache a 4x4 spatial grid instead of all 1,024 patch tokens. The measured
ImageNet-100 cost is 34.1 GB for DINOv2 and 80.9 GB for MoonViT-V2. See
`docs/measurements/moonvit-v2-448-cache-size.json`. Stage count matters more than token
width. DINOv2 writes eight slices, one for each Relative Depth point. MoonViT-V2 writes
ten because `merged` and `projected` add two wide slices at the deepest point.

The current roster estimate is about 400 GB. It assumes eight slices for each control
Tower and ten for each multimodal Tower. Replace each estimate with a measurement when
its adapter lands.

Geometry probes keep full patch tokens because depth and correspondence need the spatial
grid. This is affordable only on a capped subset. GeoNet's NYU training set holds roughly
30,000 images, which at 16.0 MB per image would cost 503 GB for one model and about 3 TB
for the roster. ADR-0011 caps the geometry training set instead. Every model and
Stage uses the same pooling code for semantic probes, so the attention pool still receives
a spatial grid. An 8x8 grid over the full image set would take 818 GB and require remote
object storage. Using 26,000 images at 8x8 would confound the separate label-budget sweep.

v1 also caches full patch tokens for a fixed 1,500-image semantic subset, about 25 GB per
model and 150 GB across the roster. An earlier version of this ADR set that subset at
5,000 images and called it 80 GB across the roster. That was wrong by the size of the
roster: 5,000 images at 16.0 MB is 83.9 GB for one model and over 500 GB for six.

Run semantic probes on pooled and full tokens for at least two Towers.
Accept pooling only if model rankings and Relative Depth curves agree. Report any
disagreement. ADR-0003 does not allow cutting this check, the parity tests, or the control
models.

## The validation ran, August 31 2026

It ran on SigLIP2 and Muse Glimmer and returned a split verdict: the mean readout validates
(rankings and Relative Depth curves agree, raw cells identical to four decimals) and the
attention readout does not - the full-token attention heads fail to train at mid Relative
Depth on the 1,500-image subset, so their curves cannot confirm the pooled ones. See
ADR-0019 for the numbers and the consequences. The 1,500-image, 25 GB-per-model subset this
ADR specifies stands; the earlier 5,000-image figure in PLAN.md was the error this ADR
already corrected, and PLAN.md now carries the corrected numbers.
