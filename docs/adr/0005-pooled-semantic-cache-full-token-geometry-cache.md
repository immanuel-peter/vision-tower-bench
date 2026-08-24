# Pooled feature cache for semantics, full patch tokens for geometry

A measured extraction of DINOv2 ViT-L/14 over 2000 COCO photographs at 448 square
(`docs/measurements/dinov2-448-cache-size.json`) writes 2.00 MB per image per Relative
Depth point, so 16.00 MB across the eight points, with no measurable safetensors
overhead. Caching every patch token for ImageNet-100 therefore costs 2.18 TB per model
and 13.1 TB across the roster, and ImageNet-1K costs 129 TB. Wider Towers push those
totals 1.3 to 1.5 times higher, because DINOv2 is only 1024-dimensional. Neither figure
fits the 429 GB of free space on the M4 Max, the bundled disks on the burst instances,
or the $150 ceiling in ADR-0002 once metered object storage enters the picture.

The semantic pillar therefore caches a 4x4 spatially pooled grid instead of all 1024
patch tokens, which brings ImageNet-100 to roughly 34 GB per model and 205 GB across the
roster. The geometry pillar keeps full patch tokens, because depth and correspondence
need the spatial resolution and they carry the headline claim, and because Probe3D-scale
data is small enough to afford it. Pooling is identical for every model and every Stage,
so the shared attention pool still reads spatial tokens and cross-model comparison holds.
An 8x8 grid over the full image set was rejected at 818 GB because it does not fit
locally and would add Cloudflare R2 plus a sync step to a project that otherwise needs
neither. Subsampling ImageNet-100 to 26k images at 8x8 was rejected because the label
budget sweep already tests label efficiency, and cutting the image pool would confound it.

Pooling stays an argument until it is measured, so v1 also caches full patch tokens for a
fixed 5000-image semantic subset, roughly 80 GB across the roster. The semantic probe runs
both ways on at least two Towers, and the pooled cache is accepted only if the model
ranking and the shape of the Relative Depth curves agree between them. A disagreement is a
result to report, not a defect to bury. This validation joins parity tests and the control
models on the list of things the cut order in ADR-0003 never touches.
