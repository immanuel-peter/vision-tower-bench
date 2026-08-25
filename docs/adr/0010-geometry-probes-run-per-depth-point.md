# Geometry probes run per Relative Depth point, with a multilayer anchor

Probe3D's heads take a list of feature maps and a list of input widths, and its example
commands pass `+backbone.return_multilayer=True`. One probe reads several layers at once
and reports a single number per backbone. That does not produce the Relative Depth curve
this suite is built around, so the geometry pillar runs one probe per Stage and depth
point, passing a single-element list. Each cell then matches the semantic pillar and lands
on the same x-axis.

Running only per-layer probes would cut the tie to the published Probe3D numbers, so every
model also gets one multilayer run over the same eight depth points. That result is the
anchor: it says whether this reimplementation lands where the paper does. A per-layer curve
that disagrees with its own multilayer anchor is a defect to chase, not a finding.

Capacity matching from ADR-0008 extends here, and the arithmetic is close. A multiscale
depth head reads 1,706,752 parameters at a `tower` width of 1024, 3,279,616 at a `merged`
width of 4096, and 4,852,480 at a `projected` width of 7168. The headline claim compares
`tower` against `projected`, which is the pair furthest apart, so each cell fits and
freezes the same PCA reduction the semantic pillar uses and trains one head size
everywhere. The unmatched arm runs too, and the matched result is accepted only if
rankings and curves agree.

Depth and surface normals come first. PLAN.md lists depth and correspondence, but Probe3D
also evaluates surface normals, and normals test spatial structure with less room for a
semantic shortcut than depth has. The three correspondence evaluations look like they score
features directly rather than train a probe, so they do not fit the cache-then-probe shape
and get scoped separately once that is confirmed.
