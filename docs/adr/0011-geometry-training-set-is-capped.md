# The geometry training set is capped, and the cap is a budget

GeoNet's NYU training set holds roughly 30,000 images, estimated from the 131.4 GB of
archives and the 9.9 MB each sample takes uncompressed. Full patch tokens cost 16.0 MB per
image for one model across eight Relative Depth points, so caching all of it would take
503 GB for one model and about 3 TB for the roster. ADR-0005 assumed Probe3D-scale data
was small enough to afford full tokens. That holds for the evaluation splits and fails for
this training set.

Geometry therefore trains on a capped subset of 4,000 images, drawn once with a fixed seed
and reused by every model, Stage, and depth point. The cost is 64 GB per model and roughly
400 GB across the roster, which fits the bundled disk on a burst instance. Evaluation uses
the NYUv2 labeled test split of 654 images, which every published NYU number also uses, so
the test side stays standard.

The cap is a deviation from Probe3D, which trains on the full set, and the writeup says so.
It weakens every cell by the same amount, and the headline claim compares `tower` against
`projected` within one model, so the comparison survives a weaker probe. What it does cost
is the multilayer anchor in ADR-0010: trained on 4,000 images rather than 30,000, that run
no longer checks this reimplementation against the published numbers and only checks it
against itself.

The label budgets in PLAN.md apply on top of the cap. One hundred percent means 4,000
images, not 30,000, and every table says so.

Downloading 131.4 GB to keep 4,000 images is wasteful, and a share link cannot serve part
of an archive. The download and the conversion therefore run on a burst instance and only
the converted subset comes back.

Nothing unzips the archives. Extracting them would take about 300 GB to keep 4,000
samples, and scipy reads a file object, so the prep step opens each sample inside the zip.
That drops the instance requirement from roughly 430 GB to about 150 GB, which is the
difference between needing a 512 GB machine and fitting a 256 GB one. Pick an instance
with bundled disk rather than a metered volume: the cheap CPU types either cap at 128 GB
or meter storage near $0.11 per GB per month, which ADR-0002 already warns about.
