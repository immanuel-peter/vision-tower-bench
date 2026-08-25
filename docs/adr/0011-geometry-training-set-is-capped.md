# Geometry uses DIODE, and the training set is capped

Geometry evaluates on DIODE rather than NYU with GeoNet annotations. DIODE publishes
direct S3 links with no login, measures depth with a survey-grade laser instead of a
Kinect, ships 1024 by 768 images against NYU's 480 by 640, and covers outdoor scenes as
well as indoor. The GeoNet archives need session cookies copied from a browser onto a
rented machine, which is both awkward and a credential risk.

The validation split is 771 images, 325 indoor and 446 outdoor, all with normals, for
7.7 GB. The training split is 25,458 images for 222 GB. v1 starts on validation alone,
splitting it into train and test, because it answers whether the claim survives without
renting anything.


The cap still binds when the training split arrives. Full patch tokens cost 16.0 MB per
image for one model across eight Relative Depth points, so all 25,458 DIODE training
images would take 407 GB for one model and about 2.4 TB for the roster. ADR-0005 assumed
Probe3D-scale data was small enough to afford full tokens. That holds for a validation
split and fails for any training split. Any training run therefore draws a capped 4,000
image subset once with a fixed seed and reuses it across every model, Stage, and depth
point, for 64 GB per model.

Changing the dataset and capping the training set are both deviations from Probe3D, and
the writeup says so. They weaken every cell by the same amount, and the headline claim
compares `tower` against `projected` within one model, so the comparison survives a weaker
probe. The multilayer anchor in ADR-0010 loses its last purpose here: nobody has published
Probe3D numbers on DIODE, so that run checks this implementation against itself and
nothing else. Keep it, and stop calling it an anchor to the literature.

Two things follow from DIODE that NYU could not give. Depth is metric and reaches 230 m
outdoors against 10 m indoors on NYU, so the depth head's bin range is read from the prep
manifest rather than assumed; a wrong range silently clamps every far pixel. And DIODE
labels each sample indoor or outdoor, so every geometry table splits by scene type at no
extra cost. Whether the Projector destroys spatial information evenly across both is a
question NYU could not pose.

ADR-0001 keeps the project free of autonomous driving apart from one KITTI column. DIODE's
outdoor scenes are tripod scans of streets and campuses, not driving footage, so the rule
holds. The writeup states this rather than leaving a reviewer to wonder why outdoor scenes
appear in the core pillar.

The label budgets in PLAN.md apply on top of the cap. One hundred percent means 4,000
images, not 30,000, and every table says so.

The validation split is small enough to prepare anywhere. A training run belongs on a
burst instance with bundled disk rather than a metered volume, because the cheap CPU types
either cap at 128 GB or meter storage near $0.11 per GB per month, which ADR-0002 already
warns about.
