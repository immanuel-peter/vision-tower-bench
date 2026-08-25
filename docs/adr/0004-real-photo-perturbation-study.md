# Perturbation study uses transformed real photos, not rendered scenes

Apply object scale, occlusion, and blur transforms to a fixed set of real photos. Change
one factor at a time and record its exact value for every image.

Rendered scenes would add a synthetic-to-real gap without helping these three factors.
Duplicate-object counting did need rendered ground truth, but ADR-0003 already cut it.
