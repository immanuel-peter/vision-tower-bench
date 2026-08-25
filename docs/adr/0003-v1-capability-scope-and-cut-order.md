# v1 capability scope and cut order

v1 requires semantic linear probes, Probe3D-style depth and correspondence probes, and
the Stage sweep. The geometry probes test the headline claim that the Projector preserves
semantics but loses spatial information.

The Perturbation Study starts with object scale, occlusion, and blur. The Transfer Probe
uses one dataset, KITTI depth. If the schedule slips, cut 896² runs first, reduce the
Perturbation Study to one factor next, and cut Transfer Probe variants last. Keep the
control models and parity tests. Controls make the comparisons meaningful, and parity is
required before publishing an extracted checkpoint.
