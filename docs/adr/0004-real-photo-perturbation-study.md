# Perturbation study uses transformed real photos, not rendered scenes

The controlled robustness experiments apply programmatic transforms (object scale, occlusion, blur — one factor at a time, exact parameters recorded per image) to a fixed subset of real photographs rather than rendering synthetic scenes. Rendered worlds were rejected because the synthetic-to-real gap gives reviewers an easy dismissal of degradation curves, and none of the three v1 factors requires synthesis to generate clean ground truth; duplicate-object counting, the one factor that did, was already cut in ADR-0003.
