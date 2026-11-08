# Reducer identity and compatibility

`vtb.probe.StandardizedReducer` is the sole channel-standardized diagnostic implementation.
`fit_reducer(features, split, width, *, standardize_channels=False)` retains the primary default.
Both methods fit training images only, isolate the fit RNG at seed 0, and oversample SVD by 16.
The diagnostic uses sample standard deviation with correction 1 and a scale floor of 1e-6.
Fixed-input comparisons against all four prior implementations are bit-exact; see
`structural-cleanup/reducer-equivalence.json`. This consolidation does not change the method.

New recognition cells and perturbation protocols carry a complete `method` descriptor.
Comparisons validate that descriptor before comparing metrics. Primary PCA, standardized PCA
and unreduced readouts are distinct methods. Equal old protocol dictionaries are insufficient.

Historical score files are unchanged. `results/method-provenance.json` binds compatibility
certificates to the SHA256 of sorted compact JSON, including all scores and source metadata.
Certificates come from recorded producer commands, exact assemblies of certified cells,
explicit diagnostic markers, the archived R2 collection, or byte-identical canonical records
from the pinned pre-diagnostic producer. Paths or missing flags alone are not certificates.
The accompanying `method-identity-audit.json` lists unresolved records as `unknown`.
Unknown records cannot participate in comparisons until their provenance is resolved.
The archived diagnostic collection is never treated as primary merely because its metadata
predates the explicit descriptor. A changed payload no longer matches its certificate.

This registry audits recognition and perturbation reducer identity. Other task payloads may
appear as unknown in the inventory; this does not invalidate their recorded task metrics.
No experiments were retrained and no historical measurements were rewritten.
