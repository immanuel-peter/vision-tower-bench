# Channel-standardization diagnostic

The parent recovered the proposed patch from
`r2:vtb/backups/v3/2026-10-19/proposed/reducer-channel-standardization.patch`.
Its SHA256 is `f5a2bc1dcda0e6be9a2d6abea9c1bcc9cb5817282ff6406489b5022dc65e330a`.
The exact patch is retained beside this note.

The opt-in `--standardize-channels --match-capacity` path in `vtb.probe_run`
uses `StandardizedReducer`. It flattens training tokens to float32, computes
the channel mean and sample standard deviation with default correction 1, clamps
the scale at 1e-6, then fits randomized low-rank SVD to standardized training
tokens with q=min(width+16,min(flat.shape)). Inference computes
((tokens-mean)/scale)@basis. The training subset and seed-0 fork remain unchanged.

`probe.Reducer`, primary defaults and primary output payloads stay unchanged.
Diagnostic cells identify the reducer explicitly. The four attention-matched
label fractions use the full 13000-image cache with fixed validation and test
splits, seed-0 LR selection and three-seed selected-LR retraining.
These results supplement the existing standardized figure source; they never
replace the primary measurements.

Tests cover the exact recovered formula, sample standard deviation, constant
channels, scale invariance, training-only fit, RNG preservation and the primary
default. Measurements remain pending GPU regrant.
