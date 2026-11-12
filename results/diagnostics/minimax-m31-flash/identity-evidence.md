# Identity per-image evidence

The current primary perturbation runner reports thirteen aggregate conditions.
It stores per-image correctness for twelve transformed conditions, each with
its same-run clean correctness. The clean condition therefore already has
measured per-image vectors in every transformed-condition record.

Before paired postprocessing, the delivery helper verifies that all twelve
copies have exactly the same image order and three-seed clean vectors. It
checks 300 unique test IDs, binary correctness, and agreement with the existing
identity aggregate and per-seed accuracies. It then materializes an identity
measurement with those vectors as both clean and perturbed correctness.

This operation preserves training, protocol fields, original scores and every
transformed-condition measurement. The original runner output is retained
byte-for-byte under `raw-probe-payloads/`. The augmented file records the original
hash, published original path, source-runner hash and materialization-helper
hash. `identity-evidence-materialization.json` binds both payload hashes.

The retained method test exercises both models and all three Stages, checks
unchanged scores and protocol fields, exact backups and idempotence, and rejects
inconsistent repeated clean vectors. See `test-identity-evidence.py` and
`tests-identity-evidence.txt`. The primary perturbation runner is unchanged.

The exact materialization source is retained in `postprocess-source-at-run.py.gz`.
Its decompressed SHA256 matches `materialization_script_sha256` in the payloads.
The working helper differs only in trailing whitespace normalization.
