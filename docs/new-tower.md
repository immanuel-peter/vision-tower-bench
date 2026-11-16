# New Tower

Adding a Tower and evaluating it are one task. The branch carries the release, the Dataset carries the measurements, and the PR body carries the report.

## Contract

Every adapter meets `TowerAdapter` in `vtb/adapters/base.py`, and its features must be:

- **Patch tokens only**, with no class or register tokens.
- **Raster order.** Undo any merge-block or window grouping before the features leave `extract`.
- **A square token grid.** The pooling and dense decoders assume one.

## Steps

1. **Adapter.** Implement the adapter in `vtb/adapters/<module>.py` from the helpers in `vtb/adapters/common.py`. Pin the parent's repo and revision as module constants. Add the `Tower(...)` line to `vtb/roster.py`. The slug is permanent once a result carries it. Done when `VTB_SKIP_WEIGHTS=1 uv run pytest -q tests/test_adapter_contract.py tests/test_roster.py` passes, since both pick the Tower up from the roster.
2. **Export.** Skip this step for a Tower loaded from its upstream release, like the DINO and SigLIP controls. For a Tower pulled from a larger checkpoint, give the module `export(out)` (or `export(out, checkpoint)` when one module serves several checkpoints) that range-reads only the vision tensors, and run `python -m vtb.export <slug>`. Done when `hf/<repo>/` holds the card, config and license.
3. **Draft PR, then upload.** Open a draft PR now. Then stop and ask the human to approve the `hf upload` command that the export printed. Parity and extraction read the release from the Hub, so nothing below runs until the release is published.
4. **Parity.** Add `vtb/parity/<module>.py` with `validate(device, dtype) -> {"weights", "forward", "passed"}`, built from the helpers in `vtb/parity/__init__.py`. Set `DTYPE` or `REQUIRES_CUDA` in the module only when the reference implementation cannot run in FP32 or on CPU. Done when `python -m vtb.parity <slug>` exits 0, meaning bit-exact in FP32 against the pinned parent.
5. **Tests.** Cover what parity cannot see, such as preprocessing, input validation and batch isolation, as cases in `tests/test_towers.py`.
6. **Measurements.** Run every `scripts/*_matrix.sh` with the new slug, writing into `results/<axis>/`. Use the same learning-rate grid as the rest of the roster and three seeds, and let validation pick the rate. The perturbation matrix also needs the slug in `batch_for` and `expected_for` in `scripts/perturbation_matrix.sh`. Upload the results as described in [results.md](results.md).
7. **Report.** Fill the New Tower section of `.github/pull_request_template.md` in the PR body and mark the PR ready. Done when every field there holds a number, a path or a link.
