## What

## How it was checked

<!-- New Tower PRs fill the section below. Delete it otherwise. -->

## New Tower

- **Tower:** `<slug>`, parent `<repo>@<revision>`
- **Release:** `hf/<repo>`, size, Hub link
- **Parity:** `python -m vtb.parity <slug>` device, dtype and result, with `max_abs` for any cell that is not bit-exact
- **Measurements:** Hub PR link for each uploaded axis, and the `cohorts` entry added to `results/index.json`
- **Findings:** what the Capability Profile shows against the nearest existing Towers, per axis and Stage
- **Open questions:**
