# AGENTS.md

Frozen-feature probing of vision Towers. Name things with the vocabulary in [CONTEXT.md](CONTEXT.md) (Tower, Projector, Stage, Connector).

## Where files go

Every file has a home. The branch carries code and releases; measurements, notes and findings live off the branch.

| What | Where |
|---|---|
| Tower adapter and its `export` | `vtb/adapters/<module>.py`, as a new class in an existing module when the Tower shares a family with one |
| Parity check | `vtb/parity/<module>.py`, same module name as the adapter |
| Registration | one `Tower(...)` line in `vtb/roster.py`; everything else derives from the roster |
| Tests | new cases in the existing `tests/` file that covers the behaviour; per-Tower cases go in `tests/test_towers.py` |
| Release card and config | `hf/<repo>/`, written by `python -m vtb.export <slug>` |
| Logic a script needs | `vtb/`; `scripts/` holds thin CLIs over it |
| Measurements | `results/`, uploaded to the Dataset as described in [docs/results.md](docs/results.md) |
| Notes, logs, checklists, one-off scripts, scratch analysis | `tmp/<task>/` |
| Findings | the PR body and your final reply |

`results/`, `tmp/`, `cache/` and `data/` are gitignored. `tests/test_layout.py` fails on tracked files outside this map.

## Tasks

- **New Tower**, whether adding or evaluating one: follow [docs/new-tower.md](docs/new-tower.md).
- **Measurements**, producing or reading them: [docs/results.md](docs/results.md).
- **Issues and PRs**: file issues with `gh issue create --template <name>.md` from `.github/ISSUE_TEMPLATE/`. Write PR bodies from `.github/pull_request_template.md`; `gh pr create --body` skips the template, so copy its sections in.

## Scope

A PR does the one thing its task names. Report adjacent cleanups, stale code and follow-up ideas in your reply, and the human decides which become tasks.

## Checks

`VTB_SKIP_WEIGHTS=1 uv run pytest -q` is the fast suite CI runs. The full suite downloads every Tower's weights.
