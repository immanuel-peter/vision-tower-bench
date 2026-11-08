# Execution sidecars

These 81 logs and validator/runtime records were added by the K3.1 Pro delivery.
Their contents are unchanged. `relocations.json` maps each former canonical-directory
path to its current path and original SHA256. Canonical score JSONs remain in place.

Active consumers in `vtb/`, `scripts/`, `tests/`, the results index and top-level
writeups do not read the relocated execution sidecars. The original scheduler and
audit snapshots under this diagnostic directory are frozen execution evidence;
their recorded commands and paths are preserved, and their raw `/tmp/vtb-delivery`
outputs remain available. To inspect a committed sidecar from an old recorded path,
resolve it through `relocations.json`. Do not rewrite historical commands or scores.
