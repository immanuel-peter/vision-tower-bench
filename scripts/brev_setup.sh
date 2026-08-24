#!/usr/bin/env bash
# Prepare a Brev instance for an extraction burst. Safe to run twice.
#
# The instance root disk is small and the bundled disk is mounted at /ephemeral, so the
# model cache, the images, and the feature cache all live there. /ephemeral does not
# survive the instance, which is why ADR-0002 pushes results off the box as they land.
set -euo pipefail

SCRATCH=${SCRATCH:-/ephemeral}
REPO=${REPO:-$HOME/vision-tower-bench}

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$SCRATCH/hf" "$SCRATCH/data" "$SCRATCH/features"

cd "$REPO"
uv sync

# PyPI serves cu130 wheels, which the CUDA 12.8 driver on these instances rejects.
# pyproject pins the cu129 index for Linux, so this check catches a silent CPU fallback.
uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(f"torch {torch.__version__} cannot see the GPU")
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}, bf16 {torch.cuda.is_bf16_supported()}")
PY

cat <<ENV

Ready. Export these before running extraction:

  export PATH="\$HOME/.local/bin:\$PATH"
  export HF_HOME=$SCRATCH/hf
ENV
