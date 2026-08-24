#!/usr/bin/env bash
# Prepare a Brev instance for an extraction burst. Safe to run twice.
#
# The root disk is small and the bundled disk mounts at /ephemeral, so the model cache,
# the images, and the feature cache all live there. Deleting the instance destroys
# /ephemeral, which is why ADR-0002 pushes results to Hugging Face as they land.
set -euo pipefail

SCRATCH=${SCRATCH:-/ephemeral}
REPO=${REPO:-$HOME/vision-tower-bench}

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$SCRATCH/hf" "$SCRATCH/data" "$SCRATCH/features"

# uv caches on /ephemeral and builds the venv on the root disk. Its hardlinks do not
# cross filesystems, and the packages it writes instead are missing their shared
# libraries, so torch imports and then dies on libcudnn.
export UV_LINK_MODE=copy

cd "$REPO"
uv sync

# PyPI serves cu130 wheels and the CUDA 12.8 driver on these instances rejects them.
# torch then falls back to CPU and only warns, so exit here instead of billing GPU
# rates for CPU work. pyproject pins the cu129 index for Linux to avoid this.
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
