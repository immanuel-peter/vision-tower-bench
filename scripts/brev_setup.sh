#!/usr/bin/env bash
# Prepare Brev for extraction; safe to rerun, but push results before shutdown deletes the cache.
set -euo pipefail

SCRATCH=${SCRATCH:-/ephemeral}
REPO=${REPO:-$HOME/vision-tower-bench}

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$SCRATCH/hf" "$SCRATCH/data" "$SCRATCH/features"

# Copy across filesystems so torch retains its shared libraries.
export UV_LINK_MODE=copy

cd "$REPO"
uv sync

# Stop if the CUDA 12.8 driver rejects the selected cu129 wheels and torch falls back to CPU.
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
