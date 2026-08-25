#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/ephemeral/data/diode}
SPLIT=${SPLIT:-val}
BASE=http://diode-dataset.s3.amazonaws.com
mkdir -p "$OUT/raw"

case "$SPLIT" in
    val)   FILES="val.tar.gz val_normals.tar.gz" ;;       # 2.77 + 4.95 GB, 771 images
    train) FILES="train.tar.gz train_normals.tar.gz" ;;   # 86.75 + 135.22 GB, 25,458 images
    *) echo "SPLIT must be val or train"; exit 1 ;;
esac

for f in $FILES; do
    [ -f "$OUT/raw/$f" ] || curl -L --fail --progress-bar -C - -o "$OUT/raw/$f" "$BASE/$f"
    tar -xzf "$OUT/raw/$f" -C "$OUT/raw"
done

du -sh "$OUT/raw"
