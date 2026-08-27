#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/ephemeral/data/diode}
SPLIT=${SPLIT:-val}
RETRIES=${RETRIES:-12}
BASE=http://diode-dataset.s3.amazonaws.com
mkdir -p "$OUT/raw"

case "$SPLIT" in
    val)   FILES="val.tar.gz val_normals.tar.gz" ;;       # 2.77 + 4.95 GB, 771 images
    train) FILES="train.tar.gz train_normals.tar.gz" ;;   # 86.75 + 135.22 GB, 25,458 images
    *) echo "SPLIT must be val or train"; exit 1 ;;
esac

remote_size() {
    curl -sIL --fail "$1" \
        | awk 'BEGIN { IGNORECASE = 1 } /^content-length:/ { n = $2 } END { gsub(/\r/, "", n); print n + 0 }'
}

local_size() {
    [ -f "$1" ] && stat -c %s "$1" || echo 0
}

# Resume partial downloads because the bucket drops long connections.
fetch() {
    local url=$1 dest=$2 want attempt=1 have
    want=$(remote_size "$url")
    [ "$want" -gt 0 ] || { echo "could not read a size for $url"; exit 1; }
    while [ "$attempt" -le "$RETRIES" ]; do
        have=$(local_size "$dest")
        if [ "$have" -eq "$want" ]; then
            echo "$(basename "$dest") complete, $want bytes"
            return 0
        fi
        [ "$have" -le "$want" ] || { echo "$dest is larger than the remote object, delete it"; exit 1; }
        echo "$(basename "$dest") attempt $attempt/$RETRIES from byte $have of $want"
        curl -L --fail --progress-bar -C - -o "$dest" "$url" || true
        attempt=$((attempt + 1))
    done
    echo "gave up on $url after $RETRIES attempts"
    exit 1
}

for f in $FILES; do
    fetch "$BASE/$f" "$OUT/raw/$f"
    tar -xzf "$OUT/raw/$f" -C "$OUT/raw"
done

du -sh "$OUT/raw"
