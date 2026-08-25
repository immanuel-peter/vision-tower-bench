#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/ephemeral/data/geonet}
mkdir -p "$OUT"

if [ $# -ne 2 ]; then
    echo "usage: $0 '<data1 link>' '<data2 link>'"
    echo "links: https://hkuhk-my.sharepoint.com/:f:/g/personal/xjqi_hku_hk/Ek0Vm--5oi1GssioLE5LjO0ByLTKpWAG00zYYUCeiydR7g"
    exit 1
fi

download() {
    local url=$1 name=$2
    [[ "$url" == *"download=1"* ]] || url="${url}&download=1"
    echo "fetching $name"
    curl -L --fail --progress-bar -C - -o "$OUT/$name" "$url"
}

download "$1" data1.zip
download "$2" data2.zip

df -h "$OUT" | tail -1
echo
echo "Do not unzip these. prep_geonet.py reads samples straight out of the archives:"
echo "  uv run python scripts/prep_geonet.py --zips $OUT/data1.zip $OUT/data2.zip \\"
echo "      --out $OUT/prepared --split train --cap 4000"
