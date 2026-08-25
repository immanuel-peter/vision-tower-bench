#!/usr/bin/env bash
# Download the GeoNet NYU training archives onto a burst instance.
#
# The archives live behind a OneDrive share and total 131.4 GB, so they do not belong on
# a laptop. Anonymous API access to the folder is blocked, which means the per-file links
# have to come from a browser. Open the folder, right click data1.zip and data2.zip, copy
# the link for each, and pass them here:
#
#   ./fetch_geonet.sh "<data1 link>" "<data2 link>"
#
# Folder: https://hkuhk-my.sharepoint.com/:f:/g/personal/xjqi_hku_hk/Ek0Vm--5oi1GssioLE5LjO0ByLTKpWAG00zYYUCeiydR7g
#
# data11.zip and data21.zip are re-uploads of the same bytes at the same sizes. Use them
# only if the originals stall.
set -euo pipefail

OUT=${OUT:-/ephemeral/data/geonet}
mkdir -p "$OUT"

if [ $# -ne 2 ]; then
    sed -n '2,16p' "$0"
    exit 1
fi

download() {
    local url=$1 name=$2
    # A share link serves the file only with download=1 appended.
    [[ "$url" == *"download=1"* ]] || url="${url}&download=1"
    echo "fetching $name"
    # -C - resumes a partial file, which a 60 GB transfer will need at least once.
    curl -L --fail --progress-bar -C - -o "$OUT/$name" "$url"
}

download "$1" data1.zip
download "$2" data2.zip

df -h "$OUT" | tail -1
echo "extracting, needs about 300 GB free"
for archive in data1 data2; do
    unzip -q -o "$OUT/$archive.zip" -d "$OUT/raw"
done
du -sh "$OUT/raw"
