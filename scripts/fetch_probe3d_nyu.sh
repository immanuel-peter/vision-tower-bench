#!/usr/bin/env bash
# Download the NYU data the depth and surface normal probes need.
#
# The GeoNet training set is not scriptable. Its authors publish data1.zip and data2.zip
# through OneDrive, so fetch those in a browser and drop them in $OUT before running this:
#   https://hkuhk-my.sharepoint.com/:f:/g/personal/xjqi_hku_hk/Ek0Vm--5oi1GssioLE5LjO0ByLTKpWAG00zYYUCeiydR7g?e=8kAdLZ
set -euo pipefail

OUT=${OUT:-data/nyu}
mkdir -p "$OUT/nyuv2"

# 2.97 GB. The labeled set carries the depth test split.
[ -f "$OUT/nyuv2/nyu_depth_v2_labeled.mat" ] || curl -L --fail --progress-bar \
    -o "$OUT/nyuv2/nyu_depth_v2_labeled.mat" \
    http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

# 0.84 GB. Surface normal annotations from Goyal et al.
if [ ! -d "$OUT/nyuv2/surfacenormal_metadata" ]; then
    curl -L --fail --progress-bar -o "$OUT/nyuv2/snorm.zip" \
        https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip
    unzip -q "$OUT/nyuv2/snorm.zip" -d "$OUT/nyuv2"
    rm "$OUT/nyuv2/snorm.zip"
fi

if [ -f "$OUT/data1.zip" ] && [ -f "$OUT/data2.zip" ]; then
    mkdir -p "$OUT/nyu_geonet"
    unzip -q -o "$OUT/data1.zip" -d "$OUT"
    unzip -q -o "$OUT/data2.zip" -d "$OUT"
    mv "$OUT"/data1/* "$OUT"/data2/* "$OUT/nyu_geonet/"
    rmdir "$OUT/data1" "$OUT/data2"
else
    echo "GeoNet data1.zip and data2.zip are not in $OUT. Download them from the OneDrive"
    echo "link at the top of this script, then run this again to unpack the training set."
fi

du -sh "$OUT"/* 2>/dev/null || true
