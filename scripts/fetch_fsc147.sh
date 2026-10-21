#!/usr/bin/env bash
# Fetch the official FSC-147 release. Images come from the authors' Google Drive zip;
# FALLBACK=hf takes them from the pinned isentropic/FSC147 mirror instead, which
# prep_fsc147.py then checks against the official annotations.
set -euo pipefail

OUT=${OUT:-$HOME/vtb-work/counting-v3/data/fsc147}
FALLBACK=${FALLBACK:-none}          # none | hf
DENSITY=${DENSITY:-0}               # 1 also fetches the authors' density maps (archive.org)
RAW=$OUT/raw
mkdir -p "$RAW"

REPO_COMMIT=00ea1888c3a7c495ae06db0bddc3b90b7db8d52f   # cvlab-stonybrook/LearningToCountEverything
REPO_RAW=https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/$REPO_COMMIT/data
DRIVE_ID=1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S                # FSC147_384_V2.zip, linked from the README
HF_REPO=isentropic/FSC147
HF_REVISION=3e420cb6537e803dd6d4516623ce82a79c0317b8
DENSITY_URL=https://archive.org/download/FSC147-GT/gt_density_map_adaptive_384_VarV2.zip

check() {
    echo "$2  $1" | sha256sum -c --quiet - || { echo "checksum mismatch for $1"; exit 1; }
}

fetch_annotation() {
    [ -f "$RAW/$1" ] || curl -sfL -o "$RAW/$1" "$REPO_RAW/$1"
    check "$RAW/$1" "$2"
}

fetch_annotation annotation_FSC147_384.json 6a1d0f43d12274219a5307cb35b24c85fb275a34b3d5acf5b67f2066f9e0bfe1
fetch_annotation Train_Test_Val_FSC_147.json e281e2d9118cd9ae6f5a2852006b6a24e7a7ea5605d9a6e0fa946dcf93308271
fetch_annotation ImageClasses_FSC147.txt ba1bc6c6a21875919625adb851eaee1aab0ad1f88ecf64d017bebda891b5815e

# Drive answers large files with a confirm page, and with HTML when the daily quota is spent.
fetch_drive() {
    local dest=$RAW/FSC147_384_V2.zip page uuid
    [ -f "$dest" ] && unzip -tq "$dest" >/dev/null && return 0
    page=$(curl -sL "https://drive.google.com/uc?export=download&id=$DRIVE_ID")
    uuid=$(grep -o 'name="uuid" value="[^"]*"' <<<"$page" | cut -d'"' -f4 || true)
    curl -sL --fail -o "$dest.part" \
        "https://drive.usercontent.google.com/download?id=$DRIVE_ID&export=download&confirm=t&uuid=$uuid" || true
    if [ -f "$dest.part" ] && unzip -tq "$dest.part" >/dev/null 2>&1; then
        mv "$dest.part" "$dest"
        return 0
    fi
    rm -f "$dest.part"
    return 1
}

if fetch_drive; then
    mkdir -p "$RAW/drive"
    [ -d "$RAW/drive/images_384_VarV2" ] || unzip -q "$RAW/FSC147_384_V2.zip" -d "$RAW/drive"
elif [ "$FALLBACK" = hf ]; then
    echo "Drive refused the download (quota); using $HF_REPO@$HF_REVISION"
    hf download "$HF_REPO" --repo-type dataset --revision "$HF_REVISION" --max-workers 8 \
        --include "images_384_VarV2/*" --include "*.json" --include "*.txt" --include README.md \
        --local-dir "$RAW/hf_isentropic_${HF_REVISION:0:7}"
else
    echo "Drive refused the download; retry later or set FALLBACK=hf"
    exit 1
fi

if [ "$DENSITY" = 1 ]; then
    mkdir -p "$RAW/archive_org_FSC147-GT"
    [ -f "$RAW/archive_org_FSC147-GT/gt_density_map_adaptive_384_VarV2.zip" ] || curl -sfL \
        -o "$RAW/archive_org_FSC147-GT/gt_density_map_adaptive_384_VarV2.zip" "$DENSITY_URL"
fi

(cd "$RAW" && find . -maxdepth 2 -type f \( -name "*.zip" -o -name "*.json" -o -name "*.txt" \) \
    -not -path "*/.cache/*" | sort | xargs sha256sum) > "$RAW/../raw_SHA256SUMS"
cat "$RAW/../raw_SHA256SUMS"
