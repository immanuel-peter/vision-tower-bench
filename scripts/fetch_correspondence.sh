#!/usr/bin/env bash
# Download the three public Probe3D correspondence subsets. No login is required.
set -euo pipefail

ROOT="${1:?usage: scripts/fetch_correspondence.sh OUT_DIR}"
mkdir -p "$ROOT/navi" "$ROOT/scannet" "$ROOT/spair"

if [ ! -d "$ROOT/navi/navi_v1" ]; then
    curl -L --fail --output "$ROOT/navi/navi_v1.tar.gz" \
        https://storage.googleapis.com/gresearch/navi-dataset/navi_v1.tar.gz
    tar -xzf "$ROOT/navi/navi_v1.tar.gz" -C "$ROOT/navi"
fi

if [ ! -d "$ROOT/scannet/scannet_test_1500" ]; then
    curl -L --fail --output "$ROOT/scannet/scannet_test_1500.tar" \
        'https://drive.usercontent.google.com/download?id=1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd&export=download&confirm=t'
    tar -xf "$ROOT/scannet/scannet_test_1500.tar" -C "$ROOT/scannet"
fi
curl -L --fail --output "$ROOT/scannet/scannet_test_1500/intrinsics.npz" \
    https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/scannet_test_1500/intrinsics.npz
curl -L --fail --output "$ROOT/scannet/scannet_test_1500/test.npz" \
    https://raw.githubusercontent.com/zju3dv/LoFTR/master/assets/scannet_test_1500/test.npz

if [ ! -d "$ROOT/spair/SPair-71k" ]; then
    curl -L --fail --output "$ROOT/spair/SPair-71k.tar.gz" \
        https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
    tar -xzf "$ROOT/spair/SPair-71k.tar.gz" -C "$ROOT/spair"
fi
