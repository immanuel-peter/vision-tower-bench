#!/usr/bin/env bash
set -euo pipefail
dest="${1:-data}"
mkdir -p "$dest"
if [ -d "$dest/val2017" ]; then
  echo "already present: $dest/val2017 ($(find "$dest/val2017" -name '*.jpg' | wc -l | tr -d ' ') images)"
  exit 0
fi
curl -L --progress-bar -o "$dest/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
unzip -q "$dest/val2017.zip" -d "$dest"
rm "$dest/val2017.zip"
find "$dest/val2017" -name '*.jpg' | wc -l
