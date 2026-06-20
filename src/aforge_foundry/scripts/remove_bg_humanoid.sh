#!/usr/bin/env bash

set -euo pipefail

INPUT="${1:-}"

if [ -z "$INPUT" ]; then
    echo "Usage:"
    echo "./remove_bg_humanoid.sh <directory>"
    exit 1
fi

find "$INPUT" \
-type f \
\( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) \
-print0 |
while IFS= read -r -d '' file
do

    if [[ "$file" == *.nobg.png ]]; then
        continue
    fi

    out="${file%.*}.nobg.png"

    if [ -f "$out" ]; then
        continue
    fi

    echo "[HUMANOID] $file"

    rembg i \
        -m u2net_human_seg \
        "$file" \
        "$out"

done

echo
echo "Humanoid processing complete."