#!/usr/bin/env bash

set -euo pipefail

INPUT="${1:-}"

if [ -z "$INPUT" ]; then
    echo "Usage:"
    echo "./remove_bg_creature.sh <directory>"
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

    echo "[CREATURE] $file"

    rembg i \
        -m isnet-general-use \
        "$file" \
        "$out"

done

echo
echo "Creature processing complete."