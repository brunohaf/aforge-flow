#!/usr/bin/env bash

set -euo pipefail

INPUT="${1:-}"

if [ -z "$INPUT" ]; then
    echo "Usage:"
    echo "./to_webmp.sh <directory>"
    exit 1
fi

find "$INPUT" -type f -print0 |
while IFS= read -r -d '' file
do

    ext="${file##*.}"
    ext=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

    case "$ext" in

        png|jpg|jpeg)

            output="${file%.*}.webp"

            if [ -f "$output" ]; then
                continue
            fi

            echo "[WEBP] $file"

            if [[ "$file" == *.nobg.png ]]; then

                cwebp \
                    -lossless \
                    -mt \
                    "$file" \
                    -o "$output" \
                    >/dev/null

            else

                cwebp \
                    -q 92 \
                    -m 6 \
                    -mt \
                    -af \
                    "$file" \
                    -o "$output" \
                    >/dev/null

            fi
            ;;

        mp4|mov|mkv|avi)

            output="${file%.*}.webm"

            if [ -f "$output" ]; then
                continue
            fi

            echo "[WEBM] $file"

            ffmpeg \
                -loglevel error \
                -y \
                -i "$file" \
                -c:v libvpx-vp9 \
                -crf 28 \
                -b:v 0 \
                -row-mt 1 \
                -tile-columns 2 \
                -frame-parallel 1 \
                -threads 0 \
                -c:a libopus \
                "$output"
            ;;

    esac

done

echo
echo "WebP/WebM conversion complete."