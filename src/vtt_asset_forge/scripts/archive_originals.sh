#!/bin/bash

ROOT="$1"
ARCHIVE="$2"

find "$ROOT" -type f | while read file
do

    ext=$(echo "${file##*.}" | tr '[:upper:]' '[:lower:]')

    if [[ "$ext" != "webp" && "$ext" != "webm" ]]
    then

        rel="${file#$ROOT/}"

        mkdir -p "$ARCHIVE/$(dirname "$rel")"

        mv "$file" "$ARCHIVE/$rel"

    fi

done
