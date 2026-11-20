#!/bin/bash

# Usage: ./check_modification_year.sh /path/to/parent "pattern"
# Example: ./check_modification_year.sh /data/runs "*.nc"

PARENT_DIR="/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/validation_ERA5/ensemble_400px"
FILE_PATTERN="sim_*.npy"
TARGET_DATE="2026-01-28"

if [[ -z "$PARENT_DIR" || -z "$FILE_PATTERN" ]]; then
    echo "Usage: $0 <parent_directory> <file_pattern>"
    exit 1
fi

for dir in "$PARENT_DIR"/*/; do
    [[ -d "$dir" ]] || continue

    found=false

    for file in "$dir"/$FILE_PATTERN; do
        # Handle case where glob does not match
        [[ -e "$file" ]] || continue
        found=true

        mod_date=$(date -r "$file" +%Y-%m-%d)

        # Check if modification date is less than target date
        if [[ "$mod_date" < "$TARGET_DATE" ]]; then
            echo "OUTDATED: $file (last modified in $mod_date)"
            echo "$file" >> outdated_files.log
        fi
    done

    if [[ "$found" = false ]]; then
        echo "MISSING: No files matching '$FILE_PATTERN' in $dir"
    fi
done

echo "Modification date check completed."