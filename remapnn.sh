#!/bin/bash

cd xxxx
source xxxx

# Directory containing the NetCDF files
INPUT_DIR="xxxx"
GRID_FILE="xxxx"

# Loop over all .nc files in the directory
for file in "$INPUT_DIR"/daily_mean_d2m_t2m_swvl3/*.nc; do
    filename=$(basename "$file")

    # Only process files from 1996-1999 or 2016
    if [[ "$filename" =~ (199[5-9]|2016) ]]; then
        # Check if the filename contains t2m, d2m, or swvl3
        if [[ "$filename" == *"t2m"* || "$filename" == *"d2m"* || "$filename" == *"swvl3"* ]]; then
            echo "Processing (nearest neighbor): $filename"
            cdo remapnn,"$GRID_FILE" "$file" "${INPUT_DIR}/regridded_to_EUR11/BonA_nn_${filename}"
        else
            echo "Skipping file (no variable match): $filename"
        fi
    else
        echo "Skipping file (year not in 1996-1999 or 2016): $filename"
    fi
done

echo "✅ All processing completed."
