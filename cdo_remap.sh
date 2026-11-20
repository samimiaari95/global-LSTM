#!/bin/bash


cd xxxx
source xxxxx

# Directory containing the NetCDF files
INPUT_DIR="xxxxx"
GRID_FILE="xxxxx"
WEIGHTS_NN="xxxxx"

# Loop over all .nc files in the directory
for file in "$INPUT_DIR"/*.nc; do
    filename=$(basename "$file")
    
    # Check if the filename contains t2m, d2m, or swvl3
    if [[ "$filename" == *"t2m"* || "$filename" == *"d2m"* || "$filename" == *"swvl3"* ]]; then
        echo "Processing (nearest neighbor): $filename"
        cdo remap,"$GRID_FILE","$WEIGHTS_NN" "$file" "${INPUT_DIR}/regridded_to_EUR11/BonA_nn_${filename}"

    # Check if the filename contains tp
    elif [[ "$filename" == *"tp"* ]]; then
        echo "Processing (conservative): $filename"
        cdo remapcon,"$GRID_FILE" "$file" "${INPUT_DIR}/regridded_to_EUR11/BonA_conservative_${filename}"

    # If none match, skip the file
    else
        echo "Skipping file (no match): $filename"
    fi
done

echo "✅ All processing completed."
