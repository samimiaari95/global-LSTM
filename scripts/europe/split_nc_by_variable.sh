#!/bin/bash

REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"

# Directory containing NetCDF files
DIR="${REPOSITORY}/inputs/EU_74_-48_69_20/raw/ERA5-Land"

# List of variables to extract
VARS=("t2m" "d2m" "swvl3")

if [[ ! -d "$DIR" ]]; then
    echo "NetCDF input directory is missing: $DIR"
    exit 1
fi

for file in "$DIR"/*.nc; do
    echo "Processing $file"
    [[ -f "$file" ]] || continue

    base=$(basename "$file" .nc)
    base="${base/_daily_mean/}"
    # cdo daymean "$file" "${DIR}/${base}_daily_mean.nc"
    for var in "${VARS[@]}"; do
        echo "  Extracting variable: $var"
        newbase="${base/t2m_d2m_swvl3/$var}"
        out="${DIR}/${newbase}.nc"
        # ncks -v "$var" "${DIR}/${base}_daily_mean.nc" "$out"        
        cdo selname,"$var" "${DIR}/${base}_daily_mean.nc" "$out"
    done
done
