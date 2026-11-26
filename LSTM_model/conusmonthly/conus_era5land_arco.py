"""Retrieve selected cells from the CDS ERA5-Land hourly Zarr archive."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from LSTM_model.conusmonthly.conus_inputs import forcing_area, stage_calendar
from LSTM_model.utils.file_io import save_json
from LSTM_model.conusmonthly.conus_observations import MONTHLY_GRID


from LSTM_model.utils.era5land_arco import (
    GROUPS, arco_headers, collect_variable, grid_indices, hourly_slice, open_arco_group,
)


def download(root, workers, keep_leap_days=False):
    import numcodecs

    dates, _ = stage_calendar(root)
    if keep_leap_days:
        end = json.loads((root / "selection.json").read_text())["end"]
        dates = pd.date_range(dates[0], end)
    pixels = pd.read_csv(root / "static_pixels.csv")
    cells = pixels[["y", "x"]].to_numpy(dtype=np.int32)
    identity = hashlib.sha256(dates.values.tobytes() + cells.tobytes()).hexdigest()[:16]
    cache = root / "era5land" / f"arco_tiles_{identity}"
    cache.mkdir(parents=True, exist_ok=True)
    full_days = pd.date_range(dates[0], dates[-1])
    selected = full_days.isin(dates)
    target = root / "era5land" / "arco_daily_points.npz"
    headers = arco_headers()
    numcodecs.blosc.set_nthreads(1)
    result = {}
    for server, group_name, variables in GROUPS:
        group = open_arco_group(server, group_name, headers)
        rows, cols = grid_indices(group, pixels)
        hours = hourly_slice(group, full_days)
        for variable in variables:
            result[variable] = collect_variable(group[variable], hours, selected, rows, cols,
                                                cache, workers)
    # Preserve sampled raw predictors and the exact order required to stage them.
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, context_dates=dates.to_numpy(dtype="datetime64[D]"),
                            selected_pixels=cells, **result)
    temporary.replace(target)
    save_json({
        "source": "ECMWF ERA5-Land ARCO",
        "start": str(dates[0].date()), "end": str(dates[-1].date()), "days": len(dates),
        "cells": len(cells), "coverage_area_nwse": forcing_area(pixels),
        "aggregation": {"tp": "01–23 UTC accumulation in metres",
                        "t2m": "24-hour UTC mean in kelvin",
                        "d2m": "24-hour UTC mean in kelvin",
                        "swvl3": "24-hour UTC mean in m³/m³"},
        "complete_cells_by_variable": {
            variable: int(np.isfinite(field).all(axis=0).sum()) for variable, field in result.items()
        },
    }, target.with_suffix(".json"))
    print(f"Saved daily ERA5-Land samples to {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=MONTHLY_GRID, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--keep-leap-days", action="store_true",
                        help="Keep every calendar date through the selected end, including February 29")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    download(args.root, args.workers, args.keep_leap_days)


if __name__ == "__main__":
    main()
