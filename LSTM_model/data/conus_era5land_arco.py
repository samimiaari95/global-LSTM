"""Retrieve selected cells from the CDS ERA5-Land hourly Zarr archive."""

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from xarray.coding.times import decode_cf_datetime

from LSTM_model.data.conus_inputs import forcing_area, stage_calendar
from LSTM_model.data.conus_observations import MONTHLY_GRID, save_json


ARCO = "https://arco.datastores.ecmwf.int"
GROUPS = (
    ("007", "sfc-2m-temperature", ("t2m", "d2m")),
    ("005", "sfc-soil-water", ("swvl3",)),
    ("009", "sfc-pressure-precipitation", ("tp",)),
)


def arco_headers():
    import cdsapi

    return {"Authorization": f"Bearer {cdsapi.Client(quiet=True).key}"}


def open_arco_group(server, group_name, headers):
    import fsspec
    import zarr

    url = f"{ARCO}/cadl-arco-geo-{server}/arco/reanalysis_era5_land/{group_name}/geoChunked.zarr"
    mapper = fsspec.get_mapper(url, headers=headers, client_kwargs={"trust_env": True})
    return zarr.open_consolidated(mapper, mode="r")


def grid_indices(group, pixels):
    latitude, longitude = group["latitude"][:], group["longitude"][:]
    lon = pixels.longitude.to_numpy()
    if longitude.min() >= 0:
        lon = lon % 360
    rows = pd.Index(latitude).get_indexer(
        pixels.latitude.to_numpy(), method="nearest", tolerance=.051)
    cols = pd.Index(longitude).get_indexer(lon, method="nearest", tolerance=.051)
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("Requested locations do not match the native ERA5-Land grid")
    return rows, cols


def hourly_slice(group, day_dates):
    time = group["time"]
    attrs = time.attrs
    units, calendar = attrs["units"], attrs.get("calendar", "standard")
    first = pd.Timestamp(decode_cf_datetime(time[:1], units, calendar)[0])
    start = int((day_dates[0] - first) / pd.Timedelta(hours=1))
    stop = start + len(day_dates) * 24
    if start < 0 or stop > time.shape[0]:
        raise ValueError("Requested dates exceed hourly ERA5-Land archive")
    actual = pd.DatetimeIndex(decode_cf_datetime(time[start:stop], units, calendar))
    expected = pd.date_range(day_dates[0], periods=len(day_dates) * 24, freq="h")
    if not actual.equals(expected):
        raise ValueError("Unexpected hourly ERA5-Land calendar")
    return slice(start, stop)


def collect_variable(array, hours, selected, rows, cols, cache, workers):
    chunk_y, chunk_x = array.chunks[1:]
    tiles = defaultdict(list)
    for i, (row, col) in enumerate(zip(rows, cols)):
        tiles[row // chunk_y, col // chunk_x].append(i)
    day_count = (hours.stop - hours.start) // 24
    output = np.empty((selected.sum(), len(rows)), dtype=np.float32)
    name = array.basename
    print(f"Sampling {name} across {len(tiles)} ERA5-Land spatial chunks", flush=True)

    def collect(tile):
        (tile_y, tile_x), indices = tile
        path = cache / f"{name}_{tile_y}_{tile_x}.npy"
        if path.is_file():
            return indices, np.load(path, allow_pickle=False)
        y0, x0 = tile_y * chunk_y, tile_x * chunk_x
        data = array[hours, y0:y0 + chunk_y, x0:x0 + chunk_x]
        hourly = data[:, rows[indices] - y0, cols[indices] - x0]
        daily = hourly.reshape(day_count, 24, len(indices))
        if name == "tp":
            # ARCO stores de-accumulated precipitation for the hour ending at
            # each timestamp. Sum 01:00 through 23:00 to match the CDS 23Z field.
            values = daily[:, 1:, :].sum(axis=1, dtype=np.float64)
        else:
            values = daily.mean(axis=1, dtype=np.float64)
        values = values[selected].astype(np.float32)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.save(stream, values)
        temporary.replace(path)
        return indices, values

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (indices, values) in enumerate(pool.map(collect, tiles.items()), 1):
            output[:, indices] = values
            if done == len(tiles) or done % 50 == 0:
                print(f"Sampled {name} from {done}/{len(tiles)} tiles", flush=True)
    return output


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
