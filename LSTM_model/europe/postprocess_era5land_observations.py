"""Evaluate monthly standardized ensemble anomalies on the native ERA5-Land grid."""
import argparse
from collections import defaultdict
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

PROJECT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT))
from LSTM_model.utils.observation_evaluation import (
    MEMBERS, METRICS, country_borders, figures, metrics, rank_histogram, save, spatial_axes,
)
from LSTM_model.model.config import INPUTPATH
from LSTM_model.model.predict_era5land_observations import (
    NATIVE_EUROPE_GRID, NATIVE_EUROPE_OUTPUT, validate_prepared_inputs,
)

PREP = Path(INPUTPATH) / "era5land_arco_1996_2012" / "prepared"
EVALUATION = NATIVE_EUROPE_OUTPUT.parent / "observation_anomalies"
EU_ZOOM = (-6.5, 43.85, 19.5, 56.25)
EU_SCATTER_LIMITS = {"NSE": -1, "KGE": -1.41}


def read_observations(meta, pixels, months):
    source = Path(meta["observation_file"]).resolve()
    with Dataset(source) as ds:
        time = ds["time"]
        dates = num2date(time[:], time.units, time.calendar)
        periods = pd.PeriodIndex.from_fields(year=[date.year for date in dates],
                                             month=[date.month for date in dates], freq="M")
        if periods.has_duplicates:
            raise ValueError("Duplicate observation months")
        positions = periods.get_indexer(months)
        if np.any(positions < 0) or np.any(np.diff(positions) != 1):
            raise ValueError("Expected consecutive observation months for the evaluation")
        field = ds["wtd_a"]
        if (field.dimensions != ("time", "latitude", "longitude")
                or list(field.shape[1:]) != meta["grid_shape"]):
            raise ValueError("Observation and prediction ERA5-Land grids disagree")
        y, x = pixels.T
        latitude = ds["latitude"][:]
        longitude = ds["longitude"][:]
        if (not np.allclose(np.diff(latitude), -0.1, rtol=0, atol=1e-7)
                or not np.allclose(np.diff(longitude), 0.1, rtol=0, atol=1e-7)):
            raise ValueError("Expected native ERA5-Land latitude and longitude axes")

        cells = pd.DataFrame(dict(model_y=y, model_x=x, longitude=longitude[x], latitude=latitude[y]))
        well_index = ds["well_index"][:].filled(-1)[y, x]
        if np.any(well_index < 0) or len(np.unique(well_index)) != len(pixels):
            raise ValueError("Each prediction cell must have one distinct selected well")
        wells = pd.DataFrame(dict(longitude=ds["well_longitude"][:][well_index],
                                  latitude=ds["well_latitude"][:][well_index]))
        # Read only the observation tiles needed for the predicted cells.
        _, tile_y, tile_x = field.chunking()
        tiles = defaultdict(list)
        for index, tile in enumerate(zip(y // tile_y, x // tile_x)):
            tiles[tile].append(index)
        obs = np.empty((len(months), len(pixels)))
        for (tile_row, tile_col), columns in tiles.items():
            y0, x0 = tile_row * tile_y, tile_col * tile_x
            block = field[positions[0]:positions[-1] + 1, y0:y0 + tile_y, x0:x0 + tile_x]
            obs[:, columns] = block[:, y[columns] - y0, x[columns] - x0].filled(np.nan)
    if not np.isfinite(obs).all():
        raise ValueError("Prepared observation cells must have complete monthly anomalies")
    return obs, cells, wells


def aggregate(prepared, predictions, grid_shape, dates, starts, counts, ncell):
    dims = (len(starts), ncell)
    total_raw = np.zeros(dims)
    mean = np.zeros(dims)
    m2 = np.zeros(dims)
    valid = np.ones(dims, dtype=bool)
    members = np.empty((ncell, MEMBERS, len(starts)), dtype=np.float32)
    for member in range(MEMBERS):
        path = predictions / f"member_{member:03d}" / "daily_wtd.npy"
        info = json.loads(path.with_suffix(".json").read_text())
        daily = np.load(path, mmap_mode="r", allow_pickle=False)
        if (info["member"] != member or daily.shape != (len(dates), ncell)
                or list(daily.shape) != info["shape"]
                or info.get("grid") != NATIVE_EUROPE_GRID or info.get("grid_shape") != grid_shape
                or info.get("quantity") != "water_table_depth"
                or info["date_start"] != str(dates[0].date())
                or info["date_end"] != str(dates[-1].date())
                or Path(info["target_dates_file"]).resolve() != prepared / "target_dates.npy"
                or Path(info["selected_pixels_file"]).resolve() != prepared / "selected_pixels.npy"):
            raise ValueError(f"Inconsistent prediction metadata: {path}")
        monthly = np.add.reduceat(daily, starts, axis=0, dtype=float) / counts[:, None]
        if not np.isfinite(monthly).all():
            raise ValueError(f"Non-finite prediction: {path}")
        total_raw += monthly
        years = monthly.reshape(-1, 12, ncell)
        scale = years.std(axis=0)
        varying = years.max(axis=0) > years.min(axis=0)
        valid.reshape(-1, 12, ncell)[:] &= varying
        anomaly = ((years - years.mean(axis=0)) / np.where(varying, scale, 1)).reshape(dims)
        members[:, member, :] = np.where(np.tile(varying, (len(years), 1)), anomaly, np.nan).T
        delta = anomaly - mean
        mean += delta / (member + 1)
        m2 += delta * (anomaly - mean)
        if (member + 1) % 20 == 0:
            print(f"Aggregated {member + 1}/{MEMBERS} members", flush=True)
    mean[~valid] = np.nan
    m2[~valid] = np.nan
    return total_raw / MEMBERS, mean, np.sqrt(np.maximum(m2, 0) / (MEMBERS - 1)), members


def observation_locations(out, wells, cells):
    longitude = np.concatenate([wells.longitude, cells.longitude])
    latitude = np.concatenate([wells.latitude, cells.latitude])
    extent = (longitude.min()-2, latitude.min()-1, longitude.max()+2, latitude.max()+1)
    borders = country_borders(box(*extent))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    for ax, points, title in [
        (axes[0], wells, f"Selected wells ({len(wells):,})"),
        (axes[1], cells, f"ERA5-Land grid cells ({len(cells):,})"),
    ]:
        spatial_axes(ax, borders, extent)
        ax.scatter(points.longitude, points.latitude, s=5, alpha=.65, linewidths=0, color="#165e91")
        ax.set(title=title, xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    save(fig, out / "well_locations_and_grid_cells.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-directory", type=Path, default=PREP)
    parser.add_argument("--predictions-directory", type=Path, default=NATIVE_EUROPE_OUTPUT)
    parser.add_argument("--output", type=Path, help="New results directory within global-TL")
    parser.add_argument("--plot-cells", type=int, nargs="+", help="Render individual PNGs only for these cell IDs (default: all). Summary outputs always use all cells.")
    args = parser.parse_args()
    prepared = args.prepared_directory.resolve()
    predictions = args.predictions_directory.resolve()
    out = (args.output or EVALUATION / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    if not out.is_relative_to(PROJECT.parent):
        parser.error("output must be within global_TL")
    if out.exists():
        parser.error(f"output already exists: {out}")
    meta = json.loads((prepared / "preprocessing.json").read_text())
    if meta.get("grid") != NATIVE_EUROPE_GRID:
        raise ValueError("Expected native European ERA5-Land preparation metadata")
    validate_prepared_inputs(prepared)
    pixels = np.load(prepared / "selected_pixels.npy", allow_pickle=False)
    indices = list(range(len(pixels))) if args.plot_cells is None else sorted(set(args.plot_cells))
    if min(indices) < 0 or max(indices) >= len(pixels):
        parser.error("plot cell IDs must index selected_pixels.npy")
    dates = pd.DatetimeIndex(np.load(prepared / "target_dates.npy", allow_pickle=False))
    _, starts, counts = np.unique(dates.to_period("M").asi8, return_index=True, return_counts=True)
    months = dates[starts].to_period("M")
    obs, cells, wells = read_observations(meta, pixels, months)
    static = np.load(prepared / "static_features.npy", allow_pickle=False)
    np.testing.assert_array_equal(static[-2:].T, cells[["longitude", "latitude"]].to_numpy(np.float32))
    out.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as workspace:
        scratch = Path(workspace)
        print(f"Reading {MEMBERS} members for {len(pixels)} cells", flush=True)
        raw, pred, spread, members = aggregate(prepared, predictions, meta["grid_shape"],
                                               dates, starts, counts, len(pixels))
        cells = cells.join(metrics(obs, pred, spread, members))
        if not cells.paired_months.sum():
            raise ValueError("No matched finite anomalies")
        result = scratch / "results"
        result.mkdir()
        cells.index.name = "cell"
        cells.to_csv(result / "cell_metrics.csv")
        cells[METRICS].describe().T.to_csv(result / "metric_summary.csv", index_label="metric")
        ds = xr.Dataset(coords={"time": months.to_timestamp(), "cell": cells.index,
                                **{k: ("cell", cells[k]) for k in cells.columns[:4]}})
        for key, data in [("monthly_prediction", raw), ("ensemble_anomaly", pred),
                          ("observed_anomaly", obs), ("ensemble_anomaly_sd", spread)]:
            ds[key] = (("time", "cell"), data.astype("float32"))
        ds["prediction_days"] = ("time", counts.astype("int32"))
        ds["latitude"].attrs.update(standard_name="latitude", units="degrees_north")
        ds["longitude"].attrs.update(standard_name="longitude", units="degrees_east")
        ds.attrs.update(members=MEMBERS, observation_file=meta["observation_file"],
                        grid=NATIVE_EUROPE_GRID, predictions_directory=str(predictions),
                        anomaly_baseline="Member-specific calendar-month z-score, 1996–2012, ddof=0")
        ds.to_netcdf(result / "monthly_series.nc", engine="scipy")
        observation_locations(result, wells, cells)
        rank_histogram(result, obs, members)
        extent = (cells.longitude.min()-2, cells.latitude.min()-1,
                  cells.longitude.max()+2, cells.latitude.max()+1)
        location_map = (country_borders(box(*extent)), extent)
        print(f"Rendering individual figures for {len(indices):,} cells", flush=True)
        figures(result, cells, months, obs, pred, members, indices,
                map_options={"zoom": EU_ZOOM, "zoom_title": "England and central Europe"},
                scatter_minimums=EU_SCATTER_LIMITS, location_map=location_map,
                value_label="WTD anomalies", series_label="WTD anomalies")
        result.rename(out)
    print(f"Saved {len(cells)} cells and {int(cells.paired_months.sum())} paired cell-months to {out}")


if __name__ == "__main__":
    main()
