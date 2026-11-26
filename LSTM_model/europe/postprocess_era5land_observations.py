"""Evaluate monthly standardized ERA5-Land ensemble anomalies on observation cells."""
import argparse
import json
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from LSTM_model.model.config import ENSEMBLE_PREDICTION_DIRECTORY

REGION = "EU_74_-48_69_20"
PREP = PROJECT / "inputs" / REGION / "era5land_inputs_1996_2012"
VALIDATION = PROJECT / "outputs" / REGION / "validation_ERA5"
ENSEMBLE = ENSEMBLE_PREDICTION_DIRECTORY
NAME = "sim_ERA5on_observed_valid_pixels_1996_2012.npy"
EU_ZOOM = (-6.5, 43.85, 19.5, 56.25)
EU_SCATTER_LIMITS = {"NSE": -1, "KGE": -1.41}


def read_observations(meta, pixels, months, scratch):
    # The supplied module environment has the netCDF C tools and SciPy backend.
    source = Path(meta["observation_file"]).resolve()
    if not source.is_relative_to(PROJECT.parent):
        raise ValueError(f"Observation file outside workspace: {source}")
    classic = scratch / "observations.nc"
    subprocess.run(["nccopy", "-k", "2", str(source), str(classic)], check=True)
    with xr.open_dataset(classic, engine="scipy") as ds:
        field = ds[meta["observation_variable"]].transpose("time", "lat", "lon")
        offset = np.array(meta["observation_grid_center_offset"])
        if not np.array_equal(np.array(field.shape[1:]) + 2 * offset, meta["grid_shape"]):
            raise ValueError("Observation and model grid shapes disagree with center offset")
        indices = pixels - offset
        if np.any(indices < 0) or np.any(indices >= field.shape[1:]):
            raise ValueError("Selected pixel outside observation grid")
        times = ds.indexes["time"].to_period("M")
        if times.has_duplicates or not months.isin(times).all():
            raise ValueError("Missing or duplicate observation month")
        obs = field.values[times.get_indexer(months)[:, None], indices[:, 0], indices[:, 1]]
        return np.where(np.isfinite(obs), obs, np.nan).astype(float)


def aggregate(dates, starts, counts, ncell):
    dims = (len(starts), ncell)
    total_raw = np.zeros(dims)
    mean = np.zeros(dims)
    m2 = np.zeros(dims)
    valid = np.ones(dims, dtype=bool)
    members = np.empty((ncell, MEMBERS, len(starts)), dtype=np.float32)
    # Existing EU sidecars predate the rename to global-LSTM.
    legacy_prep = PROJECT.with_name("spatio-temporal-LSTM") / PREP.relative_to(PROJECT)
    for member in range(MEMBERS):
        path = ENSEMBLE / f"400px_member_{member}" / NAME
        info = json.loads(path.with_suffix(".json").read_text())
        daily = np.load(path, mmap_mode="r")
        if (info["member"] != member or daily.shape != (len(dates), ncell)
                or list(daily.shape) != info["shape"]
                or Path(info["target_dates_file"]).resolve() not in
                   (PREP / "target_dates.npy", legacy_prep / "target_dates.npy")
                or Path(info["selected_pixels_file"]).resolve() not in
                   (PREP / "selected_pixels.npy", legacy_prep / "selected_pixels.npy")):
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
    mean[~valid] = np.nan
    m2[~valid] = np.nan
    return total_raw / MEMBERS, mean, np.sqrt(np.maximum(m2, 0) / (MEMBERS - 1)), members


def observation_locations(out, meta, cells):
    source = Path(meta["observation_file"])
    mapping = pd.read_csv(source.with_name(source.stem + "_station_mapping.csv"))
    mapped = mapping.drop_duplicates(["tsmp_yindex", "tsmp_xindex"])
    selected = cells.set_index(["model_y", "model_x"])
    assigned = selected.reindex(pd.MultiIndex.from_frame(mapped[["tsmp_yindex", "tsmp_xindex"]]))
    if assigned[["longitude", "latitude"]].isna().any().any():
        raise ValueError("A mapped observation well is assigned outside the EU evaluation grid")

    longitude = np.concatenate([mapping.lon, assigned.longitude])
    latitude = np.concatenate([mapping.lat, assigned.latitude])
    extent = (longitude.min()-2, latitude.min()-1, longitude.max()+2, latitude.max()+1)
    borders = country_borders(box(*extent))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    for ax, x, y, title in [
        (axes[0], mapping.lon, mapping.lat, f"Valid wells ({len(mapping):,})"),
        (axes[1], assigned.longitude, assigned.latitude, f"Assigned grid cells ({len(assigned):,})"),
    ]:
        spatial_axes(ax, borders, extent)
        ax.scatter(x, y, s=5, alpha=.65, linewidths=0, color="#165e91")
        ax.set(title=title, xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    save(fig, out / "well_locations_and_grid_cells.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New results directory within global_TL")
    parser.add_argument("--plot-cells", type=int, nargs="+", help="Render individual PNGs only for these cell IDs (default: all). Summary outputs always use all cells.")
    args = parser.parse_args()
    out = (args.output or VALIDATION / "observation_anomalies" /
           datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    if not out.is_relative_to(PROJECT.parent):
        parser.error("output must be within global_TL")
    if out.exists():
        parser.error(f"output already exists: {out}")
    meta = json.loads((PREP / "preprocessing.json").read_text())
    pixels = np.load(PREP / "selected_pixels.npy")
    if pixels.shape != (meta["selected_pixel_count"], 2) or len(np.unique(pixels, axis=0)) != len(pixels):
        raise ValueError("Invalid selected pixel list")
    indices = list(range(len(pixels))) if args.plot_cells is None else sorted(set(args.plot_cells))
    if min(indices) < 0 or max(indices) >= len(pixels):
        parser.error("plot cell IDs must index selected_pixels.npy")
    dates = pd.DatetimeIndex(np.load(PREP / "target_dates.npy"))
    expected = pd.date_range("1996-01-01", "2012-12-31")
    if not np.array_equal(dates.values, expected[~((expected.month == 2) & (expected.day == 29))].values):
        raise ValueError("Expected consecutive 1996–2012 dates excluding February 29")
    _, starts, counts = np.unique(dates.to_period("M").asi8, return_index=True, return_counts=True)
    months = dates[starts].to_period("M")
    static = np.load(PREP / "static_features.npy")
    keys = meta["feature_files"][len(meta["dynamic_variables"]):]
    cells = pd.DataFrame(dict(model_y=pixels[:, 0], model_x=pixels[:, 1],
                              longitude=static[keys.index("lon2D_ts.npy")],
                              latitude=static[keys.index("lat2D_ts.npy")]))
    out.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as workspace:
        scratch = Path(workspace)
        obs = read_observations(meta, pixels, months, scratch)
        print(f"Reading {MEMBERS} members for {len(pixels)} cells", flush=True)
        raw, pred, spread, members = aggregate(dates, starts, counts, len(pixels))
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
        ds.attrs.update(members=MEMBERS, observation_file=meta["observation_file"],
                        anomaly_baseline="Member-specific calendar-month z-score, 1996–2012, ddof=0")
        ds.to_netcdf(result / "monthly_series.nc", engine="scipy")
        observation_locations(result, meta, cells)
        rank_histogram(result, obs, members)
        extent = (cells.longitude.min()-2, cells.latitude.min()-1,
                  cells.longitude.max()+2, cells.latitude.max()+1)
        location_map = (country_borders(box(*extent)), extent)
        figures(result, cells, months, obs, pred, members, indices,
                map_options={"zoom": EU_ZOOM, "zoom_title": "England and central Europe"},
                scatter_minimums=EU_SCATTER_LIMITS, location_map=location_map,
                value_label="WTD anomalies", series_label="WTD anomalies")
        result.rename(out)
    print(f"Saved {len(cells)} cells and {int(cells.paired_months.sum())} paired cell-months to {out}")


if __name__ == "__main__":
    main()
