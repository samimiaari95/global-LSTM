"""Prepare native European ERA5-Land predictors for observation-cell inference."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from scipy.spatial import cKDTree

from LSTM_model.utils.era5land_arco import (
    ARCO, GROUPS, arco_headers, collect_variable, grid_indices, hourly_slice, open_arco_group,
    stage_daily_predictors,
)
from LSTM_model.utils.file_io import save_csv, save_json
from LSTM_model.europe.map_era5land_observations import EARTH_RADIUS_KM, INPUTS, OUTPUT as OBSERVATIONS
from LSTM_model.model.config import FEATURES_FILES, LOOKBACK

DEFAULT_ROOT = INPUTS / "era5land_arco_1996_2012"
FULL_DAYS = pd.date_range("1995-01-01", "2012-12-31")
MODEL_DAYS = FULL_DAYS[~((FULL_DAYS.month == 2) & (FULL_DAYS.day == 29))]
TARGET_DAYS = MODEL_DAYS[LOOKBACK:]
RAW_FIELDS = ("tp", "t2m", "d2m", "swvl3")
STATIC_FIELDS = ("slopex", "slopey", "soilind", "longitude", "latitude")
TP_START_HOUR = 1
TP_END_HOUR = 23


def spherical_xyz(latitude, longitude):
    lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def complete_observation_cells(path):
    with Dataset(path) as ds:
        time = ds["time"]
        dates = num2date(time[:], time.units, time.calendar)
        months = [(t.year, t.month) for t in dates]
        expected = [(t.year, t.month) for t in pd.period_range("1996-01", "2012-12", freq="M")]
        start = months.index(expected[0])
        end = start + len(expected)
        if months[start:end] != expected:
            raise ValueError("Observation NetCDF must contain all months from 1996 through 2012.")
        y, x = np.where(ds["well_index"][:].filled(-1) >= 0)
        field = ds["wtd_a"]
        _, dy, dx = field.chunking()
        complete = np.zeros(len(y), dtype=bool)
        tiles = {}
        for i, (ty, tx) in enumerate(zip(y // dy, x // dx)):
            tiles.setdefault((ty, tx), []).append(i)
        for (ty, tx), indices in tiles.items():
            y0, x0 = ty * dy, tx * dx
            values = field[start:end, y0:y0 + dy, x0:x0 + dx].filled(np.nan)
            complete[indices] = np.isfinite(values[:, y[indices] - y0, x[indices] - x0]).all(axis=0)
        y, x = y[complete], x[complete]
        if not len(y):
            raise ValueError("No observation cells have 204 finite monthly anomalies.")
        lat, lon = ds["latitude"][:], ds["longitude"][:]
        points = pd.DataFrame({"y": y, "x": x, "latitude": lat[y], "longitude": lon[x]})
        return points, (len(lat), len(lon))


def static_plane(path, shape):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    # The training inputs repeat each static raster across all days.
    if values.ndim == 3:
        values = values[0]
    if values.shape != shape:
        raise ValueError(f"{path} has shape {values.shape}, expected {shape}")
    return values


def select_points(root, observation_file, static_dir):
    points, shape = complete_observation_cells(observation_file)
    lon = np.load(static_dir / "lon2D.npy", allow_pickle=False)
    lat = np.load(static_dir / "lat2D.npy", allow_pickle=False)
    if lat.ndim != 2 or lat.shape != lon.shape:
        raise ValueError("Static donor coordinates must have matching two-dimensional shapes.")
    valid = np.isfinite(lat) & np.isfinite(lon)
    source_y, source_x = np.where(valid)
    tree = cKDTree(spherical_xyz(lat[valid].astype(float), lon[valid].astype(float)))
    distances, donors = tree.query(spherical_xyz(points.latitude, points.longitude))
    y, x = source_y[donors], source_x[donors]
    points["static_source_y"], points["static_source_x"] = y, x
    points["static_distance_km"] = 2 * EARTH_RADIUS_KM * np.arcsin(distances / 2)
    for name in STATIC_FIELDS[:3]:
        points[name] = static_plane(static_dir / f"{name}.npy", lat.shape)[y, x]
    if not np.isfinite(points[list(STATIC_FIELDS)].to_numpy()).all():
        raise ValueError("At least one nearest static donor has a missing feature.")
    root.mkdir(parents=True, exist_ok=True)
    save_csv(points, root / "observation_pixels.csv", index=False)
    save_json({
        "observation_file": str(observation_file.resolve()),
        "static_directory": str(static_dir.resolve()),
        "grid_shape": list(shape),
        "observation_cells": len(points),
    }, root / "selection.json")
    print(f"Selected {len(points):,} complete observation cells and sampled their static predictors.",
          flush=True)
    return points


def read_points(root):
    return pd.read_csv(root / "observation_pixels.csv", float_precision="round_trip")


def download(root, points, workers):
    import numcodecs

    numcodecs.blosc.set_nthreads(1)
    date_values = FULL_DAYS.to_numpy(dtype="datetime64[D]")
    cells = points[["y", "x"]].to_numpy(dtype=np.int32)
    coords = points[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    identity = hashlib.sha256(
        date_values.tobytes() + cells.tobytes() + coords.tobytes()
        + f"{ARCO}|daily-v1|tp-{TP_START_HOUR}-{TP_END_HOUR}".encode()
    ).hexdigest()[:16]
    cache = root / "arco" / f"tiles_{identity}"
    cache.mkdir(parents=True, exist_ok=True)
    target = root / "arco" / "daily_points.npz"
    if target.is_file():
        with np.load(target, allow_pickle=False) as raw:
            if not np.array_equal(raw["identity"], identity):
                raise ValueError("Published ARCO data belong to a different selection. Use a new --root.")
        print(f"Using the fresh-download artifact for this workflow at {target}", flush=True)
        return

    headers = arco_headers()
    fields = {}
    for server, name, variables in GROUPS:
        group = open_arco_group(server, name, headers)
        rows, cols = grid_indices(group, points)
        lat, lon = group["latitude"][:][rows], group["longitude"][:][cols]
        lon_difference = (lon - coords[:, 1] + 180) % 360 - 180
        if (np.max(np.abs(lat - coords[:, 0])) > 1e-4
                or np.max(np.abs(lon_difference)) > 1e-4
                or len(set(zip(rows, cols))) != len(points)):
            raise ValueError("Observation and ARCO grid centers do not align.")
        hours = hourly_slice(group, FULL_DAYS)
        for variable in variables:
            fields[variable] = collect_variable(
                group[variable], hours, np.ones(len(FULL_DAYS), dtype=bool),
                rows, cols, cache, workers,
            )
    with TemporaryDirectory(dir=target.parent) as directory:
        temporary = Path(directory) / target.name
        np.savez_compressed(temporary, dates=date_values, pixels=cells, coordinates=coords,
                            identity=identity, **fields)
        target.hardlink_to(temporary)
    print(f"Downloaded and saved daily ARCO samples to {target}", flush=True)


def prepare(root, points):
    import json

    prepared = root / "prepared"
    if prepared.exists():
        raise FileExistsError(f"{prepared} already exists. Verify it or use a new --root.")
    selection = json.loads((root / "selection.json").read_text())
    download_path = root / "arco" / "daily_points.npz"
    cells = points[["y", "x"]].to_numpy(dtype=np.int32)
    with np.load(download_path, allow_pickle=False) as raw:
        if (not np.array_equal(raw["dates"], FULL_DAYS.to_numpy(dtype="datetime64[D]"))
                or not np.array_equal(raw["pixels"], cells)
                or not np.array_equal(raw["coordinates"], points[["latitude", "longitude"]])):
            raise ValueError("ARCO dates or cells differ from the observation selection.")
        model_indices = FULL_DAYS.get_indexer(MODEL_DAYS)
        fields = {}
        for name in RAW_FIELDS:
            field = raw[name]
            if field.shape != (len(FULL_DAYS), len(points)):
                raise ValueError(f"Unexpected shape for ARCO {name}: {field.shape}")
            fields[name] = field[model_indices]
    dynamic = np.empty((3, len(MODEL_DAYS), len(points)), dtype=np.float32)
    dynamic[0] = fields["tp"] * 1000
    stage_daily_predictors(dynamic, slice(None), fields)
    del fields

    channels = ("tp", "vpd", "swvl3")
    missing = ~np.isfinite(dynamic)
    keep = ~missing.any(axis=(0, 1))
    if not keep.any():
        raise ValueError("No observed cells have complete ARCO forcing.")
    excluded = points.loc[~keep, ["y", "x", "latitude", "longitude"]].copy()
    for j, name in enumerate(channels):
        has_missing = missing[j, :, ~keep].any(axis=1)
        first_day = MODEL_DAYS[missing[j, :, ~keep].argmax(axis=1)].strftime("%Y-%m-%d")
        excluded[f"{name}_first_missing"] = np.where(has_missing, first_day, "")
    sample_shape = (int(keep.sum()) * len(TARGET_DAYS), LOOKBACK, len(FEATURES_FILES))

    with TemporaryDirectory(dir=root) as directory:
        stage = Path(directory) / "prepared"
        stage.mkdir()
        np.save(stage / "dynamic_features.npy", dynamic[:, :, keep])
        np.save(stage / "static_features.npy",
                points.loc[keep, list(STATIC_FIELDS)].to_numpy(dtype=np.float32).T)
        np.save(stage / "selected_pixels.npy", cells[keep])
        np.save(stage / "context_dates.npy", MODEL_DAYS.to_numpy(dtype="datetime64[D]"))
        np.save(stage / "target_dates.npy", TARGET_DAYS.to_numpy(dtype="datetime64[D]"))
        save_csv(points.loc[keep], stage / "selected_pixels.csv", index=False)
        save_csv(excluded, stage / "excluded_forcing_pixels.csv", index=False)
        save_json({
            **selection,
            "grid": "native ERA5-Land 0.1 degree",
            "forcing_source": ARCO,
            "forcing_file": str(download_path.resolve()),
            "feature_files": list(FEATURES_FILES),
            "feature_units": ["mm", "kPa", "m3/m3", "1", "1", "1", "degrees_east", "degrees_north"],
            "target_start_date": str(TARGET_DAYS[0].date()),
            "target_end_date": str(TARGET_DAYS[-1].date()),
            "lookback_days": LOOKBACK,
            "excluded_calendar_dates": ["02-29"],
            "selected_pixel_count": int(keep.sum()),
            "excluded_forcing_pixel_count": len(excluded),
            "logical_input_shape": list(sample_shape),
            "sample_order": "target date first, then row-major ERA5-Land cell index",
            "dtype": "float32",
        }, stage / "preprocessing.json")
        stage.rename(prepared)
    print(f"Prepared {sample_shape} member input shape in {prepared}. "
          f"Excluded {len(excluded)} observation cells with incomplete forcing.", flush=True)


def verify(root):
    from LSTM_model.model.predict_era5land_observations import validate_prepared_inputs
    from LSTM_model.model.ensemble_inputs import (
        load_feature_scalers, iter_member_input_batches,
    )

    prepared = root / "prepared"
    days, pixels = validate_prepared_inputs(prepared)
    dynamic = np.load(prepared / "dynamic_features.npy", mmap_mode="r", allow_pickle=False)
    static = np.load(prepared / "static_features.npy", mmap_mode="r", allow_pickle=False)
    if (dynamic.shape != (3, days + LOOKBACK, pixels) or static.shape != (5, pixels)
            or not np.isfinite(dynamic).all() or not np.isfinite(static).all()):
        raise ValueError("Invalid staged predictor shapes or values.")
    np.testing.assert_array_equal(np.load(prepared / "context_dates.npy"), MODEL_DAYS.values.astype("datetime64[D]"))
    np.testing.assert_array_equal(np.load(prepared / "target_dates.npy"), TARGET_DAYS.values.astype("datetime64[D]"))

    total = days * pixels
    for member in range(100):
        means, stds = load_feature_scalers(member)
        if not (stds > 0).all():
            raise ValueError(f"Member {member} has a non-positive standard deviation.")
    means, stds = load_feature_scalers(0)
    for index in sorted({0, pixels - 1, pixels, total - 1}):
        indices, batch = next(iter_member_input_batches(
            prepared, 0, batch_size=1, rank=index, world_size=total, with_indices=True,
        ))
        t, p = divmod(index, pixels)
        expected = np.empty((LOOKBACK, len(FEATURES_FILES)), dtype=np.float32)
        expected[:, :3] = dynamic[:, t:t + LOOKBACK, p].T
        expected[:, 3:] = static[:, p]
        np.testing.assert_array_equal(indices, [index])
        np.testing.assert_array_equal(batch, ((expected - means) / stds)[None])
    print(f"Verified ({total:,}, {LOOKBACK}, {len(FEATURES_FILES)}) input shape, "
          "first/last lookbacks, and the 100 member feature scalers.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "download", "prepare", "verify", "all"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--observations", type=Path, default=OBSERVATIONS)
    parser.add_argument("--static-dir", type=Path, default=INPUTS)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.stage in ("select", "all"):
        select_points(args.root, args.observations, args.static_dir)
    if args.stage in ("download", "all"):
        download(args.root, read_points(args.root), args.workers)
    if args.stage in ("prepare", "all"):
        prepare(args.root, read_points(args.root))
    if args.stage in ("verify", "all"):
        verify(args.root)


if __name__ == "__main__":
    main()
