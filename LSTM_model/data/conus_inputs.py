"""Fetch CONUS static fields and ERA5-Land predictors for the groundwater workflow."""

import argparse
import json
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from LSTM_model.data.conus_observations import (
    MONTHLY_GRID, MONTHLY_INPUTS, connect_hydrodata, read_sites, save_csv, save_json,
)
from LSTM_model.model.config import FEATURES_FILES, LOOKBACK


DAILY_VARIABLES = {
    "t2m": "2m_temperature", "d2m": "2m_dewpoint_temperature",
    "swvl3": "volumetric_soil_water_layer_3",
}
ERA5_PRODUCTS = {"daily": "derived-era5-land-daily-statistics", "tp23": "reanalysis-era5-land"}
STATIC_VARIABLES = ("slope_x", "slope_y", "longitude", "latitude", "pf_indicator")
CONUS2_SHAPE = (3256, 4442)


def download_conus2_field(hf, folder, variable):
    height, width = CONUS2_SHAPE
    shape = (10, height, width) if variable == "pf_indicator" else CONUS2_SHAPE
    path = folder / f"{variable}.npy"
    if path.is_file():
        if np.load(path, mmap_mode="r", allow_pickle=False).shape != shape:
            raise ValueError(f"Unexpected shape in cached CONUS2 field {path}")
        return

    # Limit the size of each request for the volume with ten subsurface layers.
    band_height = 512 if variable == "pf_indicator" else height
    temporary = path.with_suffix(".partial")
    result = None
    for y0 in range(0, height, band_height):
        y1 = min(y0 + band_height, height)
        print(f"Downloading CONUS2 {variable}, rows {y0}–{y1 - 1}", flush=True)
        data = np.asarray(hf.get_gridded_data(
            dataset="conus2_domain", grid="conus2", temporal_resolution="static",
            variable=variable, grid_bounds=[0, y0, width, y1],
        ))
        expected = (*shape[:-2], y1 - y0, width)
        if data.shape == (1, *expected):
            data = data[0]
        if data.shape != expected:
            raise ValueError(f"CONUS2 {variable} has shape {data.shape}; expected {expected}")
        if result is None:
            result = np.lib.format.open_memmap(temporary, mode="w+", dtype=data.dtype, shape=shape)
        elif result.dtype != data.dtype:
            raise ValueError(f"Source dtype changed between bands of CONUS2 {variable}")
        result[..., y0:y1, :] = data
        del data
    result.flush()
    del result
    temporary.replace(path)


def fetch_conus2_static(root):
    folder = root / "static" / "conus2"
    folder.mkdir(parents=True, exist_ok=True)
    variables = (*STATIC_VARIABLES, "mask")
    hf = connect_hydrodata() if any(not (folder / f"{name}.npy").is_file()
                                    for name in variables) else None
    with ThreadPoolExecutor(max_workers=3) as workers:
        list(workers.map(lambda name: download_conus2_field(hf, folder, name), variables))
    save_json({
        "dataset": "conus2_domain", "grid": "conus2", "temporal_resolution": "static",
        "shape_yx": list(CONUS2_SHAPE), "pf_indicator_layers": "10, bottom to top",
        "variables": list(variables),
    }, folder / "metadata.json")
    print(f"CONUS2 static snapshot ready in {folder}", flush=True)


def no_leap_dates(start, end):
    dates = pd.date_range(start, end)
    return dates[~((dates.month == 2) & (dates.day == 29))]


def stage_calendar(root):
    selection = json.loads((root / "selection.json").read_text())
    targets = no_leap_dates(selection["start"], selection["end"])
    prior = no_leap_dates(targets[0] - pd.Timedelta(days=LOOKBACK + 2),
                          targets[0] - pd.Timedelta(days=1))
    dates = prior[-LOOKBACK:].append(targets)
    if len(prior) < LOOKBACK or dates[0] < pd.Timestamp("1950-01-01"):
        raise ValueError("ERA5-Land cannot supply the required lookback")
    return dates, targets


def forcing_area(pixels):
    """Snap outwards and add a coastal buffer of two ERA5-Land cells."""
    return [
        round(np.ceil(pixels.latitude.max() * 10) / 10 + .2, 1),
        round(np.floor(pixels.longitude.min() * 10) / 10 - .2, 1),
        round(np.floor(pixels.latitude.min() * 10) / 10 - .2, 1),
        round(np.ceil(pixels.longitude.max() * 10) / 10 + .2, 1),
    ]


def forcing_cache(root, area):
    return root / "era5land" / "_".join(f"{coordinate:.1f}" for coordinate in area)


def forcing_month(month, last_date):
    last_day = month.days_in_month
    if month == last_date.to_period("M"):
        last_day = last_date.day
    key = str(month) if last_day == month.days_in_month else f"{month}-{last_day:02d}"
    return key, last_day


def forcing_batches(dates, label):
    """Keep daily-statistics requests within the CDS 400-field limit."""
    months = dates.to_period("M").unique()
    size = 4 if label == "daily" else 12
    for year in months.year.unique():
        year_months = months[months.year == year]
        partial_last = year_months[-1] == dates[-1].to_period("M") and (
            dates[-1].day < year_months[-1].days_in_month)
        if partial_last:
            tail = year_months[-1:]
            year_months = year_months[:-1]
        for start in range(0, len(year_months), size):
            yield year_months[start:start + size]
        if partial_last:
            yield tail


def forcing_path(folder, months, final_date, label):
    if len(months) == 1:
        key, _ = forcing_month(months[0], final_date)
    else:
        key = f"{months[0]}_to_{months[-1]}"
    return folder / f"{key}_{label}.download"


def forcing_sources(folder, months, final_date, label):
    batch_path = forcing_path(folder, months, final_date, label)
    if batch_path.is_file() or len(months) == 1:
        return [(batch_path, months)]
    individual = [(forcing_path(folder, months[i:i+1], final_date, label), months[i:i+1])
                  for i in range(len(months))]
    existing = np.array([path.is_file() for path, _ in individual])
    if not existing.any():
        return [(batch_path, months)]
    sources = []
    switches = np.flatnonzero(existing[1:] != existing[:-1]) + 1
    for start, end in zip(np.r_[0, switches], np.r_[switches, len(months)]):
        if existing[start]:
            sources.extend(individual[start:end])
        else:
            gap = months[start:end]
            sources.append((forcing_path(folder, gap, final_date, label), gap))
    return sources


def fetch_era5(root):
    import cdsapi

    dates, _ = stage_calendar(root)
    area = forcing_area(pd.read_csv(root / "static_pixels.csv"))
    folder = forcing_cache(root, area)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = []
    for label, dataset in ERA5_PRODUCTS.items():
        for batch in forcing_batches(dates, label):
            for path, months in forcing_sources(folder, batch, dates[-1], label):
                if path.is_file():
                    continue
                last_day = max(forcing_month(month, dates[-1])[1] for month in months)
                query = {
                    "year": str(months[0].year), "month": [f"{month.month:02d}" for month in months],
                    "area": area, "day": [f"{day:02d}" for day in range(1, last_day + 1)],
                }
                if label == "daily":
                    query.update(variable=list(DAILY_VARIABLES.values()), daily_statistic="daily_mean",
                                 time_zone="utc+00:00", frequency="1_hourly")
                else:
                    query.update(variable=["total_precipitation"], time=["23:00"],
                                 data_format="netcdf", download_format="unarchived")
                jobs.append((path, dataset, query))
    clients = threading.local()

    def fetch(job):
        path, dataset, query = job
        if not hasattr(clients, "cds"):
            clients.cds = cdsapi.Client(quiet=True, timeout=120, sleep_max=30, progress=False)
        print(f"Requesting ERA5-Land {path.stem} for {area}", flush=True)
        temporary = path.with_suffix(".partial")
        clients.cds.retrieve(dataset, query).download(str(temporary))
        temporary.replace(path)
        print(f"Saved ERA5-Land {path.name}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        requests = [pool.submit(fetch, job) for job in sorted(jobs, key=lambda job: job[0].name)]
        try:
            for request in requests:
                request.result()
        finally:
            for request in requests:
                request.cancel()


def read_era5(path, variables, pixels, dates, hour):
    if not path.is_file():
        raise FileNotFoundError(f"Run the ERA5-Land download stage to obtain {path}")
    values = {name: np.empty((len(dates), len(pixels)), dtype=np.float32) for name in variables}
    seen = {name: np.zeros(len(dates), dtype=bool) for name in variables}
    with tempfile.TemporaryDirectory(prefix="conus-era5-") as scratch:
        files = [path]
        if zipfile.is_zipfile(path):
            files = []
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.filename.endswith(".nc"):
                        continue
                    target = Path(scratch) / f"{len(files)}.nc"
                    with archive.open(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    files.append(target)
        for file in files:
            with xr.open_dataset(file) as ds:
                for short, long in variables.items():
                    names = [name for name in (short, long) if name in ds]
                    if not names:
                        continue
                    da = ds[names[0]]
                    time = "valid_time" if "valid_time" in da.dims else "time"
                    times = pd.DatetimeIndex(da[time].values)
                    if (not times.normalize().is_unique
                            or not (times == times.normalize() + pd.Timedelta(hours=hour)).all()):
                        raise ValueError(f"Unexpected time axis in {path} for {short}")
                    indices = times.normalize().get_indexer(dates)
                    present = indices >= 0
                    if not present.any():
                        continue
                    if (seen[short] & present).any():
                        raise ValueError(f"Duplicate {short} dates in {path}")
                    longitude = pixels.longitude.to_numpy()
                    if da.longitude.min().item() >= 0:
                        longitude = longitude % 360
                    samples = da.isel({time: indices[present]}).sel(
                        latitude=xr.DataArray(pixels.latitude.to_numpy(), dims="pixel"),
                        longitude=xr.DataArray(longitude, dims="pixel"),
                        method="nearest", tolerance=.051,
                    ).transpose(time, "pixel")
                    values[short][present] = samples.values
                    seen[short] |= present
    for short, present in seen.items():
        if not present.all():
            raise ValueError(f"ERA5-Land {short} missing for {dates[~present][0].date()} in {path}")
    return values


def stage_daily_predictors(dynamic, mask, fields):
    temperature = fields["t2m"] - 273.15
    dewpoint = fields["d2m"] - 273.15
    es = .611 * np.exp(17.27 * temperature / (temperature + 237.3))
    ea = .611 * np.exp(17.27 * dewpoint / (dewpoint + 237.3))
    dynamic[1, mask, :] = es - ea
    dynamic[2, mask, :] = fields["swvl3"]


def load_arco_predictors(root, dates, pixels, dynamic):
    path = root / "era5land" / "arco_daily_points.npz"
    with np.load(path, allow_pickle=False) as data:
        source_dates = pd.DatetimeIndex(data["context_dates"])
        if not source_dates.is_unique or source_dates.hasnans:
            raise ValueError(f"Invalid ARCO date axis: {path}")
        indices = source_dates.get_indexer(dates)
        if ((indices < 0).any()
                or not np.array_equal(data["selected_pixels"], pixels[["y", "x"]].to_numpy())):
            raise ValueError(f"ARCO does not cover the selected dates or cell order: {path}")
        if any(data[name].shape != (len(source_dates), len(pixels))
               for name in (*DAILY_VARIABLES, "tp")):
            raise ValueError(f"ARCO fields do not match their dates and cells: {path}")
        fields = {name: data[name][indices] for name in (*DAILY_VARIABLES, "tp")}
    expected_shape = (len(dates), len(pixels))
    if any(array.shape != expected_shape for array in fields.values()):
        raise ValueError(f"Expected predictors of shape {expected_shape}: {path}")
    dynamic[0] = fields["tp"] * 1000
    stage_daily_predictors(dynamic, slice(None), fields)


def prepare(root, forcing_source="cds"):
    dates, targets = stage_calendar(root)
    pixels = pd.read_csv(root / "static_pixels.csv")
    folder = forcing_cache(root, forcing_area(pixels))
    prepared = root / "prepared"
    if prepared.exists():
        raise FileExistsError(f"{prepared} already exists. Use a fresh run directory to restage inputs.")
    month_index = dates.to_period("M")
    with tempfile.TemporaryDirectory(prefix="conus-preparation-", dir=root) as scratch:
        stage = Path(scratch) / "prepared"
        stage.mkdir()
        raw_path = Path(scratch) / "uncut.npy"
        dynamic = np.lib.format.open_memmap(
            raw_path, mode="w+", dtype=np.float32,
            shape=(3, len(dates), len(pixels)))
        first_missing_month = np.full(len(pixels), "", dtype="U7")
        if forcing_source == "arco":
            load_arco_predictors(root, dates, pixels, dynamic)
        else:
            for label in ERA5_PRODUCTS:
                for months in forcing_batches(dates, label):
                    for path, source_months in forcing_sources(folder, months, dates[-1], label):
                        mask = month_index.isin(source_months)
                        if label == "tp23":
                            precip = read_era5(
                                path, {"tp": "total_precipitation"}, pixels, dates[mask], 23)["tp"]
                            dynamic[0, mask, :] = precip * 1000
                        else:
                            fields = read_era5(path, DAILY_VARIABLES, pixels, dates[mask], 0)
                            stage_daily_predictors(dynamic, mask, fields)
        for month in month_index.unique():
            mask = month_index == month
            missing = ~np.isfinite(dynamic[:, mask, :]).all(axis=(0, 1))
            first_missing_month[missing & (first_missing_month == "")] = str(month)
        dynamic.flush()
        keep = first_missing_month == ""
        excluded = pixels.loc[~keep, ["y", "x", "latitude", "longitude"]].copy()
        excluded["first_missing_month"] = first_missing_month[~keep]
        save_csv(excluded, root / "excluded_forcing_pixels.csv", index=False)
        if not keep.any():
            raise ValueError("Every selected cell had incomplete ERA5-Land forcing")
        if keep.all():
            del dynamic
            raw_path.rename(stage / "dynamic_features.npy")
        else:
            reduced = np.lib.format.open_memmap(
                stage / "dynamic_features.npy", mode="w+", dtype=np.float32,
                shape=(3, len(dates), int(keep.sum())))
            for start in range(0, len(dates), 128):
                reduced[:, start:start + 128, :] = dynamic[:, start:start + 128, :][:, :, keep]
            reduced.flush()
            del reduced, dynamic
        pixels = pixels.loc[keep].copy()
        static = pixels[["slope_x", "slope_y", "soil_indicator", "longitude", "latitude"]]
        np.save(stage / "static_features.npy", static.to_numpy(dtype=np.float32).T)
        np.save(stage / "selected_pixels.npy", pixels[["y", "x"]].to_numpy(dtype=np.int32))
        np.save(stage / "context_dates.npy", dates.to_numpy(dtype="datetime64[D]"))
        np.save(stage / "target_dates.npy", targets.to_numpy(dtype="datetime64[D]"))
        stations = read_sites(root / "stations_with_static.csv")
        matched = pd.MultiIndex.from_frame(stations[["y", "x"]]).isin(
            pd.MultiIndex.from_frame(pixels[["y", "x"]]))
        stations = stations.loc[matched]
        save_csv(stations, stage / "stations_with_static.csv")
        observations = pd.read_csv(root / "selected_wtd.csv", index_col="date")
        save_csv(observations.loc[:, stations.index], stage / "selected_wtd.csv")
        shutil.copyfile(root / "selection.json", stage / "selection.json")
        grid_metadata = root / "gridding.json"
        grid_info = json.loads(grid_metadata.read_text()) if grid_metadata.is_file() else {}
        save_json({
            "region": "CONUS", "feature_files": FEATURES_FILES, "lookback_days": LOOKBACK,
            "target_start_date": str(targets[0].date()), "target_end_date": str(targets[-1].date()),
            "forcing_source": forcing_source,
            "forcing_area": forcing_area(pd.read_csv(root / "static_pixels.csv")),
            "selected_pixel_count": len(pixels), "retained_station_count": len(stations),
            **({"grid": grid_info["grid"]} if "grid" in grid_info else {}),
        }, stage / "preprocessing.json")
        stage.rename(prepared)
    print(f"Prepared {len(targets)} daily targets at {len(pixels)} cells in {prepared}", flush=True)


def preflight(root):
    from LSTM_model.data.predict_era5land_observations import (
        ENSEMBLE_DIRECTORY, MODEL_NAME, load_target_scalers, validate_prepared_inputs,
    )
    from LSTM_model.data.preprocess_era5land_observation_inputs import iter_member_input_batches

    prepared = (root / "prepared").resolve()
    dates, pixel_count = validate_prepared_inputs(prepared, conus=True)
    pixels = np.load(prepared / "selected_pixels.npy", allow_pickle=False)
    dynamic = np.load(prepared / "dynamic_features.npy", mmap_mode="r", allow_pickle=False)
    static = np.load(prepared / "static_features.npy", mmap_mode="r", allow_pickle=False)
    dynamic_count = len(DAILY_VARIABLES)
    if (dynamic.shape != (dynamic_count, dates + LOOKBACK, pixel_count)
            or static.shape != (len(FEATURES_FILES) - dynamic_count, pixel_count)
            or not np.isfinite(dynamic).all() or not np.isfinite(static).all()):
        raise ValueError("Prepared feature arrays have invalid dimensions or missing values")

    grid_metadata = root / "gridding.json"
    if grid_metadata.is_file():
        grid = json.loads(grid_metadata.read_text())
        if observation_file := grid.get("gridded_observations_file"):
            archived = pd.read_csv(root / observation_file, usecols=["y", "x"]).drop_duplicates()
            archived_cells = set(archived.itertuples(index=False, name=None))
            excluded_cells = set()
            for name in ("excluded_static_pixels.csv", "excluded_forcing_pixels.csv"):
                excluded = set(pd.read_csv(root / name, usecols=["y", "x"]).itertuples(
                    index=False, name=None))
                if excluded & excluded_cells:
                    raise ValueError("Static and forcing exclusions contain overlapping cells")
                excluded_cells.update(excluded)
            prediction_cells = set(map(tuple, pixels))
            if (len(archived_cells) != grid["observation_cells"]
                    or prediction_cells & excluded_cells
                    or (prediction_cells | excluded_cells) != archived_cells):
                raise ValueError("Prepared cells and exclusions do not partition the gridded observations")
    for member in range(100):
        identifier = f"400px_member_{member}"
        checkpoint = ENSEMBLE_DIRECTORY / identifier / f"{identifier}_{MODEL_NAME}.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty checkpoint {checkpoint}")
        load_target_scalers(member)
        batch = next(iter_member_input_batches(prepared, member, batch_size=1))
        if batch.shape != (1, LOOKBACK, len(FEATURES_FILES)) or not np.isfinite(batch).all():
            raise ValueError(f"Invalid staged sample for ensemble member {member}")
    print(f"Prediction preflight passed for {dates} daily targets, {pixel_count} cells and 100 members.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["static-conus2", "download-era5", "prepare", "preflight"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--forcing-source", choices=["cds", "arco"], default="arco",
                        help="ERA5-Land data source used by the prepare stage")
    args = parser.parse_args()
    if args.stage == "static-conus2":
        fetch_conus2_static(args.root or MONTHLY_INPUTS)
    elif args.stage == "download-era5":
        fetch_era5(args.root or MONTHLY_GRID)
    elif args.stage == "prepare":
        prepare(args.root or MONTHLY_GRID, args.forcing_source)
    else:
        preflight(args.root or MONTHLY_GRID)


if __name__ == "__main__":
    main()
