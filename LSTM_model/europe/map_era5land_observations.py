"""Map monthly European groundwater anomalies onto the native ERA5-Land grid."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs" / "EU_74_-48_69_20"
GRID = (
    INPUTS / "raw" / "ERA5-Land" / "daily_mean_d2m_t2m_swvl3"
    / "ERA5land_swvl3_199601.nc"
)
OUTPUT = INPUTS / "localobservations" / "wtd_anomalies_era5land_0p1deg.nc"
SOURCES = {
    "England": "England_GW_data/england_gw_monthly_wells",
    "GEMS-GER": "GEMS-GER_data/gems_ger_monthly_wells",
    "GROW Europe": "GROW_europe_monthly_wells",
    "Switzerland": "Swiss_GW_data/swiss_gw_monthly_wells",
}
EARTH_RADIUS_KM = 6373.0
TILE_SIZE = 16


def read_grid(path):
    with Dataset(path) as dataset:
        axes = tuple(np.asarray(dataset[name][:]) for name in ("latitude", "longitude"))
    for axis in axes:
        if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
            raise ValueError("Expected finite, one-dimensional ERA5-Land coordinate axes.")
        step = np.diff(axis)
        if not np.allclose(step, step[0]) or not np.isclose(abs(step[0]), 0.1):
            raise ValueError("Expected the native 0.1 degree ERA5-Land grid.")
    return axes


def nearest_cell(lat, lon, grid_lat_rad, grid_lon_rad):
    """Minimize the Haversine distance over a rectilinear grid."""
    lat, lon = np.deg2rad([lat, lon])
    sin2_half_dlon = np.sin((grid_lon_rad - lon) / 2) ** 2
    # The longitude term has a nonnegative coefficient at every latitude.
    # Once longitude is fixed, search all latitudes for the spherical minimum.
    x = int(np.argmin(sin2_half_dlon))
    a = (
        np.sin((grid_lat_rad - lat) / 2) ** 2
        + np.cos(lat) * np.cos(grid_lat_rad) * sin2_half_dlon[x]
    )
    y = int(np.argmin(a))
    distance_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a[y], 0, 1)))
    return (y, x), float(distance_km)


def select_wells(root, latitude, longitude):
    grid_lat_rad, grid_lon_rad = np.deg2rad(latitude), np.deg2rad(longitude)
    lon_mid = (longitude.min() + longitude.max()) / 2
    lat_bounds = (latitude.min() - 0.05, latitude.max() + 0.05)
    lon_bounds = (longitude.min() - 0.05, longitude.max() + 0.05)
    winners, candidates = {}, Counter()
    for source, folder in SOURCES.items():
        paths = sorted((root / folder).glob("[!_.]*.csv"))
        if not paths:
            raise FileNotFoundError(f"No station CSVs in {root / folder}")
        counts = Counter()
        for path in paths:
            data = pd.read_csv(path, usecols=["month", "lat", "lon", "wtd_a"])
            valid = data.loc[np.isfinite(data["wtd_a"])]
            if valid.empty:
                counts["no anomalies"] += 1
                continue
            coordinates = valid[["lat", "lon"]].dropna().drop_duplicates()
            if len(coordinates) != 1:
                raise ValueError(f"Expected one valid well location in {path}")
            lat, lon = coordinates.iloc[0]
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"Invalid geographic coordinates in {path}")
            grid_lon = (lon - lon_mid + 180) % 360 - 180 + lon_mid
            if not (lat_bounds[0] <= lat <= lat_bounds[1]
                    and lon_bounds[0] <= grid_lon <= lon_bounds[1]):
                counts["outside grid"] += 1
                continue

            months = pd.PeriodIndex(valid["month"], freq="M")
            if months.hasnans or months.has_duplicates:
                raise ValueError(f"Invalid or repeated anomaly month in {path}")
            series = pd.Series(valid["wtd_a"].to_numpy(), index=months)
            cell, distance = nearest_cell(lat, grid_lon, grid_lat_rad, grid_lon_rad)
            candidates[cell] += 1
            counts["eligible"] += 1
            # Distance ranks first. Source and file stem resolve exact ties.
            rank = (distance, source, path.stem)
            if cell not in winners or rank < winners[cell][0]:
                winners[cell] = (rank, lat, lon, series)
        print(f"{source}: {len(paths):,} wells read, {counts['eligible']:,} eligible, "
              f"{counts['outside grid']} outside grid, {counts['no anomalies']} without anomalies",
              flush=True)
    if not winners:
        raise ValueError("No wells with finite anomalies inside the ERA5-Land domain.")
    return winners, candidates


def write_netcdf(path, grid_path, latitude, longitude, winners, candidates):
    cells = sorted(winners)
    first = min(series.index.min() for _, _, _, series in winners.values())
    last = max(series.index.max() for _, _, _, series in winners.values())
    months = pd.period_range(first, last, freq="M")
    origin = pd.Timestamp("1970-01-01")
    time = (months.to_timestamp() - origin).days
    time_end = ((months + 1).to_timestamp() - origin).days
    tiles = defaultdict(list)
    for y, x in cells:
        tiles[(y // TILE_SIZE, x // TILE_SIZE)].append((y, x))

    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.setncatts({
            "Conventions": "CF-1.8",
            "title": "European monthly well water table depth anomalies on the ERA5-Land grid",
            "grid_reference": str(grid_path),
            "assignment": "Nearest ERA5-Land center by Haversine distance (sphere radius 6373 km).",
            "selection": "Shortest distance across sources per cell. Exact ties use source and well ID.",
            "anomaly_source": "Supplied monthly well CSV wtd_a values with source climatologies.",
            "eligible_well_count": sum(candidates.values()),
            "selected_well_count": len(cells),
        })
        for name, size in (("time", len(time)), ("bounds", 2), ("well", len(cells)),
                           ("latitude", len(latitude)), ("longitude", len(longitude))):
            dataset.createDimension(name, size)

        for name, values, units, axis in (
            ("latitude", latitude, "degrees_north", "Y"),
            ("longitude", longitude, "degrees_east", "X"),
        ):
            coord = dataset.createVariable(name, "f8", (name,))
            coord.setncatts({"standard_name": name, "units": units, "axis": axis})
            coord[:] = values
        time_var = dataset.createVariable("time", "i4", ("time",))
        time_var.setncatts({
            "standard_name": "time", "axis": "T", "bounds": "time_bounds",
            "units": "days since 1970-01-01", "calendar": "proleptic_gregorian",
        })
        time_var[:] = time
        dataset.createVariable("time_bounds", "i4", ("time", "bounds"))[:] = (
            np.column_stack((time, time_end))
        )

        y_indices, x_indices = np.asarray(cells).T
        grid_well = dataset.createVariable(
            "well_index", "i4", ("latitude", "longitude"), fill_value=-1, zlib=True
        )
        grid_well.long_name = "Zero-based index into the selected well metadata"
        indices = np.full((len(latitude), len(longitude)), -1, dtype=np.int32)
        indices[y_indices, x_indices] = np.arange(len(cells))
        grid_well[:] = indices

        ranks, well_lats, well_lons, _ = zip(*(winners[cell] for cell in cells))
        distance, source, well_id = zip(*ranks)
        for name, dtype, values, description, units in (
            ("well_id", str, well_id, "Station CSV filename stem", None),
            ("well_source", str, source, "Observation collection", None),
            ("well_latitude", "f8", well_lats, "Well latitude", "degrees_north"),
            ("well_longitude", "f8", well_lons, "Well longitude", "degrees_east"),
            ("well_distance", "f8", distance, "Haversine distance from well to cell center", "km"),
            ("wells_in_cell", "i4", [candidates[cell] for cell in cells],
             "Number of candidate wells assigned to the selected well's cell", "1"),
        ):
            variable = dataset.createVariable(name, dtype, ("well",))
            variable.long_name = description
            if units is not None:
                variable.units = units
            variable[:] = np.asarray(values, dtype=object if dtype is str else None)

        anomalies = dataset.createVariable(
            "wtd_a", "f8", ("time", "latitude", "longitude"), fill_value=np.nan,
            zlib=True,
            chunksizes=(min(120, len(time)), min(TILE_SIZE, len(latitude)),
                        min(TILE_SIZE, len(longitude))),
        )
        anomalies.long_name = "Monthly standardized water table depth anomaly at the selected well"
        anomalies.units = "1"
        anomalies.ancillary_variables = "well_index"
        for (tile_y, tile_x), occupied in tiles.items():
            y0, x0 = tile_y * TILE_SIZE, tile_x * TILE_SIZE
            y1, x1 = min(y0 + TILE_SIZE, len(latitude)), min(x0 + TILE_SIZE, len(longitude))
            values = np.full((len(time), y1 - y0, x1 - x0), np.nan)
            for y, x in occupied:
                values[:, y - y0, x - x0] = winners[y, x][3].reindex(months).to_numpy()
            anomalies[:, y0:y1, x0:x1] = values
    print(f"{first} to {last}: {len(months):,} monthly time steps", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, default=ROOT.parent / "Groundwater_Observations")
    parser.add_argument("--grid", type=Path, default=GRID)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    latitude, longitude = read_grid(args.grid)
    wells, counts = select_wells(args.observations, latitude, longitude)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        write_netcdf(temporary, args.grid.resolve(), latitude, longitude, wells, counts)
        output.hardlink_to(temporary)  # Publish atomically without overwriting.
    retained = Counter(rank[1] for rank, _, _, _ in wells.values())
    print(f"Kept {len(wells):,} wells and excluded {sum(counts.values()) - len(wells):,} "
          f"farther or tied wells across {sum(n > 1 for n in counts.values()):,} shared cells.")
    print(f"Retained by collection: {dict(sorted(retained.items()))}")
    print(output)


if __name__ == "__main__":
    main()
