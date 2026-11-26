"""Read ECMWF ERA5-Land ARCO point data and derive daily predictors."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from xarray.coding.times import decode_cf_datetime


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


def stage_daily_predictors(dynamic, mask, fields):
    temperature = fields["t2m"] - 273.15
    dewpoint = fields["d2m"] - 273.15
    es = .611 * np.exp(17.27 * temperature / (temperature + 237.3))
    ea = .611 * np.exp(17.27 * dewpoint / (dewpoint + 237.3))
    dynamic[1, mask, :] = es - ea
    dynamic[2, mask, :] = fields["swvl3"]
