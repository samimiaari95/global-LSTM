"""Evaluate monthly CONUS LSTM ensemble anomalies against NWIS observations."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from LSTM_model.data.conus_observations import MONTHLY_END, MONTHLY_GRID, MONTHLY_START, read_sites
from LSTM_model.data.postprocess_era5land_observations import METRICS, MEMBERS, figures, metrics
from LSTM_model.data.predict_era5land_observations import validate_prepared_inputs

MONTHLY_OUTPUTS = Path("outputs/CONUSmonthly")


def anomalies(values, periods):
    """Use each series' own monthly climatology, including for submonthly periods."""
    result = np.full(values.shape, np.nan)
    for month in np.unique(periods.month):
        group = periods.month == month
        block = values[group]
        varying = np.ptp(block, axis=0) > 0
        result[np.ix_(group, varying)] = (
            (block[:, varying] - block[:, varying].mean(axis=0)) / block[:, varying].std(axis=0))
    return result


def observed_cells(prepared, pixels, periods):
    stations = read_sites(prepared / "stations_with_static.csv")
    values = pd.read_csv(prepared / "selected_wtd.csv", index_col="date", parse_dates=True)
    values.index = values.index.to_period(periods.freq)
    if not values.index.is_unique or not periods.isin(values.index).all():
        raise ValueError("Selected observations do not cover the prediction periods")
    pixel_index = pd.MultiIndex.from_arrays(pixels.T, names=["y", "x"])
    station_groups = stations.groupby(["y", "x"]).groups
    block = values.loc[periods]
    raw = np.column_stack([block[station_groups[cell]].mean(axis=1, skipna=False)
                           for cell in pixel_index])
    if not np.isfinite(raw).all():
        raise ValueError("Selected observations contain a gap")
    return raw, np.array([len(station_groups[cell]) for cell in pixel_index])


def aggregate_predictions(prepared, predictions, dates, periods, starts, counts, ncell, scratch):
    dims = (len(periods), ncell)
    mean_raw, mean, m2 = (np.zeros(dims) for _ in range(3))
    members = np.lib.format.open_memmap(
        scratch / "member_anomalies.npy", mode="w+", dtype=np.float32,
        shape=(ncell, MEMBERS, len(periods)))
    for member in range(MEMBERS):
        path = predictions / f"member_{member:03d}" / "daily_wtd.npy"
        info = json.loads(path.with_suffix(".json").read_text())
        daily = np.load(path, mmap_mode="r", allow_pickle=False)
        if (info["member"] != member or daily.shape != (len(dates), ncell)
                or list(daily.shape) != info["shape"]
                or Path(info["target_dates_file"]).resolve() != prepared / "target_dates.npy"
                or Path(info["selected_pixels_file"]).resolve() != prepared / "selected_pixels.npy"):
            raise ValueError(f"Prediction metadata does not match the prepared run: {path}")
        raw = np.add.reduceat(daily, starts, axis=0, dtype=float) / counts[:, None]
        if not np.isfinite(raw).all():
            raise ValueError(f"Nonfinite daily values in {path}")
        mean_raw += (raw - mean_raw) / (member + 1)
        anomaly = anomalies(raw, periods)
        members[:, member, :] = anomaly.T
        delta = anomaly - mean
        mean += delta / (member + 1)
        m2 += delta * (anomaly - mean)
    return mean_raw, mean, np.sqrt(np.maximum(m2, 0) / (MEMBERS - 1)), members


def evaluate(prepared, predictions, output, plot_cells=None):
    prepared = prepared.resolve()
    _, ncell = validate_prepared_inputs(prepared, conus=True)
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    selection = json.loads((prepared / "selection.json").read_text())
    if (selection["cadence"], selection["start"], selection["end"]) != (
        "monthly", MONTHLY_START, MONTHLY_END
    ):
        raise ValueError(f"CONUSmonthly requires the {MONTHLY_START} to {MONTHLY_END} selection")
    dates = pd.DatetimeIndex(np.load(prepared / "target_dates.npy", allow_pickle=False))
    if str(dates[0].date()) != MONTHLY_START or str(dates[-1].date()) != MONTHLY_END:
        raise ValueError("Prediction dates do not match the monthly study window")
    _, starts, counts = np.unique(dates.to_period("M").asi8,
                                 return_index=True, return_counts=True)
    periods = dates[starts].to_period("M")
    pixels = np.load(prepared / "selected_pixels.npy", allow_pickle=False)
    static = np.load(prepared / "static_features.npy", allow_pickle=False)
    indices = list(range(ncell)) if plot_cells is None else sorted(set(plot_cells))
    if any(i < 0 or i >= ncell for i in indices):
        raise ValueError("Plot cell IDs must index selected_pixels.npy")
    observed, station_count = observed_cells(prepared, pixels, periods)
    obs = anomalies(observed, periods)
    cells = pd.DataFrame(dict(model_y=pixels[:, 0], model_x=pixels[:, 1],
                              longitude=static[3], latitude=static[4], station_count=station_count))
    cells.index.name = "cell"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as workspace:
        scratch = Path(workspace)
        raw, pred, spread, members = aggregate_predictions(
            prepared, predictions, dates, periods, starts, counts, ncell, scratch)
        cells = cells.join(metrics(obs, pred, spread, members))
        if not cells.paired_months.sum():
            raise ValueError("No matched finite anomalies")
        result = scratch / "results"
        result.mkdir()
        cells.to_csv(result / "cell_metrics.csv")
        cells[METRICS].describe().T.to_csv(result / "metric_summary.csv", index_label="metric")
        dataset = xr.Dataset(coords={
            "time": periods.start_time, "cell": cells.index,
            **{name: ("cell", cells[name]) for name in
               ["model_y", "model_x", "latitude", "longitude", "station_count"]},
        })
        for name, values in [
            ("observed_wtd", observed), ("ensemble_mean_native_wtd", raw),
            ("observed_anomaly", obs), ("ensemble_anomaly", pred),
            ("ensemble_anomaly_sd", spread),
        ]:
            dataset[name] = (("time", "cell"), values.astype(np.float32))
        dataset["observed_wtd"].attrs["units"] = "m"
        dataset["prediction_days"] = ("time", counts.astype(np.int32))
        dataset.attrs.update(cadence="monthly", members=MEMBERS,
                             anomaly_baseline="Separate calendar-month z-scores, 2020–2024, ddof=0")
        dataset.to_netcdf(result / "evaluation_series.nc", engine="scipy")
        figures(result, cells, periods, obs, pred, members, indices,
                map_options={"full": (-126, 23, -65, 51), "zoom": None})
        del members
        result.rename(output)
    print(f"Saved {ncell} cells and {int(cells.paired_months.sum())} paired cell-months to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=MONTHLY_GRID / "prepared")
    parser.add_argument("--predictions", type=Path, default=MONTHLY_OUTPUTS / "predictions")
    parser.add_argument("--output", type=Path, default=MONTHLY_OUTPUTS / "evaluation" / "run_2020_2024")
    parser.add_argument("--plot-cells", type=int, nargs="+")
    args = parser.parse_args(argv)
    evaluate(args.prepared, args.predictions, args.output, args.plot_cells)


if __name__ == "__main__":
    main()
