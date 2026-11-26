"""Prepare and evaluate daily CONUS water-table depths on the ERA5-Land grid."""

import argparse
import csv
import gzip
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from LSTM_model.data.conus_observations import ROOT, read_sites, save_json
from LSTM_model.data.postprocess_era5land_observations import (
    MEMBERS, STATS, ensemble_statistics, figures,
)
from LSTM_model.data.predict_era5land_observations import validate_prepared_inputs
from matplotlib.colors import LogNorm, SymLogNorm

DAILY_INPUTS = Path("inputs/CONUSdaily")
DAILY_OUTPUTS = Path("outputs/CONUSdaily")
SOURCE_PREPARED = DAILY_INPUTS / "source" / "prepared"
SOURCE_PREDICTIONS = DAILY_OUTPUTS / "source" / "predictions"
START, END = "2022-03-01", "2024-02-28"
PLOT_COLUMNS = dict(rmse_m="RMSE", mae_m="MAE", absolute_mean_bias_m="Absolute_mean_bias",
                    pearson_r="Pearson_r", nse="NSE", kge="KGE")
METRIC_LABELS = {
    "RMSE": "RMSE (m)", "MAE": "MAE (m)", "Absolute_mean_bias": "Absolute mean bias (m)",
    "Pearson_r": "Pearson r", "NSE": "NSE", "KGE": "KGE",
}
LOG_METRICS = ("RMSE", "MAE", "Absolute_mean_bias")
LOG_STATISTICS = ("Ensemble_variance", "IQR", "Ensemble_SD", "MAD", "RMS_spread", "obs_SD", "prediction_SD")
SYMLOG_METRICS = ("NSE", "KGE")
STATISTIC_LABELS = {
    "Ensemble_variance": "Ensemble variance (m²)", "IQR": "Ensemble IQR (m)",
    "Pairwise_r": "Mean pairwise Pearson r", "Ensemble_SD": "Ensemble SD (m)",
    "MAD": "Mean absolute deviation (m)", "well_count": "Contributing wells",
    "RMS_spread": "RMS ensemble spread (m)", "obs_SD": "Observed temporal SD (m)",
    "prediction_SD": "Predicted temporal SD (m)",
}


@contextmanager
def stage_directory(destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Use a new output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        staged = Path(scratch) / "results"
        staged.mkdir()
        yield staged
        staged.rename(destination)


def daily_grid(prepared):
    validate_prepared_inputs(prepared, conus=True)
    dates = pd.DatetimeIndex(np.load(prepared / "target_dates.npy").astype("datetime64[ns]"), name="date")
    if not dates.equals(pd.date_range(START, END)):
        raise ValueError(f"CONUSdaily requires daily targets from {START} through {END}")
    pixels = np.load(prepared / "selected_pixels.npy")
    return dates, pd.MultiIndex.from_arrays(pixels.T, names=["y", "x"])


def archive_values(archive, stations, dates):
    parts = []
    remaining = set(stations)
    for path in sorted(archive.glob("sites_*.csv.gz")):
        with gzip.open(path, "rt", newline="") as source:
            header = next(csv.reader(source))
        ids = [site for site in header[1:] if site in stations]
        if not ids:
            continue
        if len(set(ids)) != len(ids) or not remaining.issuperset(ids):
            raise ValueError(f"Duplicate well ID in {path}")
        block = pd.read_csv(path, usecols=["date", *ids], index_col="date", parse_dates=True)
        if not block.index.is_unique:
            raise ValueError(f"Duplicate daily dates in {path}")
        parts.append(block.reindex(dates))
        remaining.difference_update(ids)
    if remaining:
        raise ValueError(f"{len(remaining)} candidate wells are missing from {archive}")
    return pd.concat(parts, axis=1).loc[:, stations]


def prepare(source, archive, destination):
    dates, source_cells = daily_grid(source)
    stations = read_sites(source / "stations_with_static.csv")
    daily = archive_values(archive, stations.index, dates)
    complete = np.isfinite(daily.to_numpy()).all(axis=0)
    eligible = stations.loc[complete]
    groups = eligible.groupby(["y", "x"]).groups
    cells = source_cells[source_cells.isin(groups.keys())]
    if len(cells) == 0 or len(cells) != len(groups):
        raise ValueError("Eligible well cells must fall within the prepared prediction grid")
    columns = source_cells.get_indexer(cells)
    observed = np.column_stack([daily[groups[cell]].mean(axis=1) for cell in cells])
    counts = [len(groups[cell]) for cell in cells]
    records = pd.DataFrame({
        "date": dates.repeat(len(cells)),
        "y": np.tile(cells.get_level_values("y"), len(dates)),
        "x": np.tile(cells.get_level_values("x"), len(dates)),
        "wtd_m": observed.ravel(),
        "well_count": np.tile(counts, len(dates)),
    })
    metadata = json.loads((source / "preprocessing.json").read_text())
    metadata.update(
        subproject="CONUSdaily", selected_pixel_count=len(cells), retained_station_count=len(eligible),
        candidate_station_count=len(stations), candidate_cell_count=len(source_cells),
        source_prepared=str(source.resolve()), source_daily_observations=str(archive.resolve()),
        eligibility="Finite WTD on all target dates for each contributing well",
    )
    with stage_directory(destination) as staged:
        records.to_csv(staged / "observations_gridded.csv", index=False)
        eligible.to_csv(staged / "stations_with_static.csv")
        for name in ("target_dates.npy", "context_dates.npy"):
            shutil.copyfile(source / name, staged / name)
        for name in ("selected_pixels.npy", "dynamic_features.npy", "static_features.npy"):
            values = np.load(source / name, mmap_mode="r")
            subset = values[columns] if name == "selected_pixels.npy" else values[..., columns]
            np.save(staged / name, subset)
        save_json(metadata, staged / "preprocessing.json")
        daily_grid(staged)
    print(f"Prepared {len(dates)} daily values at {len(cells)} cells from {len(eligible)} wells: {destination}")


def read_member(predictions, prepared, member, shape):
    path = predictions / f"member_{member:03d}" / "daily_wtd.npy"
    info = json.loads(path.with_suffix(".json").read_text())
    daily = np.load(path, mmap_mode="r")
    if (info["member"] != member or daily.shape != shape or info["shape"] != list(shape)
            or Path(info["target_dates_file"]).resolve() != (prepared / "target_dates.npy").resolve()
            or Path(info["selected_pixels_file"]).resolve() != (prepared / "selected_pixels.npy").resolve()):
        raise ValueError(f"Prediction metadata does not match prepared dates and cells: {path}")
    if not np.isfinite(daily).all():
        raise ValueError(f"Nonfinite predictions in {path}")
    return path, daily, info


def import_predictions(source, predictions, prepared, destination):
    dates, source_cells = daily_grid(source)
    _, cells = daily_grid(prepared)
    columns = source_cells.get_indexer(cells)
    if (columns < 0).any():
        raise ValueError("Daily cells are missing from the source prediction grid")
    with stage_directory(destination) as staged:
        for member in range(MEMBERS):
            path, daily, info = read_member(predictions, source, member, (len(dates), len(source_cells)))
            relative = path.relative_to(predictions)
            target = staged / relative
            target.parent.mkdir()
            np.save(target, daily[:, columns])
            info.update(
                prepared_inputs=str(prepared.resolve()),
                prediction_file=str((destination / relative).resolve()),
                target_dates_file=str((prepared / "target_dates.npy").resolve()),
                selected_pixels_file=str((prepared / "selected_pixels.npy").resolve()),
                shape=[len(dates), len(cells)],
                source_prediction_file=str(path.resolve()),
            )
            save_json(info, target.with_suffix(".json"))
    print(f"Imported {MEMBERS} daily prediction members for {len(cells)} cells: {destination}")


def ensemble(prepared, predictions, shape):
    mean, squared_deviations = np.zeros(shape), np.zeros(shape)
    days, cells = shape
    members = np.empty((cells, MEMBERS, days), dtype=np.float32)
    for member in range(MEMBERS):
        _, daily, _ = read_member(predictions, prepared, member, shape)
        members[:, member, :] = daily.T
        delta = daily - mean
        mean += delta / (member + 1)
        squared_deviations += delta * (daily - mean)
    return mean, np.sqrt(np.maximum(squared_deviations, 0) / (MEMBERS - 1)), members


def scores(observed, predicted):
    residual = predicted - observed
    obs_mean = observed.mean(axis=0)
    pred_mean = predicted.mean(axis=0)
    obs_centered = observed - obs_mean
    pred_centered = predicted - pred_mean
    obs_variance = np.square(obs_centered).sum(axis=0)
    pred_variance = np.square(pred_centered).sum(axis=0)
    correlation_denominator = np.sqrt(obs_variance * pred_variance)
    sse = np.square(residual).sum(axis=0)
    missing = np.full(observed.shape[1], np.nan)
    obs_varies = (np.ptp(observed, axis=0) > 0) & (obs_variance > 0)
    pred_varies = np.ptp(predicted, axis=0) > 0
    r = np.divide((obs_centered * pred_centered).sum(axis=0), correlation_denominator,
                  out=missing.copy(), where=obs_varies & pred_varies & (correlation_denominator > 0))
    alpha = np.sqrt(np.divide(pred_variance, obs_variance, out=missing.copy(), where=obs_varies))
    beta = np.divide(pred_mean, obs_mean, out=missing.copy(), where=obs_mean != 0)
    mean_bias = residual.mean(axis=0)
    return {
        "rmse_m": np.sqrt(sse / len(observed)),
        "mae_m": np.abs(residual).mean(axis=0),
        "bias_m": mean_bias,
        "absolute_mean_bias_m": np.abs(mean_bias),
        "pearson_r": r,
        "nse": 1 - np.divide(sse, obs_variance, out=missing, where=obs_varies),
        "kge": 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2),
    }


def log_color_scale(values):
    positive = values[values > 0]
    if positive.empty:
        return LogNorm(1, 10)
    lower, upper = positive.min(), positive.max()
    return LogNorm(lower, upper if upper > lower else lower * 10)


def daily_figures(output, table, dates, observed, predicted, members, indices):
    cells = table.rename(columns=PLOT_COLUMNS)
    log_keys = (*LOG_METRICS, *LOG_STATISTICS)
    axis_scales = {**dict.fromkeys(log_keys, "log"), **dict.fromkeys(SYMLOG_METRICS, "symlog")}
    color_scales = {key: ("viridis", log_color_scale(cells[key]))
                    for key in (*LOG_METRICS, "RMS_spread", "obs_SD", "prediction_SD")}
    color_scales.update({key: ("viridis", SymLogNorm(1, vmin=min(-1, cells[key].min()), vmax=1))
                         for key in SYMLOG_METRICS})
    figures(
        output, cells, dates, observed, predicted, members, indices,
        period="day", value_label="WTD (m)", paired_filename="paired_wtd.png",
        metric_labels=METRIC_LABELS, statistic_labels=STATISTIC_LABELS,
        axis_scales=axis_scales, show_example_members=True,
        map_options={
            "full": (-126, 23, -65, 51), "zoom": None, "coverage": "well_count",
            "color_scales": color_scales,
        },
    )


def evaluate(prepared, predictions, output, plot_cells=None):
    dates, cells = daily_grid(prepared)
    indices = list(range(len(cells))) if plot_cells is None else sorted(set(plot_cells))
    if any(i < 0 or i >= len(cells) for i in indices):
        raise ValueError("Plot cell IDs must index the CONUSdaily selected_pixels.npy")
    records = pd.read_csv(prepared / "observations_gridded.csv", parse_dates=["date"])
    records = records.set_index(["date", "y", "x"], verify_integrity=True)
    y, x = (cells.get_level_values(name) for name in cells.names)
    expected = pd.MultiIndex.from_arrays(
        [dates.repeat(len(cells)), np.tile(y, len(dates)), np.tile(x, len(dates))],
        names=records.index.names,
    )
    if len(records) != len(expected) or not expected.isin(records.index).all():
        raise ValueError("Gridded observations must match the prediction dates and cells")
    shape = (len(dates), len(cells))
    records = records.loc[expected]
    observed = records["wtd_m"].to_numpy().reshape(shape)
    counts = records["well_count"].to_numpy().reshape(shape)
    if not np.isfinite(observed).all() or (counts[0] < 1).any() or not (counts == counts[0]).all():
        raise ValueError("Each cell must have finite WTD and a fixed set of contributing wells")
    predicted, spread, members = ensemble(prepared, predictions, shape)
    static = np.load(prepared / "static_features.npy")
    metrics = scores(observed, predicted)
    table = pd.DataFrame(dict(
        y=y, x=x, longitude=static[3], latitude=static[4],
        well_count=counts[0], paired_days=len(dates), **metrics,
        obs_SD=observed.std(axis=0), prediction_SD=predicted.std(axis=0),
        RMS_spread=np.sqrt(np.square(spread).mean(axis=0)),
    ))
    stats = pd.DataFrame(ensemble_statistics(block.astype(float)) for block in members)
    table = table.join(stats.reindex(columns=STATS))
    table.index.name = "cell"
    dataset = xr.Dataset(
        data_vars={
            "observed_wtd": (("time", "cell"), observed, {"units": "m"}),
            "ensemble_mean_wtd": (("time", "cell"), predicted, {"units": "m"}),
            "ensemble_wtd_sd": (("time", "cell"), spread, {"units": "m", "ddof": 1}),
        },
        coords={
            "time": dates.to_numpy(), "cell": table.index.to_numpy(),
            **{name: ("cell", table[name].to_numpy())
               for name in ["y", "x", "longitude", "latitude", "well_count"]},
        },
        attrs={"members": MEMBERS, "observation_aggregation": "Equal-weight mean of complete wells",
               "prediction_transform": "Native destandardized WTD", "calendar_note": "Leap day excluded"},
    )
    with stage_directory(output) as staged:
        dataset.to_netcdf(staged / "evaluation_series.nc", engine="scipy")
        table.to_csv(staged / "cell_metrics.csv")
        table[list(metrics)].describe().T.to_csv(staged / "metric_summary.csv", index_label="metric")
        daily_figures(staged, table, dates, observed, predicted, members, indices)
    print(f"Evaluated {observed.size:,} paired cell-days: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source-prepared", type=Path, default=SOURCE_PREPARED)
    preparation.add_argument("--archive", type=Path, default=ROOT / "observations" / "daily")
    preparation.add_argument("--prepared", type=Path, default=DAILY_INPUTS / "prepared")
    reuse = commands.add_parser("import-predictions")
    reuse.add_argument("--source-prepared", type=Path, default=SOURCE_PREPARED)
    reuse.add_argument("--source-predictions", type=Path, default=SOURCE_PREDICTIONS)
    reuse.add_argument("--prepared", type=Path, default=DAILY_INPUTS / "prepared")
    reuse.add_argument("--predictions", type=Path, default=DAILY_OUTPUTS / "predictions")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--prepared", type=Path, default=DAILY_INPUTS / "prepared")
    evaluation.add_argument("--predictions", type=Path, default=DAILY_OUTPUTS / "predictions")
    evaluation.add_argument("--output", type=Path, default=DAILY_OUTPUTS / "evaluation" / "run_2022_2024")
    evaluation.add_argument("--plot-cells", type=int, nargs="+",
                            help="Limit individual plots to these cell IDs. Summary plots use all cells.")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_prepared, args.archive, args.prepared)
    elif args.command == "import-predictions":
        import_predictions(args.source_prepared, args.source_predictions, args.prepared, args.predictions)
    else:
        evaluate(args.prepared, args.predictions, args.output, args.plot_cells)


if __name__ == "__main__":
    main()
