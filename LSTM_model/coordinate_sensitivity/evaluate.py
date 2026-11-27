"""Evaluate the paired CONUS monthly coordinate sensitivity scenarios."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from LSTM_model.conusmonthly.conusmonthly_evaluation import (
    MAP_EXTENT, MONTHLY_END, MONTHLY_START, aggregate_predictions, anomalies, observed_cells,
)
from LSTM_model.coordinate_sensitivity.prepare import (
    RUN, SCENARIOS, STATIC, baseline_inputs, static_row, verify,
)
from LSTM_model.utils.observation_evaluation import (
    METRICS, country_borders, metrics, save, spatial_axes,
)

import matplotlib.pyplot as plt

LABELS = {
    "RMSE": "RMSE (standardized anomaly)", "MAE": "MAE (standardized anomaly)",
    "Pearson_r": "Pearson correlation", "NSE": "NSE", "KGE": "KGE",
    "prediction_RMSE": "Prediction anomaly RMSE",
}
MAP_METRICS = ("RMSE", "Pearson_r", "KGE", "NSE")
SCENARIO_LABELS = {"A0": "Baseline", "A1": "Longitude", "A2": "Latitude", "A3": "Both"}
COLORS = ("#242424", "#2677a8", "#d18700", "#8d3c8f")


def paths_for_evaluation(run, manifest):
    yield baseline_inputs(manifest)
    for scenario in list(SCENARIOS)[1:]:
        prepared = run / "prepared" / scenario
        predictions = run / "predictions" / scenario
        for member, artifacts in enumerate(manifest["models"]):
            member_directory = predictions / f"member_{member:03d}"
            path = member_directory / "daily_wtd.json"
            if not path.is_file() or not (member_directory / "daily_wtd.npy").is_file():
                raise FileNotFoundError(f"Incomplete {scenario} member {member:03d}: {member_directory}")
            info = json.loads(path.read_text())
            if (Path(info["prepared_inputs"]).resolve() != prepared
                    or any(info[key] != artifacts[key]["path"] for key in ("model", "scaler"))):
                raise ValueError(f"Incorrect scenario inputs, checkpoint or scaler in {path}")
        yield prepared, predictions


def summarize(table):
    rows = []
    for scenario in SCENARIOS:
        data = table.loc[scenario]
        for metric in (*METRICS, "prediction_RMSE"):
            for column in [metric, f"delta_{metric}"] if metric in METRICS else [metric]:
                values = data[column].dropna()
                q25, median, q75 = values.quantile([.25, .50, .75])
                rows.append(dict(scenario=scenario, quantity=column, cells=len(values),
                                 median=median, q25=q25, q75=q75, iqr=q75 - q25))
    return pd.DataFrame(rows)


def plot_boxplots(output, table):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        boxes = ax.boxplot([table.loc[scenario, metric].dropna() for scenario in SCENARIOS],
                           patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], COLORS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(.55)
        ax.set(xticks=range(1, 5), xticklabels=list(SCENARIOS), ylabel=LABELS[metric])
        ax.grid(axis="y", alpha=.25)
        if metric == "Pearson_r":
            ax.set_ylim(-1, 1)
        if metric in ("KGE", "NSE"):
            ax.set_ylim(top=1)
    axes.flat[-1].set_axis_off()
    save(fig, output / "performance_boxplots.png")


def plot_changes(output, table, metric, borders):
    modified = list(SCENARIOS)[1:]
    values = table.loc[modified, f"delta_{metric}"].dropna()
    limit = max(values.abs().max(), 1e-12) if len(values) else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True, sharex=True, sharey=True)
    for ax, scenario in zip(axes, modified, strict=True):
        data = table.loc[scenario].dropna(subset=[f"delta_{metric}"])
        spatial_axes(ax, borders, MAP_EXTENT)
        points = ax.scatter(data.longitude, data.latitude, c=data[f"delta_{metric}"],
                            s=10, cmap="RdBu_r", vmin=-limit, vmax=limit, linewidths=0)
        ax.set(title=f"{scenario} − A0  ({SCENARIO_LABELS[scenario]})",
               xlabel="Longitude (degrees east)")
    axes[0].set_ylabel("Latitude (degrees north)")
    fig.colorbar(points, ax=axes, label=f"Change in {LABELS[metric]}")
    save(fig, output / f"delta_{metric}.png")


def plot_series(output, dataset, table):
    ranking = table.loc["A3", "prediction_RMSE"].dropna().sort_values(kind="stable")
    if len(ranking) < 3:
        raise ValueError("At least three paired cells are required to plot representative locations")
    ranks = np.rint(np.array([.10, .50, .90]) * (len(ranking) - 1)).astype(int)
    categories = ["Low", "Moderate", "High"]
    cells = ranking.index[ranks].to_numpy()
    columns = ["y", "x", "longitude", "latitude", "site_id", "well_distance_km", "prediction_RMSE"]
    selected = table.loc["A3"].loc[cells, columns].copy()
    selected.insert(0, "sensitivity", categories)
    selected.insert(1, "target_quantile", [.10, .50, .90])
    selected.to_csv(output / "representative_locations.csv")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
    for ax, (cell, row) in zip(axes, selected.iterrows(), strict=True):
        data = dataset.sel(cell=cell)
        mask = data.evaluation_mask.astype(bool)
        ax.plot(data.time, data.observed_anomaly.where(mask), color=".45",
                linestyle="--", label="Nearest NWIS well")
        for scenario, color in zip(SCENARIOS, COLORS, strict=True):
            prediction = data.ensemble_anomaly.sel(scenario=scenario)
            ax.plot(data.time, prediction.where(mask), color=color,
                    label=f"{scenario} {SCENARIO_LABELS[scenario]}")
        ax.set(title=f"{row.sensitivity} · {row.latitude:.2f}°, {row.longitude:.2f}° · "
                     f"A3 vs A0 RMSE {row.prediction_RMSE:.3f}",
               ylabel="WTD anomaly")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=5, fontsize=9)
    axes[-1].set_xlabel("Month")
    save(fig, output / "representative_anomaly_series.png")


def write_results(path, summary, common):
    lines = [
        "# CONUS coordinate sensitivity results", "",
        f"Evaluation used {common.sum():,} common cell-months at "
        f"{common.any(axis=0).sum():,} cells. Each cell uses the nearest eligible NWIS well.", "",
        "| Scenario | Quantity | Cells | Median | Q25 | Q75 | IQR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.scenario} | {row.quantity} | {row.cells} | {row.median:.4g} | "
                     f"{row.q25:.4g} | {row.q75:.4g} | {row.iqr:.4g} |")
    lines += [
        "", "## Interpretation", "",
        "Negative changes in RMSE and MAE indicate better accuracy. Positive changes in Pearson "
        "correlation, KGE and NSE indicate better agreement with the observations.", "",
    ]
    for scenario in list(SCENARIOS)[1:]:
        data = summary.set_index(["scenario", "quantity"]).loc[scenario]
        changes = ", ".join(f"{metric} {data.loc[f'delta_{metric}', 'median']:+.4g}"
                            for metric in MAP_METRICS)
        distance = data.loc["prediction_RMSE"]
        lines.append(f"- **{scenario} ({SCENARIO_LABELS[scenario]})** had median score changes of "
                     f"{changes}. Its ensemble prediction anomaly RMSE versus A0 had a median "
                     f"of {distance['median']:.4g} and an interquartile range of "
                     f"{distance['q25']:.4g}–{distance['q75']:.4g} standardized anomaly units.")
    lines += [
        "", "Score changes measure accuracy relative to NWIS. Prediction anomaly RMSE measures "
        "the change in the modeled time series itself. Read it alongside the score changes and "
        "the representative series when judging robustness.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(run):
    output = run / "evaluation"
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    manifest = verify(run)
    paths = list(paths_for_evaluation(run, manifest))
    prepared = run / "prepared/A0"
    dates = pd.DatetimeIndex(np.load(prepared / "target_dates.npy", allow_pickle=False))
    selection = json.loads((prepared / "selection.json").read_text())
    if (selection["cadence"], selection["start"], selection["end"],
            str(dates[0].date()), str(dates[-1].date())) != (
        "monthly", MONTHLY_START, MONTHLY_END, MONTHLY_START, MONTHLY_END
    ):
        raise ValueError("Expected the existing CONUSmonthly observation and target period")
    _, starts, counts = np.unique(dates.to_period("M").asi8, return_index=True, return_counts=True)
    periods = dates[starts].to_period("M")
    pixels = np.load(prepared / "selected_pixels.npy", allow_pickle=False)
    static = np.load(prepared / STATIC, allow_pickle=False)
    observed, selected_wells = observed_cells(prepared, pixels, periods)
    obs = anomalies(observed, periods)
    locations = pd.DataFrame(dict(
        y=pixels[:, 0], x=pixels[:, 1], well_count=1,
        longitude=static[static_row("longitude")], latitude=static[static_row("latitude")],
        site_id=selected_wells.site_id.to_numpy(),
        well_distance_km=selected_wells.well_distance_km.to_numpy(),
        candidate_well_count=selected_wells.candidate_well_count.to_numpy(),
    ))
    locations.index.name = "cell"
    with TemporaryDirectory(dir=run, prefix=".evaluate-") as scratch:
        workspace = Path(scratch)
        results = workspace / "evaluation"
        results.mkdir()
        aggregates = []
        for scenario, (source, predictions) in zip(SCENARIOS, paths, strict=True):
            member_store = workspace / scenario
            member_store.mkdir()
            aggregates.append(aggregate_predictions(source, predictions, dates, periods, starts,
                                                    counts, len(pixels), member_store))
        predicted = np.stack([pred for _, pred, _, _ in aggregates])
        common = np.isfinite(obs) & np.isfinite(predicted).all(axis=0)
        if not common.any():
            raise ValueError("No valid cell-months are shared by all scenarios")
        masked_obs = np.where(common, obs, np.nan)
        scores = []
        for (_, pred, spread, members) in aggregates:
            scores.append(metrics(masked_obs, np.where(common, pred, np.nan), spread, members))
        # Each score uses the same population for all four distributions and paired changes.
        comparable = np.isfinite(np.stack([score[METRICS].to_numpy() for score in scores])).all(axis=0)
        comparisons = []
        pair_count = common.sum(axis=0)
        for score, pred in zip(scores, predicted, strict=True):
            values = score[METRICS].where(comparable)
            delta = (values - scores[0][METRICS]).add_prefix("delta_")
            squared = np.where(common, np.square(pred - predicted[0]), 0).sum(axis=0)
            prediction_rmse = np.sqrt(np.divide(squared, pair_count, where=pair_count > 0,
                                               out=np.full(len(pixels), np.nan)))
            comparisons.append(locations.join(values).join(delta).assign(
                paired_months=pair_count, prediction_RMSE=prediction_rmse))
        table = pd.concat(comparisons, keys=SCENARIOS, names=["scenario", "cell"])
        summary = summarize(table)
        table.to_csv(results / "cell_metrics_and_changes.csv")
        summary.to_csv(results / "metric_summary.csv", index=False)
        dataset = xr.Dataset(
            data_vars={
                "observed_wtd": (("time", "cell"), observed, {"units": "m"}),
                "observed_anomaly": (("time", "cell"), obs),
                "evaluation_mask": (("time", "cell"), common.astype(np.int8)),
                "ensemble_anomaly": (("scenario", "time", "cell"), predicted),
                "ensemble_mean_native_wtd": (("scenario", "time", "cell"),
                                             np.stack([raw for raw, _, _, _ in aggregates]), {"units": "m"}),
            },
            coords={"time": periods.start_time, "scenario": list(SCENARIOS), "cell": locations.index,
                    **{name: ("cell", locations[name].to_numpy()) for name in locations}},
            attrs={"members": len(manifest["models"]),
                   "observation_mapping": "Single nearest eligible well per assigned cell",
                   "anomaly_baseline": "Separate calendar-month z-scores, 2020–2024, ddof=0",
                   "evaluation_mask": "Finite observed anomalies and finite ensemble anomalies in all scenarios"},
        )
        dataset.to_netcdf(results / "evaluation_series.nc", engine="scipy")
        figures = results / "figures"
        figures.mkdir()
        plot_boxplots(figures, table)
        borders = country_borders(box(*MAP_EXTENT))
        for metric in MAP_METRICS:
            plot_changes(figures, table, metric, borders)
        plot_series(figures, dataset, table)
        write_results(results / "results.md", summary, common)
        results.rename(output)
    print(f"Evaluated {common.sum():,} common cell-months: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    evaluate(args.run.resolve())


if __name__ == "__main__":
    main()
