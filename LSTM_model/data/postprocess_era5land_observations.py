"""Evaluate monthly standardized ERA5-Land ensemble anomalies on observation cells."""
import argparse
import json
import sys
from time import monotonic
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import shapefile
import xarray as xr
from shapely.geometry import box, shape

PROJECT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT))
from LSTM_model.utils.utils import utilities

REGION = "EU_74_-48_69_20"
PREP = PROJECT / "inputs" / REGION / "era5land_inputs_1996_2012"
VALIDATION = PROJECT / "outputs" / REGION / "validation_ERA5"
ENSEMBLE = VALIDATION / "ensemble_400px"
NAME = "sim_ERA5on_observed_valid_pixels_1996_2012.npy"
MEMBERS = 100
METRICS = ["RMSE", "MAE", "Pearson_r", "NSE", "KGE"]
STATS = ["Ensemble_variance", "IQR", "Pairwise_r", "Ensemble_SD", "MAD"]


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
    for member in range(MEMBERS):
        path = ENSEMBLE / f"400px_member_{member}" / NAME
        info = json.loads(path.with_suffix(".json").read_text())
        daily = np.load(path, mmap_mode="r")
        if (info["member"] != member or daily.shape != (len(dates), ncell)
                or list(daily.shape) != info["shape"]
                or Path(info["target_dates_file"]).resolve() != PREP / "target_dates.npy"
                or Path(info["selected_pixels_file"]).resolve() != PREP / "selected_pixels.npy"):
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


def prediction_r2(observed, predicted):
    if len(observed) < 2 or np.ptp(observed) == 0:
        return np.nan
    return 1 - np.sum((predicted - observed)**2) / np.sum((observed - observed.mean())**2)


def ensemble_statistics(block):
    """Summarize the spread and agreement of member series (member, time)."""
    q25, q75 = np.percentile(block, [25, 75], axis=0)
    stats = dict(Ensemble_variance=block.var(axis=0).mean(), IQR=(q75-q25).mean(),
                 Ensemble_SD=block.std(axis=0).mean(), MAD=np.abs(block-block.mean(axis=0)).mean())
    if block.shape[1] >= 2 and np.all(np.ptp(block, axis=1) > 0):
        corr = np.corrcoef(block)
        stats["Pairwise_r"] = corr[np.triu_indices(len(block), k=1)].mean()
    return stats


def metrics(obs, pred, spread, members):
    rows = []
    utility = utilities()
    for i, ensemble in enumerate(members):
        paired = np.isfinite(obs[:, i]) & np.isfinite(pred[:, i])
        o, p = obs[paired, i], pred[paired, i]
        row = dict(paired_months=len(o))
        if len(o):
            varying = min(np.ptp(o), np.ptp(p)) > 0
            row.update(RMSE=np.sqrt(np.mean((p-o)**2)), MAE=np.abs(p-o).mean(),
                       NSE=prediction_r2(o, p), Pearson_r=np.corrcoef(o, p)[0, 1] if varying else np.nan,
                       KGE=utility.calculate_kge(o, p) if varying else np.nan,
                       obs_SD=o.std(), prediction_SD=p.std(),
                       RMS_spread=np.sqrt(np.mean(spread[paired, i]**2)))
            row.update(ensemble_statistics(ensemble[:, paired].astype(float)))
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=["paired_months", *METRICS, *STATS,
                                               "obs_SD", "prediction_SD", "RMS_spread"])


def number(value):
    return f"{value:.3f}" if np.isfinite(value) else "undefined"


def save(fig, path, dpi=145):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def maps(out, cells, *, full=None, zoom=(-6.5, 41.0, 17.5, 56.25),
         zoom_title="England and western–central Europe", coverage="paired_months",
         metric_labels=None, statistic_labels=None, color_scales=None):
    x, y = cells.longitude, cells.latitude
    metric_labels = metric_labels or {key: key for key in METRICS}
    statistic_labels = statistic_labels or {}
    color_scales = color_scales or {}
    full = full or (x.min()-2, y.min()-1, x.max()+2, y.max()+1)
    bounds = box(*full) if zoom is None else box(*full).union(box(*zoom)).envelope
    borders = []
    shp = PROJECT.parent / "ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    with shapefile.Reader(str(shp)) as reader:
        for feature in reader.iterShapes(bbox=bounds.bounds):
            lines = shape(feature.__geo_interface__).boundary.simplify(.03).intersection(bounds)
            for line in getattr(lines, "geoms", [lines]):
                if line.geom_type == "LineString" and not line.is_empty:
                    borders.append(np.asarray(line.coords))

    def draw(ax, key, extent, title):
        ax.add_collection(LineCollection(borders, colors=".65", linewidths=.45))
        values = cells[key].dropna()
        signed = key in ("Pearson_r", "NSE", "KGE")
        norm = Normalize(-1, 1) if signed else None
        if len(values) and not signed:
            norm = Normalize(values.min(), values.quantile(.98))
        cmap, norm = color_scales.get(key, ("RdBu_r" if signed else "viridis", norm))
        if isinstance(norm, LogNorm):
            values = values[values > 0]
        points = ax.scatter(x[values.index], y[values.index], c=values,
                            cmap=cmap, norm=norm,
                            s=6, linewidths=0)
        west, south, east, north = extent
        ax.set(title=title, xlabel="Longitude (°E)", ylabel="Latitude (°N)",
               xlim=(west, east), ylim=(south, north))
        ax.set_aspect(1 / np.cos(np.deg2rad(y.median())))
        return points

    folder = out / "metric_maps"
    folder.mkdir()
    for key, label in metric_labels.items():
        if zoom is None:
            fig, axes = plt.subplots(figsize=(9, 5), constrained_layout=True)
            points = draw(axes, key, full, "Full study area")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
            draw(axes[0], key, full, "Full study area")
            points = draw(axes[1], key, zoom, zoom_title)
        fig.colorbar(points, ax=axes, shrink=.7, label=label, extend="both")
        fig.suptitle(label)
        save(fig, folder / f"{key}.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, key in zip(axes.flat, [coverage, "RMS_spread", "obs_SD", "prediction_SD"]):
        fig.colorbar(draw(ax, key, full, statistic_labels.get(key, key)), ax=ax, shrink=.7)
    save(fig, out / "statistics_maps.png")


def paired_plot(ax, obs, pred, score, pooled=False, period="month",
                value_label="standardized anomaly"):
    if len(obs):
        if pooled:
            density = ax.hexbin(obs, pred, gridsize=60, bins="log", mincnt=1)
            ax.figure.colorbar(density, ax=ax, label=f"Cell-{period} count")
        else:
            ax.scatter(obs, pred, s=10, alpha=.55, linewidths=0)
        lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
        ax.set_aspect("equal", adjustable="box")
    ax.text(.025, .975, f"R² = {number(score)}   n = {len(obs):,}",
            transform=ax.transAxes, va="top", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    ax.set(xlabel=f"Observed {value_label}", ylabel=f"Ensemble mean {value_label}")


def plot_members(ax, dates, members):
    lines = ax.plot(dates, members.T, color=".45", alpha=.23, lw=.55)
    lines[0].set_label("Individual members")


def cell_figures(out, cells, dates, obs, pred, members, indices, *,
                 metric_labels=None, value_label="standardized anomaly"):
    metric_labels = metric_labels or {key: key for key in METRICS}
    series_label = "Std. anomaly" if value_label == "standardized anomaly" else value_label
    ts = out / "timeseries"
    scatter = out / "cell_r2"
    ts.mkdir()
    scatter.mkdir()
    for i in indices:
        row = cells.loc[i]
        name = f"cell_{i:04d}.png"
        title = f"Cell {i}   ({row.latitude:.2f}°, {row.longitude:.2f}°)"
        fig, ax = plt.subplots(figsize=(9, 3.4))
        fig.subplots_adjust(top=.78, bottom=.18)
        plot_members(ax, dates, members[i])
        ax.plot(dates, obs[:, i], label="Observation", color="#165e91", lw=1.3)
        ax.plot(dates, pred[:, i], label="Ensemble mean", color="#ae4120", lw=1.4)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=3, frameon=False)
        ax.set_ylabel(series_label)
        ax.set_title("    ".join(f"{label} {number(row[key])}" for key, label in metric_labels.items()),
                     fontsize=9)
        fig.suptitle(title)
        save(fig, ts / name, dpi=110)

        paired = np.isfinite(obs[:, i]) & np.isfinite(pred[:, i])
        fig, ax = plt.subplots(figsize=(5, 4.3))
        fig.subplots_adjust(left=.17, bottom=.14)
        paired_plot(ax, obs[paired, i], pred[paired, i], row.NSE, value_label=value_label)
        ax.set_title(title)
        save(fig, scatter / name, dpi=110)


def scale_metric_axis(ax, direction, values, scale):
    if scale == "log":
        getattr(ax, f"set_{direction}scale")("log")
    elif scale == "symlog":
        getattr(ax, f"set_{direction}scale")("symlog", linthresh=1)
        getattr(ax, f"set_{direction}lim")(min(-1, values.min()), 1)
        getattr(ax, f"{direction}axis").get_major_locator().set_params(numticks=5)


def figures(out, cells, periods, obs, pred, members, indices, *, period="month", map_options=None,
            metric_labels=None, statistic_labels=None, value_label="standardized anomaly",
            paired_filename="paired_anomalies.png", axis_scales=None, show_example_members=False):
    metric_labels = metric_labels or {key: key for key in METRICS}
    statistic_labels = statistic_labels or {}
    axis_scales = axis_scales or {}
    maps(out, cells, metric_labels=metric_labels, statistic_labels=statistic_labels,
         **(map_options or {}))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, metric_labels.items()):
        v = np.sort(cells[key].dropna())
        fraction = np.arange(1, len(v)+1) / max(len(v), 1)
        shown = v > 0 if axis_scales.get(key) == "log" else slice(None)
        ax.step(v[shown], fraction[shown], where="post")
        ax.set(xlabel=label, ylabel="Fraction of cells", ylim=(0, 1.02))
        scale_metric_axis(ax, "x", cells[key], axis_scales.get(key))
    for ax in axes.flat[len(metric_labels):]:
        ax.remove()
    save(fig, out / "metric_cdfs.png")

    fig, axes = plt.subplots(len(metric_labels), len(STATS), figsize=(16, 14),
                             constrained_layout=True, squeeze=False)
    for row, (key, label) in zip(axes, metric_labels.items()):
        for ax, stat in zip(row, STATS):
            pairs = cells[[stat, key]].dropna()
            for column in (stat, key):
                if axis_scales.get(column) == "log":
                    pairs = pairs.loc[pairs[column] > 0]
            ax.scatter(pairs[stat], pairs[key], s=3, alpha=.35)
            ax.set(xlabel=statistic_labels.get(stat, stat.replace("_", " ")), ylabel=label)
            scale_metric_axis(ax, "x", cells[stat], axis_scales.get(stat))
            scale_metric_axis(ax, "y", cells[key], axis_scales.get(key))
    save(fig, out / "statistics_vs_metrics.png")

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    paired = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[paired], pred[paired]
    score = prediction_r2(o, p)
    paired_plot(ax, o, p, score, pooled=True, period=period, value_label=value_label)
    ax.set_title(f"All paired cell-{period}s")
    save(fig, out / paired_filename)
    print(f"Global R² = {number(score)}", flush=True)

    ranked = cells.RMSE.dropna().sort_values()
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), constrained_layout=True)
    dates = periods.to_timestamp() if isinstance(periods, pd.PeriodIndex) else periods
    series_label = "Std. anomaly" if value_label == "standardized anomaly" else value_label
    for ax, q in zip(axes, [.1, .5, .9]):
        i = ranked.index[round(q*(len(ranked)-1))]
        if show_example_members:
            plot_members(ax, dates, members[i])
        ax.plot(dates, obs[:, i], label="Observation", lw=1)
        ax.plot(dates, pred[:, i], label="Ensemble mean", lw=1)
        ax.set(title=f"Cell {i} · {q:.0%} RMSE rank · {metric_labels['RMSE']} {ranked[i]:.2f}",
               ylabel=series_label)
    axes[0].legend(ncol=3 if show_example_members else 2)
    save(fig, out / "example_series.png")
    start = monotonic()
    cell_figures(out, cells, dates, obs, pred, members, indices, metric_labels=metric_labels,
                 value_label=value_label)
    print(f"Rendered both figures for {len(indices)} cells in {monotonic()-start:.1f} s", flush=True)


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
        figures(result, cells, months, obs, pred, members, indices)
        result.rename(out)
    print(f"Saved {len(cells)} cells and {int(cells.paired_months.sum())} paired cell-months to {out}")


if __name__ == "__main__":
    main()
