"""Statistics and plots for evaluating ensemble predictions at observation cells."""

from pathlib import Path
from time import monotonic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import box, shape

from LSTM_model.utils.utils import utilities

PROJECT = Path(__file__).resolve().parents[2]
MEMBERS = 100
METRICS = ["RMSE", "MAE", "Pearson_r", "NSE", "KGE"]
STATS = ["Ensemble_variance", "IQR", "Pairwise_r", "Ensemble_SD", "MAD"]


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


def country_borders(bounds):
    borders = []
    shp = PROJECT.parent / "ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
    with shapefile.Reader(str(shp)) as reader:
        for feature in reader.iterShapes(bbox=bounds.bounds):
            lines = shape(feature.__geo_interface__).boundary.simplify(.03).intersection(bounds)
            for line in getattr(lines, "geoms", [lines]):
                if line.geom_type == "LineString" and not line.is_empty:
                    borders.append(np.asarray(line.coords))
    return borders


def spatial_axes(ax, borders, extent):
    ax.add_collection(LineCollection(borders, colors=".65", linewidths=.45))
    west, south, east, north = extent
    ax.set(xlim=(west, east), ylim=(south, north))
    ax.set_aspect(1 / np.cos(np.deg2rad((south + north) / 2)))


def cell_location(ax, row, context):
    borders, extent = context
    spatial_axes(ax, borders, extent)
    ax.scatter(row.longitude, row.latitude, s=23, color="#ae4120",
               edgecolor="white", linewidth=.6, zorder=3)
    ax.set(title="Cell location", xticks=[], yticks=[])


def pooled_rank_counts(obs, members):
    """Count observation ranks across complete cell-months, with seeded random ties."""
    member_count = members.shape[1]
    counts = np.zeros(member_count + 1, dtype=np.int64)
    rng = np.random.default_rng(0)
    for observed, ensemble in zip(obs.T, members):
        valid = np.isfinite(observed) & np.isfinite(ensemble).all(axis=0)
        observed = observed[valid]
        ensemble = ensemble[:, valid]
        lower = (ensemble < observed).sum(axis=0)
        tied = (ensemble == observed).sum(axis=0)
        ranks = lower + rng.integers(tied + 1)
        counts += np.bincount(ranks, minlength=len(counts))
    return counts


def rank_histogram(out, obs, members):
    counts = pooled_rank_counts(obs, members)
    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    ranks = np.arange(len(counts))
    ax.bar(ranks, counts, color="#165e91")
    ax.axhline(counts.mean(), color=".25", linestyle="--", label="Uniform reference")
    ax.set(title=f"Pooled ensemble rank histogram ({counts.sum():,} cell-months)",
           xlabel="Observation rank (0–100)",
           ylabel="Cell-month count", xlim=(-1, len(counts)))
    ax.legend()
    save(fig, out / "rank_histogram.png")


def maps(out, cells, *, full=None, zoom=(-6.5, 41.0, 17.5, 56.25),
         zoom_title="England and western–central Europe", coverage="paired_months",
         metric_labels=None, statistic_labels=None, color_scales=None):
    x, y = cells.longitude, cells.latitude
    metric_labels = metric_labels or {key: key for key in METRICS}
    statistic_labels = statistic_labels or {}
    color_scales = color_scales or {}
    full = full or (x.min()-2, y.min()-1, x.max()+2, y.max()+1)
    bounds = box(*full) if zoom is None else box(*full).union(box(*zoom)).envelope
    borders = country_borders(bounds)

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
                 metric_labels=None, value_label="standardized anomaly", location_map=None,
                 series_label=None):
    metric_labels = metric_labels or {key: key for key in METRICS}
    series_label = series_label or ("Std. anomaly" if value_label == "standardized anomaly" else value_label)
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
        if location_map is not None:
            fig.subplots_adjust(right=.77)
            cell_location(fig.add_axes([.79, .19, .19, .56]), row, location_map)
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
        if location_map is not None:
            fig.set_size_inches(7.5, 4.3)
            fig.subplots_adjust(left=.12, right=.70)
            cell_location(fig.add_axes([.73, .19, .24, .56]), row, location_map)
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
            paired_filename="paired_anomalies.png", axis_scales=None, show_example_members=False,
            scatter_minimums=None, location_map=None, series_label=None):
    metric_labels = metric_labels or {key: key for key in METRICS}
    statistic_labels = statistic_labels or {}
    axis_scales = axis_scales or {}
    scatter_minimums = scatter_minimums or {}
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
            minimum = scatter_minimums.get(key)
            if minimum is not None:
                pairs = pairs.loc[pairs[key] >= minimum]
            ax.scatter(pairs[stat], pairs[key], s=3, alpha=.35)
            ax.set(xlabel=statistic_labels.get(stat, stat.replace("_", " ")), ylabel=label)
            scale_metric_axis(ax, "x", cells[stat], axis_scales.get(stat))
            scale_metric_axis(ax, "y", cells[key], axis_scales.get(key))
            if minimum is not None:
                ax.set_ylim(minimum, 1.02)
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
    series_label = series_label or ("Std. anomaly" if value_label == "standardized anomaly" else value_label)
    for ax, q in zip(axes, [.1, .5, .9]):
        i = ranked.index[round(q*(len(ranked)-1))]
        if location_map is not None:
            cell_location(ax.inset_axes([1.025, .05, .17, .85]), cells.loc[i], location_map)
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
                 value_label=value_label, location_map=location_map, series_label=series_label)
    print(f"Rendered both figures for {len(indices)} cells in {monotonic()-start:.1f} s", flush=True)
