"""Download NWIS WTD and select a gap-free CONUS monthly anomaly window."""

import argparse
import getpass
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("inputs/CONUS")
CONUS2_RUN = ROOT / "monthly_anomaly_run"
ERA5LAND_RUN = CONUS2_RUN / "era5land_grid"
FREQUENCIES = {"daily": "D", "weekly": "W-SUN", "monthly": "M"}
STATES = set("AL AZ AR CA CO CT DE DC FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT "
             "NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())
NWIS = dict(dataset="usgs_nwis", variable="water_table_depth", aggregation="mean")


def save_csv(data, path, **kwargs):
    temporary = path.with_name(f".{path.name}.partial")
    data.to_csv(temporary, **kwargs)
    temporary.replace(path)


def save_json(data, path):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def connect_hydrodata():
    import hf_hydrodata as hf

    pin = os.environ.get("CONUS_HYDRODATA_PIN") or getpass.getpass("HydroData PIN: ")
    hf.register_api_pin(email="miaarisami@gmail.com", pin=pin)
    return hf


def read_sites(path):
    return pd.read_csv(path, dtype={"site_id": str}).set_index("site_id", verify_integrity=True)


def download(root):
    cache = root / "observations"
    cache.mkdir(parents=True, exist_ok=True)
    hf = None

    def client():
        nonlocal hf
        if hf is None:
            hf = connect_hydrodata()
        return hf

    for cadence in FREQUENCIES:
        folder = cache / cadence
        folder.mkdir(exist_ok=True)
        marker = folder / "complete.json"
        if marker.is_file():
            continue
        try:
            download_cadence(client, cadence, folder)
        except ValueError as error:
            message = str(error).lower()
            unsupported = ("temporal_resolution" in message
                           and ("unexpected value" in message or "not currently supported" in message))
            unavailable = message == f"no conus station metadata returned for {cadence}"
            if cadence == "daily" or not (unsupported or unavailable):
                raise
            print(f"NWIS point API does not supply {cadence} WTD. Will aggregate daily WTD.",
                  flush=True)
            save_json({"source": "daily_aggregation", "reason": str(error)}, marker)


def download_cadence(client, cadence, folder):
    options = dict(NWIS, temporal_resolution=cadence)
    metadata = folder / "sites.csv"
    if not metadata.exists():
        print(f"Downloading {cadence} NWIS metadata", flush=True)
        sites = client().get_point_metadata(options)
        sites = sites.loc[sites.state.isin(STATES)].sort_values("site_id")
        if sites.empty:
            raise ValueError(f"No CONUS station metadata returned for {cadence}")
        save_csv(sites, metadata, index=False)
    sites = read_sites(metadata)
    first = pd.to_datetime(sites.first_date_data_available)
    last = pd.to_datetime(sites.last_date_data_available)
    if first.isna().any() or last.isna().any() or (first > last).any():
        raise ValueError(f"Invalid WTD coverage dates in {metadata}")
    pending = {state: folder / f"sites_{state}.csv.gz" for state in sorted(sites.state.unique())}
    pending = {state: path for state, path in pending.items() if not path.exists()}
    if pending:
        hf = client()

        def fetch_batch(state, path):
            batch = sites.index[sites.state == state]
            print(f"Downloading {cadence} WTD history for {state} ({len(batch)} stations)",
                  flush=True)
            data = hf.get_point_data(**options, state=state)
            if "date" not in data or set(data.columns) - {"date", *batch}:
                raise ValueError(f"Unexpected NWIS response for {path}")
            save_csv(data, path, index=False, compression="gzip")

        with ThreadPoolExecutor(max_workers=3) as pool:
            requests = [pool.submit(fetch_batch, state, path) for state, path in pending.items()]
            try:
                for request in requests:
                    request.result()
            finally:
                for request in requests:
                    request.cancel()
    save_json({"source": "native", "first_year": first.min().year, "last_year": last.max().year},
              folder / "complete.json")


def load_observations(folder, cadence, site_ids):
    blocks = []
    for path in sorted(folder.glob("sites_*.csv.gz")):
        frame = pd.read_csv(
            path, usecols=lambda col: col == "date" or col in site_ids,
            index_col="date", dtype=defaultdict(lambda: np.float32, {"date": str}),
        )
        if frame.empty:
            continue
        periods = pd.DatetimeIndex(pd.to_datetime(frame.index)).to_period(FREQUENCIES[cadence])
        if not periods.is_unique or periods.hasnans:
            raise ValueError(f"Invalid or repeated {cadence} dates in {path}")
        frame.index = periods
        blocks.append(frame)
    if not blocks:
        return pd.DataFrame()
    data = pd.concat(blocks, axis=1, verify_integrity=True).sort_index()
    periods = data.index
    return data.replace([np.inf, -np.inf], np.nan).reindex(
        pd.period_range(periods.min(), periods.max(), freq=periods.freq))


def best_window(data, minimum_start="1951-01-01", latest_end=None):
    """Maximize stations with a defined calendar-month z-score throughout."""
    if data.empty:
        return None
    periods = data.index
    eligible = periods.start_time >= pd.Timestamp(minimum_start)
    if latest_end is not None:
        eligible &= periods.end_time.normalize() <= pd.Timestamp(latest_end)
    periods = periods[eligible]
    values = data.loc[eligible].to_numpy()
    month_count, station_count = values.shape
    if month_count < 24:
        return None

    # For each station and start, a missing month caps the possible window.
    # A different value in the same calendar month makes its z-score defined.
    next_gap = np.full((month_count + 1, station_count), month_count, dtype=np.int32)
    next_change = np.full(values.shape, month_count, dtype=np.int32)
    for i in range(month_count - 1, -1, -1):
        next_gap[i] = np.where(np.isfinite(values[i]), next_gap[i + 1], i)
        if i + 12 < month_count:
            next_change[i] = np.where(values[i] != values[i + 12], i + 12, next_change[i + 12])

    best = None
    for start in range(month_count - 23):
        first_end = np.maximum(start + 23, next_change[start:start + 12].max(axis=0))
        stop = next_gap[start]
        valid = first_end < stop
        if not valid.any():
            continue
        added = np.bincount(first_end[valid], minlength=month_count + 1)
        removed = np.bincount(stop[valid], minlength=month_count + 1)
        counts = np.cumsum(added - removed)[:month_count]
        end = month_count - 1 - counts[::-1].argmax()
        score = (int(counts[end]), end - start + 1, -start)
        if best is None or score > best[0]:
            best = score, periods[start], periods[end]
    return best


def eligible_observations(folder, cadence):
    sites = read_sites(folder / "sites.csv")
    sites = sites.loc[sites.state.isin(STATES)]
    return sites, load_observations(folder, cadence, set(sites.index))


def selection_data(cache):
    monthly = cache / "monthly"
    marker = monthly / "complete.json"
    if marker.is_file() and json.loads(marker.read_text()).get("source", "native") == "native":
        sites, data = eligible_observations(monthly, "monthly")
        return sites, data, "native"
    daily = cache / "daily"
    if not (daily / "complete.json").is_file():
        raise FileNotFoundError(f"Daily observation download is incomplete in {daily}")
    sites, data = eligible_observations(daily, "daily")
    if not data.empty:
        data = data.groupby(data.index.asfreq("M")).mean()
    return sites, data, "daily_aggregation"


def select(root, latest_end, archive=None):
    sites, data, source = selection_data(archive or root / "observations")
    result = best_window(data, latest_end=latest_end)
    if result is None:
        raise ValueError("No window of at least 24 months with complete monthly anomalies")
    score, first, last = result
    winner = {
        "cadence": "monthly", "source": source, "domain": "conus", "station_count": score[0],
        "months": int(score[1]), "days": (last.end_time.date() - first.start_time.date()).days + 1,
        "start": str(first.start_time.date()), "end": str(last.end_time.date()),
        "latest_end_cutoff": latest_end,
        "criterion": "gap-free monthly observations with nonzero variance in every calendar month",
    }
    data = data.loc[first:last].dropna(axis=1)
    by_month = data.groupby(data.index.month)
    varying = (by_month.max() > by_month.min()).all(axis=0)
    data = data.loc[:, varying].sort_index(axis=1)
    if data.shape[1] != winner["station_count"]:
        raise ValueError("Selected station count differs from window search")
    selected_sites = sites.loc[data.columns].rename_axis("site_id")
    root.mkdir(parents=True, exist_ok=True)
    save_csv(pd.DataFrame([winner]), root / "candidate_windows.csv", index=False)
    data.index = data.index.start_time.rename("date")
    save_csv(data, root / "selected_wtd.csv")
    save_csv(selected_sites, root / "selected_stations.csv")
    save_json(winner, root / "selection.json")
    print(winner, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "select"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--archive", type=Path, help="Existing observation archive for selection")
    parser.add_argument(
        "--latest-end", help="Last date with complete ERA5-Land coverage (default: six UTC days ago)")
    args = parser.parse_args()
    if args.stage == "download":
        download(args.root or ROOT)
    else:
        latest_end = args.latest_end or str(
            (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=6)).date())
        archive = args.archive or (ROOT / "observations" if args.root is None else None)
        select(args.root or CONUS2_RUN, latest_end, archive)


if __name__ == "__main__":
    main()
