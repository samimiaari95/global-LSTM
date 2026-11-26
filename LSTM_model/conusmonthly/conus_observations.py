"""Download NWIS WTD and select a gap-free CONUS monthly anomaly window."""

import argparse
import getpass
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from LSTM_model.utils.file_io import save_csv, save_json


ROOT = Path("inputs/CONUS")
MONTHLY_INPUTS = Path("inputs/CONUSmonthly")
MONTHLY_GRID = MONTHLY_INPUTS / "era5land_grid"
MONTHLY_START = "2020-01-01"
MONTHLY_END = "2024-12-31"
FREQUENCIES = {"daily": "D", "weekly": "W-SUN", "monthly": "M"}
STATES = set("AL AZ AR CA CO CT DE DC FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT "
             "NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())
NWIS = dict(dataset="usgs_nwis", variable="water_table_depth", aggregation="mean")


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


def load_observations(folder, cadence, site_ids, days=None):
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
        blocks.append(frame if days is None else frame.reindex(days))
    if not blocks:
        return pd.DataFrame()
    data = pd.concat(blocks, axis=1, verify_integrity=True).sort_index()
    periods = data.index
    return data.replace([np.inf, -np.inf], np.nan).reindex(
        pd.period_range(periods.min(), periods.max(), freq=periods.freq))


def select(root, archive):
    # Keep the daily NWIS archive as the source used to optimize this window.
    daily = archive / "daily"
    if not (daily / "complete.json").is_file():
        raise FileNotFoundError(f"Daily observation download is incomplete in {daily}")
    sites = read_sites(daily / "sites.csv")
    sites = sites.loc[sites.state.isin(STATES)]
    days = pd.period_range(MONTHLY_START, MONTHLY_END, freq="D")
    months = days.asfreq("M").unique()
    samples = load_observations(daily, "daily", set(sites.index), days=days)
    if samples.empty:
        raise ValueError(f"No daily observation batches in {daily}")
    data = samples.groupby(samples.index.asfreq("M")).mean().reindex(months).dropna(axis=1)
    by_month = data.groupby(data.index.month)
    varying = (by_month.max() > by_month.min()).all(axis=0)
    data = data.loc[:, varying].sort_index(axis=1)
    if data.empty:
        raise ValueError("No wells have complete, varying monthly observations for the study window")
    winner = {
        "cadence": "monthly", "source": "daily_aggregation", "domain": "conus",
        "station_count": data.shape[1], "months": len(months), "days": len(days),
        "start": MONTHLY_START, "end": MONTHLY_END, "minimum_months": 60,
        "optimization": "Maximize native ERA5-Land cells with continuous monthly anomalies",
        "criterion": "Use wells with finite monthly means and variation in each calendar month",
    }
    selected_sites = sites.loc[data.columns].rename_axis("site_id")
    root.mkdir(parents=True, exist_ok=True)
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
    args = parser.parse_args()
    if args.stage == "download":
        download(args.root or ROOT)
    else:
        select(args.root or MONTHLY_INPUTS, args.archive or ROOT / "observations")


if __name__ == "__main__":
    main()
