#!/usr/bin/env python3
"""Ingestion CLI.

    python ingest.py demo                 # synthetic data, no key needed
    python ingest.py backfill             # full vintage history from ALFRED
    python ingest.py sync                 # current values only, for daily runs
    python ingest.py coverage             # what you actually have

Backfill is slow and only needs running once per series. Sync is what you put
on a schedule. With Macrobond COM that schedule has to be a Windows machine
with the desktop app installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

from src.drivers import load_config, required_series
from src.store import Store


def cmd_backfill(args):
    from src.sources.fred import FredSource

    cfg = load_config(args.config)
    store = Store(args.db)
    source = FredSource()
    series = required_series(cfg)

    print(f"Backfilling {len(series)} series with full vintage history.")
    for i, sid in enumerate(series, 1):
        try:
            df = source.fetch_with_vintages(sid)
            n = store.upsert_observations(df)
            meta = source.describe(sid)
            store.record_meta(sid, "fred", **meta)
            vintages = df["vintage_date"].nunique() if not df.empty else 0
            print(f"  [{i}/{len(series)}] {sid:<14} {n:>7} rows  "
                  f"{vintages:>4} vintages  {meta['title'][:44]}")
        except Exception as exc:
            print(f"  [{i}/{len(series)}] {sid:<14} FAILED: {exc}")
    store.close()


def cmd_sync(args):
    from src.sources.fred import FredSource

    cfg = load_config(args.config)
    store = Store(args.db)
    source = FredSource()
    total = 0
    for sid in required_series(cfg):
        try:
            total += store.upsert_observations(source.fetch_current(sid))
        except Exception as exc:
            print(f"  {sid}: {exc}", file=sys.stderr)
    print(f"Synced {total} observations at {dt.date.today()}.")
    store.close()


def cmd_coverage(args):
    store = Store(args.db)
    cov = store.coverage()
    if cov.empty:
        print("Store is empty. Run `python ingest.py demo` or `backfill`.")
    else:
        print(cov.to_string(index=False))
        thin = cov[cov["vintages"] <= 1]["series_id"].tolist()
        if thin:
            print("\nNo revision history, so backtests on these are "
                  "as-revised, not point-in-time:")
            print("  " + ", ".join(thin))
    store.close()


def cmd_demo(args):
    store = Store(args.db)
    n, series = load_demo(store, load_config(args.config))
    store.close()
    print(f"Loaded {n} synthetic observations across {series} series.")
    print("Now run: streamlit run app.py")


def load_demo(store: Store, cfg: dict) -> tuple[int, int]:
    """Synthetic series with plausible dynamics and fake vintages.

    Exists so you can see the dashboard work end to end today. The shapes are
    invented. Do not read anything into them.
    """
    rng = np.random.default_rng(11)
    idx = pd.date_range("1985-01-31", "2026-08-31", freq="ME")

    anchors = {
        "PCEC96": (11000, 0.0021), "RSAFS": (450000, 0.0033),
        "PAYEMS": (95000, 0.0013), "UNRATE": (5.5, None),
        "NROU": (4.6, None), "T5YIFR": (2.3, None), "T10YIE": (2.25, None),
        "MICH": (3.1, None), "EXPINF10YR": (2.1, None), "TCU": (79.0, None),
        "JTSJOL": (6500, None), "UNEMPLOY": (7500, None), "ULCNFB": (100, 0.0018),
        "LNS12300060": (79.5, None), "FEDFUNDS": (3.0, None),
        "PCEPILFE": (80, 0.0017), "NFCI": (-0.2, None),
        "FYFSGDA188S": (-3.5, None), "NEWORDER": (55000, 0.0025),
        "PNFIC1": (1800, 0.0028), "TLMFGCONS": (40000, 0.0035),
        "DRTSCILM": (5.0, None),
    }

    rows = []
    for sid in required_series(cfg):
        base, drift = anchors.get(sid, (100.0, 0.001))
        if drift:
            shock = rng.normal(drift, drift * 1.6, len(idx))
            values = base * np.exp(np.cumsum(shock))
        else:
            cycle = np.sin(np.arange(len(idx)) / 34) * abs(base) * 0.18
            values = base + cycle + np.cumsum(rng.normal(0, abs(base) * 0.012,
                                                         len(idx)))
        # One revision per observation, published with a realistic lag.
        for d, v in zip(idx, values):
            first = (d + pd.Timedelta(days=35)).date()
            rows.append((sid, d.date(), first, float(v)))
            if drift and rng.random() < 0.35:
                rows.append((sid, d.date(),
                             (d + pd.Timedelta(days=400)).date(),
                             float(v) * (1 + rng.normal(0, 0.004))))
        store.record_meta(sid, "demo", title=f"Synthetic {sid}",
                          frequency="M", has_vintages=True)

    df = pd.DataFrame(rows, columns=["series_id", "observation_date",
                                     "vintage_date", "value"])
    n = store.upsert_observations(df)
    return n, df["series_id"].nunique()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/indicators.yml")
    p.add_argument("--db", default=os.environ.get("MACRO_REGIME_DB",
                                                  "data/regime.duckdb"))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("backfill", cmd_backfill), ("sync", cmd_sync),
                     ("coverage", cmd_coverage), ("demo", cmd_demo)]:
        sub.add_parser(name).set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
