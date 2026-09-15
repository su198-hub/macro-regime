#!/usr/bin/env python3
"""Ingestion CLI.

    python ingest.py demo                 # synthetic data, no key needed
    python ingest.py backfill             # full vintage history
    python ingest.py sync                 # refresh, for scheduled runs
    python ingest.py coverage             # what you actually have
    python ingest.py publish              # push a snapshot for the hosted app

Which vendor each series comes from is set in config/sources.yml; pass
--source to override for a run. Point --db (or MACRO_REGIME_DB) at a file that
holds only real data: backfill refuses to write into a demo store.

FRED backfill is slow and only needs running once; FRED sync pulls current
values. Macrobond returns a series' full revision history in about a second,
so its sync simply refetches it. With Macrobond COM the schedule has to be a
Windows machine with the desktop app installed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import yaml

from src.drivers import load_config, required_series
from src.store import Store


def load_sources(path: str) -> dict:
    if not os.path.exists(path):
        return {"default": "fred", "series": {}}
    with open(path) as f:
        return yaml.safe_load(f) or {"default": "fred", "series": {}}


def route(sid: str, sources: dict, override: str | None) -> tuple[str, str]:
    """(vendor, vendor's code) for one series name from indicators.yml."""
    entry = (sources.get("series") or {}).get(sid) or {}
    vendor = override or entry.get("use") or sources.get("default", "fred")
    code = sid if vendor == "fred" else entry.get(vendor)
    if not code:
        raise KeyError(f"no {vendor} code for {sid} in config/sources.yml")
    return vendor, code


def open_vendor(vendor: str):
    if vendor == "macrobond":
        from src.sources.macrobond import MacrobondSource
        return MacrobondSource()
    from src.sources.fred import FredSource
    return FredSource()


def refuse_demo_store(store: Store, db: str) -> None:
    if "demo" in store.sources():
        store.close()
        sys.exit(f"{db} holds synthetic demo data. Real data goes in its own file, "
                 f"for example:\n  python ingest.py --db "
                 f"\"%LOCALAPPDATA%\\macro-regime\\macrobond.duckdb\" backfill")


def _pull(args, full_history: bool):
    cfg = load_config(args.config)
    sources = load_sources(args.sources)
    store = Store(args.db)
    refuse_demo_store(store, args.db)
    series = required_series(cfg)
    vendors: dict[str, object] = {}

    verb = "Backfilling" if full_history else "Syncing"
    print(f"{verb} {len(series)} series into {args.db}.")
    failed = 0
    for i, sid in enumerate(series, 1):
        try:
            vendor, code = route(sid, sources, args.source)
            if vendor not in vendors:
                vendors[vendor] = open_vendor(vendor)
            source = vendors[vendor]
            # Macrobond's full history is one fast call, so it always refetches.
            if full_history or vendor == "macrobond":
                df = source.fetch_with_vintages(code)
            else:
                df = source.fetch_current(code)
            df = df.assign(series_id=sid)
            n = store.upsert_observations(df)
            meta = source.describe(code)
            store.record_meta(sid, vendor, source_code=code, **meta)
            vintages = df["vintage_date"].nunique() if not df.empty else 0
            print(f"  [{i:>2}/{len(series)}] {sid:<12} {vendor}:{code:<20} {n:>7} rows "
                  f"{vintages:>5} vintages  {str(meta['title'])[:50]}")
        except Exception as exc:
            failed += 1
            print(f"  [{i:>2}/{len(series)}] {sid:<12} FAILED: {exc}", file=sys.stderr)
    for source in vendors.values():
        if hasattr(source, "close"):
            source.close()
    store.close()
    print(f"Done at {dt.datetime.now():%Y-%m-%d %H:%M}. {failed} failed.")
    if failed:
        sys.exit(1)


def cmd_backfill(args):
    _pull(args, full_history=True)


def cmd_sync(args):
    _pull(args, full_history=False)


def cmd_coverage(args):
    store = Store(args.db)
    cov = store.coverage()
    if cov.empty:
        print("Store is empty. Run `python ingest.py demo` or `backfill`.")
    else:
        meta = store.con.execute(
            "SELECT series_id, source, source_code FROM series_meta").df()
        cov = meta.merge(cov, on="series_id", how="right")
        print(cov.to_string(index=False))
        thin = cov[cov["vintages"] <= 1]["series_id"].tolist()
        if thin:
            print("\nNo revision history, so backtests on these are "
                  "as-revised, not point-in-time:")
            print("  " + ", ".join(thin))
    store.close()


DATA_BRANCH = "data"
DATA_README = """# Data snapshot

Published by `python ingest.py publish` from the machine that holds the
Macrobond connection. The hosted app reads it via MACRO_REGIME_DATA_URL.
This branch is replaced on every publish, so it never accumulates history.

- observations.parquet: series_id, observation_date, vintage_date, value
- series_meta.parquet: source, vendor code, title, units, frequency
- manifest.json: when it was published and what it holds
"""


def cmd_publish(args):
    """Export the store and force-push it to the repository's data branch."""
    import shutil
    import subprocess
    import tempfile

    from src.snapshot import export_snapshot

    store = Store(args.db)
    if store.coverage().empty:
        sys.exit(f"{args.db} is empty. Run backfill first.")
    if "demo" in store.sources():
        sys.exit(f"{args.db} holds demo data; refusing to publish it as real data.")

    git = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    root = os.path.dirname(os.path.abspath(__file__))

    def repo_git(*a):
        return subprocess.run([git, "-C", root, *a], capture_output=True, text=True,
                              check=True).stdout.strip()

    remote = repo_git("remote", "get-url", "origin")
    name, email = repo_git("config", "user.name"), repo_git("config", "user.email")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        manifest = export_snapshot(store, tmp)
        store.close()
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(DATA_README)

        def run(*a):
            subprocess.run([git, "-C", tmp, *a], check=True)

        run("init", "-q", "-b", DATA_BRANCH)
        run("config", "user.name", name)
        run("config", "user.email", email)
        run("add", "-A")
        run("commit", "-q", "-m", f"Data snapshot {manifest['published_at']}")
        print(f"Publishing {manifest['rows']:,} observations across {manifest['series']} "
              f"series (latest vintage {manifest['latest_vintage']}) to {remote} "
              f"branch '{DATA_BRANCH}'.")
        run("push", "-q", "--force", remote, f"HEAD:{DATA_BRANCH}")

    slug = remote.removesuffix(".git").split("github.com")[-1].strip("/:")
    print(f"Done. The hosted app reads it from:\n  "
          f"https://raw.githubusercontent.com/{slug}/{DATA_BRANCH}")


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
    p.add_argument("--sources", default="config/sources.yml")
    p.add_argument("--source", choices=["fred", "macrobond"], default=None,
                   help="Use this vendor for every series, ignoring sources.yml.")
    p.add_argument("--db", default=os.environ.get("MACRO_REGIME_DB",
                                                  "data/regime.duckdb"))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("backfill", cmd_backfill), ("sync", cmd_sync),
                     ("coverage", cmd_coverage), ("publish", cmd_publish),
                     ("demo", cmd_demo)]:
        sub.add_parser(name).set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
