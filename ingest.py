#!/usr/bin/env python3
"""Ingestion CLI.

    python ingest.py demo                 # synthetic data, no key needed
    python ingest.py backfill             # full vintage history
    python ingest.py sync                 # refresh, for scheduled runs
    python ingest.py coverage             # what you actually have
    python ingest.py publish              # push a snapshot for the hosted app
    python ingest.py energy               # the /energy page's public series

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
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

from src.drivers import load_config, required_series
from src.store import Store


# Vendors whose terms forbid redistribution. `publish` pushes to a public
# branch, so it refuses to run while the store holds any of their series.
RESTRICTED_VENDORS = {"bloomberg", "ciq"}


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


def cmd_ciq_login(args) -> None:
    """Store Capital IQ credentials from a masked prompt.

    Deliberately interactive only. The password is typed into getpass, so it is
    not echoed, not a command line, and not in shell history; it goes to the
    Windows user environment, which neither the repo nor OneDrive can see.
    """
    import getpass

    from src.sources.ciq import set_user_env, user_env

    if sys.platform != "win32":
        print("  Not on Windows: export CIQ_USERNAME and CIQ_PASSWORD in your shell "
              "profile instead, or use your OS keychain.")
        return
    if args.clear:
        for name in ("CIQ_USERNAME", "CIQ_PASSWORD"):
            set_user_env(name, None)
        print("  Capital IQ credentials removed from your Windows user environment.")
        return
    not_here = ("  Run this in your own terminal. It prompts for the password, and a "
                "prompt cannot be answered from a script — which is the point.")
    # isatty() alone is not enough on Windows: the NUL device reports itself as a
    # terminal, so a script with no input gets past it and hits EOF instead.
    if not sys.stdin.isatty():
        print(not_here)
        return

    current = user_env("CIQ_USERNAME")
    hint = f" [{current}]" if current else ""
    try:
        username = input(f"  Capital IQ username{hint}: ").strip() or current
        password = getpass.getpass("  Capital IQ password (not shown): ")
    except (EOFError, KeyboardInterrupt):
        print("\n" + not_here)
        return
    if not (username and password):
        print("  Nothing stored: both are needed.")
        return
    set_user_env("CIQ_USERNAME", username)
    set_user_env("CIQ_PASSWORD", password)
    print(f"\n  Stored for {username}. Nothing was written to the repo.")
    print("  Next:  python ingest.py ciq-probe --id IQ24937 --code IQ_CLOSEPRICE")


def cmd_ciq_probe(args) -> None:
    """One call, printed raw, so the entitlement question is answered by the
    vendor rather than guessed at — and so the response shape is visible before
    anything is written to the store."""
    import json

    from src.sources.ciq import CiqAuthError, CiqSource

    code = f"{args.id}|{args.code}"
    try:
        source = CiqSource()
    except RuntimeError as exc:
        print(f"  {exc}")
        return
    print(f"  endpoint  {source.endpoint}")
    print(f"  user      {source.username}")
    print(f"  asking    {code}\n")
    try:
        payload = source.request(source._body(args.id, args.code, None, None))
    except CiqAuthError as exc:
        print(f"  NOT ENTITLED (or wrong credentials)\n\n  {exc}")
        return
    except Exception as exc:
        print(f"  call failed: {type(exc).__name__}: {exc}")
        return
    finally:
        source.close()

    text = json.dumps(payload, indent=2)
    print(text[:4000] + ("\n  ...truncated" if len(text) > 4000 else ""))
    rows = []
    try:
        rows = source._rows(payload)
    except Exception as exc:
        print(f"\n  the parser did not understand this shape: {exc}")
    print(f"\n  parsed {len(rows)} rows"
          + (f", first {rows[0]}, last {rows[-1]}" if rows else
             " — adjust CiqSource._rows to match the shape above"))


def cmd_twin_build(args) -> None:
    """Build the twin's own store: the main store plus the twin-only series.

    Separate so the Bloomberg series never enter the store `publish` pushes.
    Rerun it to refresh; it recopies the main store each time, so the twin never
    scores older data than the presentation does.
    """
    from src import twin as tw

    live = load_config(args.config)
    cfg = tw.build_config(live, tw.load_overlay())
    target = args.twin_db or tw.default_db()
    tw.build_store(args.db, target, live, cfg, load_sources(args.sources), route, open_vendor)
    print(f"\nTwin store ready at {target}. Open localhost:8501/twin to compare.")


def open_vendor(vendor: str):
    if vendor == "macrobond":
        from src.sources.macrobond import MacrobondSource
        return MacrobondSource()
    if vendor == "ciq":
        from src.sources.ciq import CiqSource
        return CiqSource()
    if vendor == "bloomberg":
        from src.sources.bloomberg import BloombergSource
        return BloombergSource()
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
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - set(series)
        if unknown:
            raise SystemExit(f"not in this config: {', '.join(sorted(unknown))}")
        series = [s for s in series if s in wanted]
    vendors: dict[str, object] = {}

    verb = "Backfilling" if full_history else "Syncing"
    print(f"{verb} {len(series)} series into {args.db}.")
    failed = 0
    for i, sid in enumerate(series, 1):
        try:
            entry = (sources.get("series") or {}).get(sid) or {}
            if entry.get("use") == "energy" and not args.source:
                # Built by `ingest.py energy` from public sources, not fetched here.
                print(f"  [{i}/{len(series)}] {sid}: from `ingest.py energy`, skipped")
                continue
            if entry.get("use") == "derived" and not args.source:
                # Built from a vendor's vintage history rather than fetched: a
                # target-dated projection has to be turned into a dated revision
                # first. Same builders the twin store uses, so the two agree.
                from src import twin as tw
                if "macrobond" not in vendors:
                    vendors["macrobond"] = open_vendor("macrobond")
                df = tw.DERIVED[sid](vendors["macrobond"].api).assign(series_id=sid)
                n = store.upsert_observations(df)
                code = entry.get("derived_from", sid)
                store.record_meta(sid, "macrobond", source_code=code,
                                  title=str(entry.get("note", sid)).strip()[:120],
                                  units="", frequency="irregular", has_vintages=True)
                print(f"  [{i:>2}/{len(series)}] {sid:<12} derived:{code:<20} {n:>7} rows")
                continue
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


def cmd_energy(args):
    """Pull everything the /energy page reads: EIA, FRED prices, Epoch AI.

    Every source here is public, so the result can be published. Sources are
    tagged energy-*, not fred or eia, so the dashboard's own source line --
    which names the vendors behind the presented model -- is unchanged.
    """
    from src.sources import eia
    from src.sources.public import (as_observations, bra_month, census_data_centers,
                                    epoch_power_path, forward_capacity_price, fred_series,
                                    geopolitical_risk, pjm_capacity_prices)

    store = Store(args.db)
    refuse_demo_store(store, args.db)
    stored = []

    # Each series is written as soon as it is fetched, so a timeout late in the
    # run (FRED is the usual one) does not throw away the slow EIA work before it.
    def add(sid, frame, source, title, units, freq, code=None, vintages=True):
        if frame.empty:
            print(f"  {sid}: nothing to store")
            return
        stored.append(store.upsert_observations(frame))
        store.record_meta(sid, source, title=title, units=units, frequency=freq,
                          has_vintages=vintages, source_code=code or sid)

    print("EIA Short-Term Energy Outlook, every release since 2009 ...")
    com_paths: list = []
    steo = eia.steo_history(log=print, paths_out=com_paths)
    for col, sid, title, units in [
        ("elec_fwd_12m_twh", "EN_STEO_ELEC_FWD",
         "EIA expected US electricity sales over the next 12 months", "billion kWh"),
        ("com_fwd_12m_twh", "EN_STEO_COM_FWD",
         "EIA expected US commercial electricity sales over the next 12 months", "billion kWh"),
        ("elec_growth_12m", "EN_STEO_ELEC_GROWTH",
         "EIA expected growth in US electricity sales, next 12 vs last 12 months", "%"),
        ("oil_draw_12m", "EN_STEO_OIL_DRAW",
         "EIA expected world liquids inventory draws, next 12 months", "million b/d"),
        ("opec_spare_12m", "EN_STEO_OPEC_SPARE",
         "EIA expected OPEC surplus production capacity, next 12 months", "million b/d"),
        ("henry_hub_12m", "EN_STEO_HENRY_HUB",
         "EIA expected Henry Hub spot price, next 12 months", "$/Mcf"),
        ("oecd_stocks", "EN_STEO_OECD_STOCKS",
         "OECD commercial oil inventories, latest month reported", "million barrels"),
    ]:
        s = steo.set_index("month")[col].dropna()
        released = pd.Series(pd.to_datetime(steo.set_index("month")["released"]))
        frame = pd.DataFrame({"series_id": sid, "observation_date": s.index.date,
                              "vintage_date": released.reindex(s.index).dt.date.to_numpy(),
                              "value": s.to_numpy()})
        add(sid, frame, "energy-eia", title, units, "Monthly", code="STEO")

    # Every release's whole path of commercial sales, each dated at its
    # release: the vintages ASR's Chart 2 draws, one line per outlook.
    path_rows = [pd.DataFrame({"series_id": "EN_STEO_COM_PATH",
                               "observation_date": s.index.date,
                               "vintage_date": released, "value": s.to_numpy(dtype=float)})
                 for released, s in com_paths]
    if path_rows:
        add("EN_STEO_COM_PATH", pd.concat(path_rows, ignore_index=True), "energy-eia",
            "EIA commercial electricity sales path, history and forecast, by outlook",
            "billion kWh a month", "Monthly", code="STEO")

    print("EIA Form 860M, quarterly since July 2015 plus the latest ...")
    gen = eia.gen_history(log=print).set_index("month")
    for col, sid, title, units in [
        ("net_add_36m_pct", "EN_860M_NET_ADD",
         "Planned net capacity additions over the next 3 years, % of operating fleet", "%"),
        ("firm_net_add_36m_pct", "EN_860M_FIRM_NET_ADD",
         "Firm capacity due in the next 3 years net of retirements, % of firm fleet", "%"),
        ("firm_add_36m_gw", "EN_860M_FIRM_ADD", "Firm capacity due online in the next 3 years", "GW"),
        ("gas_add_36m_gw", "EN_860M_GAS_ADD", "Gas capacity due online in the next 3 years", "GW"),
        ("clean_add_36m_gw", "EN_860M_CLEAN_ADD",
         "Solar, wind, storage, nuclear, hydro and geothermal due in the next 3 years", "GW"),
        ("retire_36m_gw", "EN_860M_RETIRE", "Capacity scheduled to retire in the next 3 years", "GW"),
        ("cancel_share_pct", "EN_860M_CANCEL_SHARE",
         "Canceled or postponed capacity, % of canceled plus planned", "%"),
    ]:
        # An inventory is published about two months after the month it describes.
        add(sid, as_observations(sid, gen[col], lag_days=55), "energy-eia", title, units,
            "Quarterly", code="860M")

    print("FRED producer prices, copper, power construction and GDP ...")
    for sid, title, units in [
        ("WPU0542", "PPI: commercial electric power", "index"),
        ("WPU0543", "PPI: industrial electric power", "index"),
        ("PRPWRCONS", "Private construction spending: power", "$ millions, SAAR"),
        ("GDP", "Gross domestic product", "$ billions, SAAR"),
        ("PRMFGCONS", "Private construction spending: manufacturing", "$ millions, SAAR"),
        ("B935RC1Q027SBEA", "Private fixed investment: computers and peripheral equipment",
         "$ billions, SAAR"),
        ("PCU335311335311", "PPI: power, distribution and specialty transformers", "index"),
        ("PCU335313335313", "PPI: switchgear and switchboard apparatus", "index"),
        ("PPIACO", "PPI: all commodities", "index 1982=100"),
        ("PCOPPUSDM", "Global price of copper (IMF)", "$/tonne"),
    ]:
        add(sid, as_observations(sid, fred_series(sid), lag_days=45), "energy-fred", title,
            units, "Monthly", vintages=False)

    print("EIA commercial and industrial electricity sales ...")
    add("EN_EIA_CI_SALES", as_observations("EN_EIA_CI_SALES", eia.retail_sales(), lag_days=55),
        "energy-eia", "US electricity sales to commercial and industrial customers",
        "million kWh", "Monthly", code="electricity/retail-sales", vintages=False)

    print("PJM capacity prices, from Monitoring Analytics ...")
    # Each month is known from the base auction that set it, so it is dated there.
    pjm = forward_capacity_price(pjm_capacity_prices())
    add("EN_PJM_CAPACITY", as_observations("EN_PJM_CAPACITY", pjm, lag_days=0),
        "energy-pjm", "PJM capacity price, latest delivery year auctioned (weighted average RPM)",
        "$/MW-day", "Monthly", code="State of the Market, RPM revenue table", vintages=False)

    add("EN_EIA_COM_SALES", as_observations("EN_EIA_COM_SALES", eia.retail_sales(("COM",)),
                                            lag_days=55),
        "energy-eia", "US electricity sales to commercial customers", "million kWh",
        "Monthly", code="electricity/retail-sales", vintages=False)

    print("Census data center construction ...")
    add("EN_CENSUS_DC", as_observations("EN_CENSUS_DC", census_data_centers(), lag_days=32),
        "energy-census", "Private construction spending: data centers", "$ millions, SAAR",
        "Monthly", code="privsatime.xlsx", vintages=False)

    print("Geopolitical risk index ...")
    add("EN_GPR", as_observations("EN_GPR", geopolitical_risk(), lag_days=5),
        "energy-gpr", "Geopolitical risk index (Caldara and Iacoviello)", "index, 1985-2019 = 100",
        "Monthly", code="GPR", vintages=False)

    print("Epoch AI Frontier Data Centers ...")
    add("EN_EPOCH_US_POWER",
        as_observations("EN_EPOCH_US_POWER", epoch_power_path(), vintage=dt.date.today()),
        "energy-epoch", "Power of the US AI data centers Epoch tracks, built and planned",
        "GW", "Monthly", code="data_center_timelines.csv")

    # The composite itself, so the regime model can read energy tightness as one
    # series. Recomputed from today's data, so it is as-revised, not
    # point-in-time; each month is dated a release cycle after it.
    from src import energy as en
    ecfg = en.load_config()
    wide = store.as_of(en.required_series(ecfg), dt.date.today()).astype(float)
    comp = en.compute(ecfg, wide)["composite"]
    # Replaced whole, so a month the rules now leave blank does not linger.
    store.con.execute("DELETE FROM observations WHERE series_id IN "
                      "('EN_ENERGY_TIGHTNESS', 'EN_POWER_TIGHTNESS', 'EN_FUELS_TIGHTNESS')")
    for col, sid, title in [("energy", "EN_ENERGY_TIGHTNESS", "Energy tightness (energy page composite)"),
                            ("power", "EN_POWER_TIGHTNESS", "Power tightness (energy page)"),
                            ("fuels", "EN_FUELS_TIGHTNESS", "Oil and gas tightness (energy page)")]:
        add(sid, as_observations(sid, comp[col], lag_days=45), "energy-composite", title,
            "score, -1 to +1", "Monthly", code="config/energy.yml", vintages=False)

    print(f"Stored {sum(stored)} observations across {len(stored)} series in {args.db}.")
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


def cmd_truth(args):
    """Label every month since 1980 with the regime it turned out to be in.

    Reads Macrobond directly (the inputs are not dashboard indicators), stores
    the inputs as TR_ series so a publish carries them to the hosted app's
    /history page, and writes the labels to a CSV for a quick look.
    """
    import datetime as dt

    from macrobond_data_api.com import ComClient

    from src import truth

    with ComClient() as api:
        m = truth.fetch(api)
    store = Store(args.db)
    refuse_demo_store(store, args.db)
    # Today's revised history is all the labels use, so replace rather than
    # stack a new vintage on every run.
    store.con.execute("DELETE FROM observations WHERE starts_with(series_id, 'TR_')")
    store.con.execute("DELETE FROM series_meta WHERE starts_with(series_id, 'TR_')")
    store.upsert_observations(truth.to_observations(m, dt.date.today()))
    for name, sid in truth.STORE_IDS.items():
        title, units, freq = truth.TITLES[name]
        store.record_meta(sid, "macrobond", title=title, units=units, frequency=freq,
                          has_vintages=False, source_code=truth.CODES[name])
    store.close()
    lab = truth.build(m)
    out = args.out or os.path.join(os.environ.get("LOCALAPPDATA", "data"), "macro-regime", "truth_labels.csv")
    lab.to_csv(out)
    for k, v in truth.summary(lab).items():
        print(f"{k}: {v}")
    print(truth.episode_check(lab).to_string(index=False))
    print(f"\nStored {len(truth.STORE_IDS)} input series in {args.db}; wrote {len(lab)} months to {out}")


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
    # A publish force-pushes over the branch the hosted app reads, so it checks
    # first that this store could actually serve the model. A test fixture with
    # one row in it once reached this far and replaced the real snapshot; the
    # guard below is what would have stopped it.
    have = set(store.coverage()["series_id"]) if not store.coverage().empty else set()
    wanted = set(required_series(load_config(args.config)))
    if len(have & wanted) < max(3, len(wanted) // 2):
        store.close()
        sys.exit(
            f"{args.db} holds {len(have & wanted)} of the {len(wanted)} series the model "
            f"needs, so publishing it would replace the snapshot with something that "
            f"cannot score the model. Refusing. Run backfill against the real store "
            f"first, or pass --db explicitly if you meant a different one.")

    restricted = sorted(set(store.sources()) & RESTRICTED_VENDORS)
    if restricted and not getattr(args, "allow_licensed", False):
        store.close()
        sys.exit(
            f"{args.db} holds series from {', '.join(restricted)}, whose licences do "
            f"not allow redistribution by default, and publish pushes to a public "
            f"branch. Refusing rather than dropping them quietly, which would make the "
            f"hosted app score a different model from the local one.\n\n"
            f"If the vendor has agreed to this, pass --allow-licensed. That is a "
            f"deliberate act and it is recorded: the flag is in your shell history, and "
            f"the snapshot's manifest names the vendors it carries.")
    if restricted:
        print(f"--allow-licensed: publishing {', '.join(restricted)} data to a PUBLIC "
              f"branch on the stated authority of the vendor. Recorded in the manifest.")

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
        if restricted:
            # The audit trail. Anyone who finds this snapshot can see which
            # licensed vendors it carries and that it was published knowingly.
            manifest["licensed_vendors"] = restricted
            manifest["licensed_note"] = (
                "Published with --allow-licensed on the stated authority of the "
                "vendor. Contains data from: " + ", ".join(restricted))
            with open(os.path.join(tmp, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2, default=str)
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
    # --country picks both config files from config/countries.yml; --config and
    # --sources still win, for a one-off file that is not in the registry yet.
    p.add_argument("--country", default=None,
                   help="Country code from config/countries.yml (default: its default).")
    p.add_argument("--config", default=None)
    p.add_argument("--sources", default=None)
    p.add_argument("--source", choices=["fred", "macrobond", "ciq", "bloomberg"], default=None,
                   help="Use this vendor for every series, ignoring sources.yml.")
    # Comma separated, not nargs="+", which would swallow the subcommand.
    p.add_argument("--only", default=None, metavar="A,B,C",
                   help="Fetch only these series names, for adding an indicator "
                        "without refetching everything.")
    p.add_argument("--db", default=os.environ.get("MACRO_REGIME_DB",
                                                  "data/regime.duckdb"))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("backfill", cmd_backfill), ("sync", cmd_sync),
                     ("coverage", cmd_coverage), ("demo", cmd_demo)]:
        sub.add_parser(name).set_defaults(func=fn)
    sub.add_parser(
        "energy",
        help="Pull the /energy page's public series: EIA outlooks and 860M, FRED prices, "
             "Epoch AI. Slow the first time (it reads every archived release); cached after.",
    ).set_defaults(func=cmd_energy)
    tr = sub.add_parser(
        "truth",
        help="Label each month since 1980 with its realised growth x inflation regime "
             "(GDP vs potential, core PCE vs expectations), from Macrobond.")
    tr.add_argument("--out", default=None,
                    help="CSV path (default %%LOCALAPPDATA%%\\macro-regime\\truth_labels.csv).")
    tr.set_defaults(func=cmd_truth)
    pub = sub.add_parser("publish", help="Export the store to the public data branch.")
    pub.add_argument(
        "--allow-licensed", action="store_true",
        help="Publish even though the store holds vendor data whose default terms "
             "forbid redistribution. Only with the vendor's agreement: this pushes to "
             "a PUBLIC branch read by a public app. The choice is recorded in the "
             "snapshot manifest.")
    pub.set_defaults(func=cmd_publish)
    twin = sub.add_parser(
        "twin-build",
        help="Build the twin's store (main store + proposed series) for the /twin page.")
    twin.add_argument("--twin-db", default=None,
                      help="Where to write it (default %%LOCALAPPDATA%%\\macro-regime\\twin.duckdb).")
    twin.set_defaults(func=cmd_twin_build)
    login = sub.add_parser(
        "ciq-login",
        help="Store Capital IQ credentials from a masked prompt, as Windows user "
             "environment variables. Run it in your own terminal.")
    login.add_argument("--clear", action="store_true",
                       help="Remove the stored credentials instead.")
    login.set_defaults(func=cmd_ciq_login)
    probe = sub.add_parser(
        "ciq-probe",
        help="One Capital IQ call, printed raw. Run this before any backfill: it "
             "says in seconds whether the API is entitled on your seat, and shows "
             "the response shape the parser has to match.")
    probe.add_argument("--id", required=True,
                       help="Entity identifier, e.g. IQ12345 or a ticker.")
    probe.add_argument("--code", required=True, help="Mnemonic, e.g. IQ_CLOSEPRICE.")
    probe.set_defaults(func=cmd_ciq_probe)
    args = p.parse_args()
    from src.countries import load_registry, resolve
    here = resolve(load_registry(), args.country, strict=args.country is not None)
    args.config = args.config or here["indicators"]
    args.sources = args.sources or here["sources"]
    args.func(args)


if __name__ == "__main__":
    main()
