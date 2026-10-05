"""EIA files for the energy page: the Short-Term Energy Outlook and Form EIA-860M.

Neither is read through the Source interface the model's series use, because
what the energy page needs from them is not a series EIA publishes but a
summary of each release: what EIA expected, as of that month, for the next
twelve months (STEO), and what was planned to come online, or go off, over the
next three years (860M). Each release becomes one observation, dated at the
month it was published, so the history is point-in-time by construction --
nothing in it was knowable later than its date says.

THE STEO ARCHIVE. Every monthly outlook since the 1980s is kept at
eia.gov/outlooks/steo/archives, as a workbook from 2009: .xls to mid-2013,
.xlsx after. The current month is not in the archive until the next one is out,
so it comes from the live file. Each table sheet carries its series codes in the
first column, a row of years over a row of month names, and one column per
month across a six-year window.

THE 860M ARCHIVE. A monthly inventory of every generator operating, planned,
retired or canceled, from July 2015. The files are large (about 14 MB), so the
history is sampled quarterly; the latest file is always read.

Downloads are cached outside the repo and outside OneDrive, and a release that
has been parsed is never fetched again: archived releases do not change.

openpyxl and xlrd read the workbooks. They are needed only here, at ingest, on
the desk machine; the hosted app reads the stored results.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

STEO_ARCHIVE = "https://www.eia.gov/outlooks/steo/archives/{tag}_base.{ext}"
STEO_CURRENT = "https://www.eia.gov/outlooks/steo/xls/STEO_m.xlsx"
GEN_URLS = ("https://www.eia.gov/electricity/data/eia860m/archive/xls/{month}_generator{year}.{ext}",
            "https://www.eia.gov/electricity/data/eia860m/xls/{month}_generator{year}.{ext}")
HEADERS = {"User-Agent": "macro-regime research (energy page)"}
XLSX_FROM = pd.Timestamp("2013-09-01")  # the archive is .xls before this, .xlsx after
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]

# STEO codes this page reads. Case varies between years, so matching ignores it,
# and some codes were renamed: electricity sales were EXTCPUS (billion kWh a
# day) until the tables were redone, ELTCTWH (billion kWh a month) after. Only
# growth rates are taken from them, so the change of unit does not matter.
STEO_CODES = {
    "ELTCTWH": "US electricity sales to ultimate customers, billion kWh",
    "EXTCPUS": "US electricity retail sales, billion kWh per day (older releases)",
    "T3_STCHANGE_WORLD": "World liquids inventory net withdrawals, million b/d",
    "COPS_OPEC": "OPEC surplus crude oil production capacity, million b/d",
    "NGHHMCF": "Henry Hub spot natural gas, $/Mcf",
}

# 860M energy source codes, grouped the way the supply question needs them.
GAS = {"NG"}
CLEAN = {"SUN", "WND", "MWH", "NUC", "WAT", "GEO"}

# How much of a megawatt counts at the summer peak, by fuel. Approximate, in
# the spirit of PJM's effective load carrying capability (ELCC) class ratings:
# a solar megawatt is worth about a tenth of a nuclear one when it matters, so
# nameplate additions overstate firm supply in a solar-led build. Gas peakers
# (combustion turbines, engines) rate lower than combined cycle.
FIRM_CREDIT = {
    "NUC": 0.95, "GEO": 0.90,
    "BIT": 0.80, "SUB": 0.80, "LIG": 0.80, "RC": 0.80, "WC": 0.80, "ANT": 0.80,
    "NG": 0.75, "DFO": 0.60, "RFO": 0.60, "JF": 0.60, "KER": 0.60, "WO": 0.60,
    "WAT": 0.50, "MWH": 0.50, "WND": 0.30, "SUN": 0.10,
}
PEAKER_PRIME_MOVERS = {"GT", "IC"}
OTHER_CREDIT = 0.60  # biomass, waste, landfill gas and the rest


def firm_credit(df: pd.DataFrame) -> pd.Series:
    """Each unit's peak-hour credit, from its fuel and, for gas, its prime mover."""
    if "Energy Source Code" not in df:
        return pd.Series(OTHER_CREDIT, index=df.index)
    src = df["Energy Source Code"].astype(str).str.strip().str.upper()
    credit = src.map(FIRM_CREDIT).fillna(OTHER_CREDIT)
    if "Prime Mover Code" in df:
        pm = df["Prime Mover Code"].astype(str).str.strip().str.upper()
        credit = credit.where(~((src == "NG") & pm.isin(PEAKER_PRIME_MOVERS)), 0.60)
    return credit


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".cache")
    path = Path(base) / "macro-regime" / "cache" / "eia"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path, session=requests) -> bool:
    """Fetch url to dest once. False if EIA has no such file."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    r = session.get(url, headers=HEADERS, timeout=120)
    # EIA answers a missing file with a 200 and its HTML error page.
    if r.status_code != 200 or r.content[:15].lstrip().lower().startswith(b"<!doctype html"):
        return False
    dest.write_bytes(r.content)
    time.sleep(0.5)  # one file at a time, politely
    return True


# ---------- STEO ----------

def _sheet_rows(path: Path) -> dict[str, list[tuple]]:
    """Every sheet of a workbook as lists of row tuples, .xls or .xlsx."""
    if path.suffix == ".xls":
        import xlrd
        book = xlrd.open_workbook(str(path))
        return {s.name: [tuple(s.row_values(i)) for i in range(s.nrows)] for s in book.sheets()}
    import openpyxl
    book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        return {ws.title: list(ws.iter_rows(values_only=True)) for ws in book.worksheets}
    finally:
        book.close()


def _month_columns(rows: list[tuple]) -> dict[int, pd.Timestamp]:
    """Map column index to month from the sheet's year row over its month row."""
    for i in range(min(len(rows) - 1, 12)):
        months = [str(v).strip()[:3].lower() if v not in (None, "") else "" for v in rows[i + 1]]
        if sum(m in MONTHS for m in months) < 12:
            continue
        year, out = None, {}
        for j, (y, m) in enumerate(zip(rows[i], months)):
            if isinstance(y, (int, float)) and 1980 < float(y) < 2100:
                year = int(y)
            if year and m in MONTHS:
                out[j] = pd.Timestamp(year=year, month=MONTHS.index(m) + 1, day=1)
        if out:
            return out
    return {}


def parse_steo(sheets: dict[str, list[tuple]], codes=STEO_CODES) -> dict[str, pd.Series]:
    """Monthly series for each wanted code, from one release's sheets."""
    wanted = {c.upper() for c in codes}
    found: dict[str, pd.Series] = {}
    for rows in sheets.values():
        cols = None
        for row in rows:
            code = str(row[0]).strip().upper() if row and row[0] is not None else ""
            if code not in wanted or code in found:
                continue
            cols = cols if cols is not None else _month_columns(rows)
            vals = {}
            for j, month in cols.items():
                v = row[j] if j < len(row) else None
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    vals[month] = float(v)
            if vals:
                found[code] = pd.Series(vals).sort_index()
    return found


def release_date(sheets: dict[str, list[tuple]], fallback: pd.Timestamp) -> dt.date:
    """The forecast date printed under 'Forecast date:', else the 10th of the month."""
    for rows in sheets.values():
        for row in rows[:6]:
            for v in row[:2]:
                if isinstance(v, str):
                    m = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", v)
                    if m:
                        try:
                            return dt.datetime.strptime(m.group(1), "%B %d, %Y").date()
                        except ValueError:
                            pass
                if isinstance(v, dt.datetime):
                    return v.date()
    return (fallback + pd.Timedelta(days=9)).date()


def steo_summary(series: dict[str, pd.Series], month: pd.Timestamp) -> dict[str, float]:
    """What one release expected for the twelve months starting with its own.

    'Next twelve' is the release month and the eleven after it; 'last twelve'
    the twelve before, which in a release are mostly actuals. Every release
    since 2009 runs to at least December of the following year, so twelve
    months ahead is always there; anything further is not.
    """
    ahead = pd.date_range(month, periods=12, freq="MS")
    behind = pd.date_range(month - pd.DateOffset(months=12), periods=12, freq="MS")
    out = {}

    def mean(code, idx):
        s = series.get(code)
        if s is None or not set(idx) <= set(s.index):
            return np.nan
        return float(s.reindex(idx).mean())

    sales = "ELTCTWH" if "ELTCTWH" in series else "EXTCPUS"
    fwd, back = mean(sales, ahead), mean(sales, behind)
    out["elec_growth_12m"] = (fwd / back - 1) * 100 if back and not np.isnan(back) else np.nan
    # The same forecast as a level, in billion kWh over the twelve months, so it
    # can be set against the level the release a year earlier expected. Older
    # releases give a daily rate, so each month is scaled by its days.
    s = series.get(sales)
    if s is not None and set(ahead) <= set(s.index):
        level = s.reindex(ahead)
        if sales == "EXTCPUS":
            level = level * ahead.days_in_month
        out["elec_fwd_12m_twh"] = float(level.sum())
    else:
        out["elec_fwd_12m_twh"] = np.nan
    out["oil_draw_12m"] = mean("T3_STCHANGE_WORLD", ahead)
    out["opec_spare_12m"] = mean("COPS_OPEC", ahead)
    out["henry_hub_12m"] = mean("NGHHMCF", ahead)
    return out


def steo_history(start: str = "2009-01", log=print) -> pd.DataFrame:
    """One row per monthly release: its date and what it expected."""
    session, cache = requests, cache_dir()
    rows = []
    months = pd.date_range(start, pd.Timestamp.today().normalize(), freq="MS")

    def fetch(month):
        tag = f"{MONTHS[month.month - 1]}{month.year % 100:02d}"
        # EIA takes ~10s to say a file is missing, so ask for the right format first.
        for ext in (("xls", "xlsx") if month < XLSX_FROM else ("xlsx", "xls")):
            dest = cache / f"steo_{tag}.{ext}"
            if _download(STEO_ARCHIVE.format(tag=tag, ext=ext), dest, session):
                return dest
        return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = [p for p in pool.map(fetch, months) if p is not None]
    log(f"  STEO: {len(paths)} archived releases on disk")
    # The latest release is not archived until the next one is out. The live
    # file is cached under today's date, so a new release is picked up.
    live = cache / f"steo_live_{pd.Timestamp.today():%Y%m%d}.xlsx"
    if _download(STEO_CURRENT, live, session):
        paths.append(live)
    for path in paths:
        sheets = _sheet_rows(path)
        tag = re.search(r"steo_([a-z]{3})(\d{2})", path.name)
        month = (pd.Timestamp(year=2000 + int(tag.group(2)), month=MONTHS.index(tag.group(1)) + 1,
                              day=1) if tag else pd.Timestamp.today().to_period("M").to_timestamp())
        released = release_date(sheets, month)
        rel_month = pd.Timestamp(released).to_period("M").to_timestamp()
        if any(r["month"] == rel_month for r in rows):
            continue  # the live file was last month's release, already read
        summary = steo_summary(parse_steo(sheets), rel_month)
        rows.append({"month": rel_month, "released": released, **summary})
        if len(rows) % 24 == 0:
            log(f"  STEO: read {len(rows)} releases, latest {rel_month:%b %Y}")
    return pd.DataFrame(rows)


# ---------- 860M ----------

CAPACITY = "Net Summer Capacity (MW)"  # the one capacity column every inventory since 2015 has


def _clean(v) -> str:
    """Headers in the 2015 files carry line breaks and padding."""
    return " ".join(str(v).split())


def _find_header(raw: pd.DataFrame) -> int:
    for i in range(min(len(raw), 10)):
        if any(_clean(v) == CAPACITY for v in raw.iloc[i]):
            return i
    raise ValueError(f"860M: no header row with '{CAPACITY}'")


def _sheet(book: pd.ExcelFile, name: str) -> pd.DataFrame:
    raw = pd.read_excel(book, name, header=None, nrows=12)
    df = pd.read_excel(book, name, header=_find_header(raw))
    df.columns = [_clean(c) for c in df.columns]
    df = df[pd.to_numeric(df[CAPACITY], errors="coerce").notna()].copy()
    df["mw"] = df[CAPACITY].astype(float)
    return df


def _when(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    y = pd.to_numeric(df.get(year_col), errors="coerce")
    m = pd.to_numeric(df.get(month_col), errors="coerce").fillna(6).clip(1, 12)
    ok = y.notna()
    out = pd.Series(pd.NaT, index=df.index)
    out[ok] = pd.to_datetime(dict(year=y[ok].astype(int), month=m[ok].astype(int), day=1))
    return out


def gen_summary(planned: pd.DataFrame, operating: pd.DataFrame, canceled: pd.DataFrame,
                month: pd.Timestamp, years: int = 3) -> dict[str, float]:
    """Capacity due on and off over the next `years`, from one inventory.

    Net additions are planned summer capacity due within the window less
    operating capacity scheduled to retire within it, as a share of what is
    operating now, so the number reads as growth in the fleet. The firm version
    weighs every unit by its peak-hour credit (FIRM_CREDIT) before doing the
    same sum, so it asks how much dependable supply is coming. The canceled
    share is canceled-or-postponed capacity against that plus what is still
    planned: a rising share means projects are falling away faster.
    """
    end = month + pd.DateOffset(years=years)
    due = _when(planned, "Planned Operation Year", "Planned Operation Month")
    window = planned[(due >= month) & (due < end)]
    retire = _when(operating, "Planned Retirement Year", "Planned Retirement Month")
    retiring = operating[(retire >= month) & (retire < end)]
    fleet = operating["mw"].sum()
    src = window["Energy Source Code"].astype(str).str.strip().str.upper()
    add = window["mw"].sum()
    canceled_mw = canceled["mw"].sum()
    planned_all = planned["mw"].sum()
    firm_fleet = (operating["mw"] * firm_credit(operating)).sum()
    firm_add = (window["mw"] * firm_credit(window)).sum()
    firm_retire = (retiring["mw"] * firm_credit(retiring)).sum()
    return {
        "net_add_36m_pct": (add - retiring["mw"].sum()) / fleet * 100,
        "firm_net_add_36m_pct": (firm_add - firm_retire) / firm_fleet * 100,
        "firm_add_36m_gw": firm_add / 1000,
        "add_36m_gw": add / 1000,
        "gas_add_36m_gw": window.loc[src.isin(GAS), "mw"].sum() / 1000,
        "clean_add_36m_gw": window.loc[src.isin(CLEAN), "mw"].sum() / 1000,
        "retire_36m_gw": retiring["mw"].sum() / 1000,
        "cancel_share_pct": canceled_mw / (canceled_mw + planned_all) * 100,
        "fleet_gw": fleet / 1000,
    }


def gen_inventory_month(book: pd.ExcelFile) -> pd.Timestamp | None:
    title = str(pd.read_excel(book, "Planned", header=None, nrows=1).iloc[0, 0])
    m = re.search(r"as of ([A-Za-z]+) (\d{4})", title)
    if not m or m.group(1).lower() not in MONTH_NAMES:
        return None
    return pd.Timestamp(year=int(m.group(2)), month=MONTH_NAMES.index(m.group(1).lower()) + 1, day=1)


def gen_history(start: str = "2015-07", every: int = 3, log=print) -> pd.DataFrame:
    """One row per sampled 860M inventory: quarterly history plus the latest."""
    cache = cache_dir()
    today = pd.Timestamp.today().normalize()
    months = list(pd.date_range(start, today, freq="MS"))
    sample = [m for m in months if (m.month - 1) % every == 0]
    sample += [m for m in months[-4:] if m not in sample]  # always try the latest

    def fetch(month):
        name = MONTH_NAMES[month.month - 1]
        for ext in ("xlsx", "xls"):
            dest = cache / f"860m_{month:%Y%m}.{ext}"
            if any(_download(u.format(month=name, year=month.year, ext=ext), dest)
                   for u in GEN_URLS):
                return dest
        return None

    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = [p for p in pool.map(fetch, sample) if p is not None]
    rows = []
    for path in paths:
        book = pd.ExcelFile(path)
        as_of = gen_inventory_month(book) or pd.Timestamp(path.stem.split("_")[1] + "01")
        if as_of > today or any(r["month"] == as_of for r in rows):
            continue
        summary = gen_summary(_sheet(book, "Planned"), _sheet(book, "Operating"),
                              _sheet(book, "Canceled or Postponed"), as_of)
        rows.append({"month": as_of, **summary})
        log(f"  860M: {as_of:%b %Y} read")
    return pd.DataFrame(rows)


# ---------- electricity sales, as they happened ----------

RETAIL_API = "https://api.eia.gov/v2/electricity/retail-sales/data/"


def api_key() -> str:
    """EIA_API_KEY from the process, else from the Windows user environment."""
    from src.sources.ciq import credential
    key = credential("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY is not set. Get a free key at eia.gov/opendata "
                           "and store it as a Windows user environment variable.")
    return key


def retail_sales(sectors=("COM", "IND")) -> pd.Series:
    """US electricity sold to the given sectors, million kWh a month, from 2001.

    Commercial is where most data centers are metered; industrial carries the
    factories. Residential is left out: it moves with the weather and with
    households, not with the build-out this page is about.
    """
    params = {"api_key": api_key(), "frequency": "monthly", "data[0]": "sales",
              "facets[stateid][]": "US", "facets[sectorid][]": list(sectors),
              "sort[0][column]": "period", "sort[0][direction]": "asc", "length": 5000}
    r = requests.get(RETAIL_API, params=params, timeout=60)
    r.raise_for_status()
    rows = pd.DataFrame(r.json()["response"]["data"])
    rows["sales"] = pd.to_numeric(rows["sales"], errors="coerce")
    total = rows.groupby("period")["sales"].sum(min_count=len(sectors))
    total.index = pd.to_datetime(total.index)
    return total.dropna().sort_index()
