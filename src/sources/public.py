"""Small public feeds for the energy page: FRED, Census, Epoch AI and the GPR index.

FRED. Read from fredgraph.csv, which needs no key. That returns the series as
revised now, not its vintages, so these are stored as single-vintage and are
as-revised in any backtest. For producer prices that matters little: they are
revised once, four months after first release.

EPOCH AI. The Frontier Data Centers hub (epoch.ai/data/data-centers) tracks the
large US AI sites from satellite imagery, permits and filings, with a timeline
per site that runs into the planned future. Published under CC BY; the download
is the whole database as it stands today, with no history of earlier
estimates. Each ingest therefore stores the full path under today's date, so
the revision history -- how the planned pipeline has grown or slipped --
accumulates from now on.
"""

from __future__ import annotations

import io
import re
import time

import numpy as np

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
CENSUS_PRIVATE = "https://www.census.gov/construction/c30/xlsx/privsatime.xlsx"
GPR_XLS = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
EPOCH_SITES = "https://epoch.ai/data/data_centers/data_centers.csv"
EPOCH_TIMELINES = "https://epoch.ai/data/data_centers/data_center_timelines.csv"
HEADERS = {"User-Agent": "macro-regime research (energy page)"}


def fred_series(sid: str, tries: int = 4) -> pd.Series:
    # No User-Agent of our own: fredgraph leaves any custom one hanging until it
    # times out, and answers the library default at once.
    for attempt in range(tries):
        try:
            r = requests.get(FRED_CSV.format(sid=sid), timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index(pd.to_datetime(df.dropna()["date"]))["value"]


def us_power_path(sites: pd.DataFrame, timelines: pd.DataFrame,
                  start: str = "2019-01", end: str = "2030-12") -> pd.Series:
    """Total power of the US sites Epoch tracks, month by month, in GW.

    Each site's power steps to each value in its timeline on that entry's date
    and holds until the next, so a month's total is what every site had reached,
    or was planned to reach, by then.
    """
    us = set(sites.loc[sites["Country"].astype(str).str.strip() == "United States", "Name"])
    t = timelines[timelines["Data center"].isin(us)].copy()
    t["Date"] = pd.to_datetime(t["Date"], errors="coerce")
    t["Power (MW)"] = pd.to_numeric(t["Power (MW)"], errors="coerce")
    t = t.dropna(subset=["Date", "Power (MW)"]).sort_values("Date")
    months = pd.date_range(start, end, freq="MS")
    total = pd.Series(0.0, index=months)
    for _, site in t.groupby("Data center"):
        path = site.set_index("Date")["Power (MW)"]
        path = path[~path.index.duplicated(keep="last")]
        total += path.reindex(path.index.union(months)).ffill().reindex(months).fillna(0)
    return total / 1000


def epoch_power_path() -> pd.Series:
    def get(url):
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    return us_power_path(get(EPOCH_SITES), get(EPOCH_TIMELINES))


def as_observations(series_id: str, s: pd.Series, vintage=None,
                    lag_days: int = 0) -> pd.DataFrame:
    """Store rows. Without `vintage`, each value is dated `lag_days` after its observation."""
    s = s.dropna()
    obs = pd.to_datetime(s.index)
    vint = ([pd.Timestamp(vintage)] * len(s) if vintage is not None
            else obs + pd.Timedelta(days=lag_days))
    # A value in hand today was published by today, whatever the usual lag
    # says; a later date would hide it from a view of what is known now.
    today = pd.Timestamp.today().normalize()
    vint = pd.DatetimeIndex(vint)
    vint = vint.where(vint <= today, today)
    return pd.DataFrame({"series_id": series_id,
                         "observation_date": obs.date,
                         "vintage_date": pd.DatetimeIndex(vint).date,
                         "value": s.to_numpy(dtype=float)})


def census_data_centers() -> pd.Series:
    """Private data center construction, $ millions at a seasonally adjusted annual rate.

    Census broke data centers out of private office construction in 2024, with
    monthly estimates back to January 2014. It is not on FRED, so it is read
    from the Census workbook, whose dates read "Aug-26p" (p preliminary, r revised).
    """
    r = requests.get(CENSUS_PRIVATE, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content), header=None)
    head = next(i for i in range(10) if any(str(v).strip() == "Data center" for v in raw.iloc[i]))
    col = next(j for j, v in enumerate(raw.iloc[head]) if str(v).strip() == "Data center")
    body = raw.iloc[head + 1:, [0, col]].dropna()
    body = body[body[0].astype(str).str.match(r"^[A-Z][a-z]{2}-\d{2}")]
    idx = pd.to_datetime(body[0].astype(str).str[:6], format="%b-%y")
    return pd.Series(body[col].astype(float).to_numpy(), index=idx).sort_index()


def geopolitical_risk() -> pd.Series:
    """Caldara and Iacoviello's monthly geopolitical risk index, 1985 on."""
    r = requests.get(GPR_XLS, timeout=60)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content))
    return df.set_index(pd.to_datetime(df["month"]))["GPR"].dropna().astype(float)


# ---------- PJM capacity prices ----------

SOM_SEC5 = ("https://www.monitoringanalytics.com/reports/PJM_State_of_the_Market/"
            "{year}/{year}-som-pjm-sec5.pdf")

# When each delivery year's price became known: the month of its base residual
# auction. The standard rule, May three years ahead, held for 2011/12 through
# 2021/22. The transition years before it and the compressed schedule after are
# listed; the four transition dates are approximate to the season.
BRA_MONTH = {
    2007: "2007-04", 2008: "2008-01", 2009: "2008-07", 2010: "2009-01",   # approximate
    2022: "2021-05", 2023: "2022-06", 2024: "2022-12", 2025: "2024-07",
    2026: "2025-07", 2027: "2025-12",
}


def parse_rpm_prices(text: str) -> pd.Series:
    """Weighted average RPM price by delivery year, from the IMM's revenue table.

    The table ("RPM revenue by delivery year") weighs every auction for a
    delivery year by the capacity it cleared, so it is the price actually paid.
    Keyed by the delivery year's first calendar year (2027 for 2027/2028).
    """
    pattern = r"(20\d\d)/20\d\d\s+\$([\d,.]+)\s+([\d,.]+)\s+(\d+)\s+\$([\d,]+)"
    # The title appears in the text before the table itself; read the first
    # occurrence that is followed by rows.
    for m in re.finditer("RPM revenue by delivery year", text):
        rows = re.findall(pattern, text[m.start():m.start() + 4000])
        if rows:
            return pd.Series({int(y): float(p.replace(",", "")) for y, p, *_ in rows}).sort_index()
    raise ValueError("could not read an 'RPM revenue by delivery year' table in the report")


def bra_month(delivery_year: int) -> pd.Timestamp:
    return pd.Timestamp(BRA_MONTH.get(delivery_year, f"{delivery_year - 3}-05"))


def forward_capacity_price(prices: pd.Series, end=None) -> pd.Series:
    """Month by month, the price for the latest delivery year already auctioned.

    That is what the market knew about the cost of capacity one to three years
    out at each date. Uses the final weighted average for each year, so the
    months between a base auction and its later incremental auctions carry a
    little hindsight.
    """
    first = min(bra_month(y) for y in prices.index)
    months = pd.date_range(first, end or pd.Timestamp.today(), freq="MS")
    out = []
    for m in months:
        # the furthest delivery year known by m
        years = [y for y in prices.index if bra_month(y) <= m]
        out.append(prices[max(years)] if years else np.nan)
    return pd.Series(out, index=months).dropna()


def pjm_capacity_prices() -> pd.Series:
    """Read the latest State of the Market capacity section that exists."""
    import pymupdf
    for year in range(pd.Timestamp.today().year, pd.Timestamp.today().year - 4, -1):
        r = requests.get(SOM_SEC5.format(year=year), headers={"User-Agent": "Mozilla/5.0"},
                         timeout=120)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            doc = pymupdf.open(stream=r.content, filetype="pdf")
            return parse_rpm_prices("\n".join(p.get_text() for p in doc))
    raise RuntimeError("no State of the Market capacity section found for the last four years")
