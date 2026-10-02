"""Two small public feeds for the energy page: FRED price series and Epoch AI.

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
import time

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
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
    return pd.DataFrame({"series_id": series_id,
                         "observation_date": obs.date,
                         "vintage_date": pd.DatetimeIndex(vint).date,
                         "value": s.to_numpy(dtype=float)})
