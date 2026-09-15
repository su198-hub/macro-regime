"""Source adapter interface.

Everything downstream of this file is source-agnostic. Adding a vendor means
writing one class here with these three methods and mapping its codes in
config/sources.yml.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol

import pandas as pd


class Source(Protocol):
    name: str

    def fetch_with_vintages(
        self, series_id: str, start: dt.date | None = None
    ) -> pd.DataFrame:
        """Return columns: series_id, observation_date, vintage_date, value.

        One row per (observation, vintage). A series with no revision history
        returns a single vintage per observation, dated at first publication
        if known and at fetch date otherwise.
        """
        ...

    def fetch_current(self, series_id: str) -> pd.DataFrame:
        """Latest values only. Same columns, vintage_date = today."""
        ...

    def describe(self, series_id: str) -> dict:
        """title, units, frequency, has_vintages."""
        ...


def tidy_vintages(df: pd.DataFrame) -> pd.DataFrame:
    """Drop projections and unchanged repeats from a full vintage history.

    Vendors that return every vintage as a complete copy of the series repeat
    each unchanged value hundreds of times. Keeping only the vintage where a
    value first appears or changes gives exactly the same answer from
    Store.as_of, which takes the latest vintage on or before a date.

    Observations dated after their own vintage are projections (CBO, OMB), not
    data, and are dropped so a forecast never passes for something observed.
    """
    if df.empty:
        return df
    out = df.copy()
    out["observation_date"] = pd.to_datetime(out["observation_date"]).dt.date
    out["vintage_date"] = pd.to_datetime(out["vintage_date"]).dt.date
    out = out[out["observation_date"] <= out["vintage_date"]]
    out = out.dropna(subset=["value"])
    out = (out.sort_values(["series_id", "observation_date", "vintage_date"])
              .drop_duplicates(["series_id", "observation_date", "vintage_date"], keep="last"))
    prev = out.groupby(["series_id", "observation_date"])["value"].shift()
    return out[prev.isna() | (out["value"] != prev)].reset_index(drop=True)
