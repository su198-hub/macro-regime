"""The Federal Reserve Board's FRB/US model data: quarterly history of the
model's variables, published with each model release.

Used for PTR, expected PCE inflation over the next ten years. From 1979 it is
read from surveys (Hoey, then the Survey of Professional Forecasters); before
that it is the Kozicki-Tinsley estimate, which starts in 1968. It is the
longest-running measure of long-run inflation expectations, and it lets the
answer key and the inflation signposts reach back into the 1970s.

A code in config/sources.yml is a column of HISTDATA.TXT, e.g. "PTR". Public
data, free to use. Each release restates the whole history, with no record of
earlier releases. Stored dated today, a rewind to any earlier date would see
none of it, so each quarter is instead treated as published on the first day of
the month after the quarter ends, when the survey behind it was out. Values
are as the latest release states them, not as first published.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pandas as pd
import requests

PACKAGE = "https://www.federalreserve.gov/econres/files/data_only_package.zip"
MEMBER = "data_only_package/HISTDATA.TXT"


class FrbusSource:
    name = "frbus"

    def __init__(self):
        self.session = requests.Session()
        self._table: pd.DataFrame | None = None

    def close(self):
        self.session.close()

    def _history(self) -> pd.DataFrame:
        if self._table is None:
            r = self.session.get(PACKAGE, timeout=120)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z, z.open(MEMBER) as f:
                table = pd.read_csv(f)
            table.index = pd.PeriodIndex(table.pop("OBS"), freq="Q")
            self._table = table
        return self._table

    def fetch_with_vintages(self, series_id, start=None) -> pd.DataFrame:
        return self.fetch_current(series_id)

    def fetch_current(self, series_id) -> pd.DataFrame:
        s = pd.to_numeric(self._history()[series_id], errors="coerce").dropna()
        today = dt.date.today()
        return pd.DataFrame({
            # First day of the quarter, as the other quarterly series are stored.
            "observation_date": [p.start_time.date() for p in s.index],
            # Known from the month after the quarter ends, and never after today.
            "vintage_date": [min((p + 1).start_time.date() + dt.timedelta(days=31), today)
                             for p in s.index],
            "value": s.to_numpy(),
        })

    def series(self, series_id) -> pd.Series:
        """The column on a monthly PeriodIndex, first month of each quarter."""
        s = pd.to_numeric(self._history()[series_id], errors="coerce").dropna()
        s.index = s.index.asfreq("M", how="start")
        return s

    def describe(self, series_id) -> dict:
        return {"title": f"FRB/US {series_id}", "units": "%", "frequency": "quarterly",
                "has_vintages": False}
