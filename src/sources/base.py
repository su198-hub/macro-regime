"""Source adapter interface.

Everything downstream of this file is source-agnostic. Swapping FRED for
Macrobond means writing one new class here and changing one line in ingest.py.
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
