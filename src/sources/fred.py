"""FRED and ALFRED adapter.

ALFRED is the part that matters. Passing the full realtime range to the
observations endpoint returns every vintage of every observation, which is
exactly the shape the store wants. Most series have vintages back to the
mid-1990s; market series like breakevens are never revised and come back as
a single vintage each.

Get a free key at https://fredaccount.stlouisfed.org/apikeys
"""

from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd
import requests

BASE = "https://api.stlouisfed.org/fred"
FULL_RANGE = {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31"}


class FredSource:
    name = "fred"

    def __init__(self, api_key: str | None = None, pause: float = 0.6):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No FRED key. Set FRED_API_KEY or pass api_key=..."
            )
        self.pause = pause
        self.session = requests.Session()

    def _get(self, endpoint: str, **params) -> dict:
        params.update(api_key=self.api_key, file_type="json")
        for attempt in range(4):
            r = self.session.get(f"{BASE}/{endpoint}", params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            time.sleep(self.pause)
            return r.json()
        raise RuntimeError(f"FRED rate limit not clearing on {endpoint}")

    def fetch_with_vintages(self, series_id, start=None) -> pd.DataFrame:
        params = dict(series_id=series_id, **FULL_RANGE)
        if start:
            params["observation_start"] = start.isoformat()

        rows, offset = [], 0
        while True:
            payload = self._get("series/observations", offset=offset,
                                limit=100_000, **params)
            batch = payload.get("observations", [])
            rows.extend(batch)
            if len(batch) < 100_000:
                break
            offset += 100_000

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        out = pd.DataFrame({
            "series_id": series_id,
            "observation_date": pd.to_datetime(df["date"]).dt.date,
            # realtime_start is the date the value first appeared: the vintage.
            "vintage_date": pd.to_datetime(df["realtime_start"]).dt.date,
            "value": df["value"].astype(float),
        })
        return out.drop_duplicates(
            ["series_id", "observation_date", "vintage_date"], keep="last"
        )

    def fetch_current(self, series_id) -> pd.DataFrame:
        payload = self._get("series/observations", series_id=series_id)
        rows = payload.get("observations", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        return pd.DataFrame({
            "series_id": series_id,
            "observation_date": pd.to_datetime(df["date"]).dt.date,
            "vintage_date": dt.date.today(),
            "value": df["value"].astype(float),
        })

    def describe(self, series_id) -> dict:
        info = self._get("series", series_id=series_id)["seriess"][0]
        return {
            "title": info.get("title", ""),
            "units": info.get("units_short", ""),
            "frequency": info.get("frequency_short", ""),
            "has_vintages": True,
        }
