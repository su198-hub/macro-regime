"""Macrobond adapter.

Requires a Data+, Professional or Enterprise licence. A standard Analysis
seat will not authenticate. Check before wiring this in:

    pip install macrobond-data-api
    python -c "from macrobond_data_api.com import ComClient; \
               print(ComClient().__enter__().get_one_series('usgdp').values[:3])"

Two clients, and the choice determines where your ingestion can run:

  ComClient  - talks to the local desktop application, no separate credentials
               because they come from Macrobond Analysis itself. Windows only,
               and the app must be installed on the same machine. Rules out
               Linux, containers and hosted schedulers.

  WebClient  - REST, takes a client id and secret, runs anywhere. A separate
               entitlement on top of Data+. Worth pricing if you want this
               pipeline running somewhere other than a desk machine.

Not every series carries revision history. Macrobond flags the ones that do,
so call `has_vintages` across your whole universe before you assume coverage
and build a backtest on top of it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd


class MacrobondSource:
    name = "macrobond"

    def __init__(self, client: str = "auto", client_id=None, client_secret=None):
        import macrobond_data_api as mb

        if client == "com":
            from macrobond_data_api.com import ComClient
            self.api = ComClient().__enter__()
        elif client == "web":
            from macrobond_data_api.web import WebClient
            self.api = WebClient(client_id, client_secret).__enter__()
        else:
            # Uses the Web API if credentials are in the system keyring,
            # otherwise falls back to the local desktop COM API.
            self.api = mb

    def fetch_with_vintages(self, series_id, start=None) -> pd.DataFrame:
        """Full revision array for one series.

        Macrobond's own recommended pattern is a one-off full backfill like
        this, then incremental syncing with get_many_series_with_revisions
        using the timestamps from the previous pull. Build the history once,
        then switch to incremental — see sync_incremental below.
        """
        info = self.api.get_revision_info(series_id)[0]
        if not info.stores_revisions:
            s = self.api.get_one_series(series_id)
            return self._flatten(series_id, s, dt.date.today())

        frames = []
        for vintage in info.vintage_time_stamps:
            s = self.api.get_vintage_series(series_id, vintage)
            frames.append(self._flatten(series_id, s, vintage.date()))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if start:
            out = out[out["observation_date"] >= start]
        return out

    def fetch_current(self, series_id) -> pd.DataFrame:
        return self._flatten(series_id, self.api.get_one_series(series_id),
                             dt.date.today())

    def unrevised(self, series_id) -> pd.DataFrame:
        """First-print values only. The honest input to any backtest."""
        return self._flatten(series_id, self.api.get_nth_release(0, series_id),
                             dt.date.today())

    def describe(self, series_id) -> dict:
        s = self.api.get_one_series(series_id)
        info = self.api.get_revision_info(series_id)[0]
        return {
            "title": s.metadata.get("Description", ""),
            "units": s.metadata.get("DisplayUnit", ""),
            "frequency": s.metadata.get("Frequency", ""),
            "has_vintages": bool(info.stores_revisions),
        }

    @staticmethod
    def _flatten(series_id, series, vintage) -> pd.DataFrame:
        dates = pd.to_datetime(pd.Series(series.dates)).dt.date
        return pd.DataFrame({
            "series_id": series_id,
            "observation_date": dates,
            "vintage_date": vintage,
            "value": pd.to_numeric(pd.Series(series.values), errors="coerce"),
        }).dropna(subset=["value"])
