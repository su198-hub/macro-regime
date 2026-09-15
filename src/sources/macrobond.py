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

Vintages. get_all_vintage_series returns every stored vintage as a complete
copy of the series, each stamped with the time it was published. The oldest
copy carries no timestamp: it is the history as it stood before Macrobond
began recording revisions. It is dated at the first recorded revision, which
is conservative. A rewind to an earlier date sees none of that history rather
than a revised version of it, the same rule ALFRED data follows.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .base import tidy_vintages


class MacrobondSource:
    name = "macrobond"

    def __init__(self, client: str = "com", client_id=None, client_secret=None):
        if client == "web":
            from macrobond_data_api.web import WebClient
            self._client = WebClient(client_id, client_secret)
        else:
            from macrobond_data_api.com import ComClient
            self._client = ComClient()
        self.api = self._client.__enter__()

    def close(self):
        self._client.__exit__(None, None, None)

    def fetch_with_vintages(self, series_id, start=None) -> pd.DataFrame:
        result = self.api.get_all_vintage_series(series_id)
        vintages = list(result.series)
        if not vintages:
            return self.fetch_current(series_id)

        stamps = [v.revision_time_stamp for v in vintages]
        # Every vintage is a full copy of the series, so keep only values that
        # are new or changed against the last kept value as each one arrives.
        # Concatenating first would build (observations x vintages) rows, which
        # for a long daily series runs to tens of millions.
        last = pd.Series(dtype=float)
        changes = []
        for i, v in enumerate(vintages):
            stamp = stamps[i] or next((s for s in stamps[i + 1:] if s), None)
            vintage = stamp.date() if stamp else dt.date.today()
            frame = self._flatten(series_id, v, vintage)
            frame = frame[frame["observation_date"] <= vintage]
            cur = frame.set_index("observation_date")["value"]
            cur = cur[~cur.index.duplicated(keep="last")]
            prev = last.reindex(cur.index)
            changed = prev.isna() | (cur != prev)
            if changed.any():
                changes.append(frame[frame["observation_date"].isin(cur.index[changed])])
                last = pd.concat([last.drop(cur.index[changed], errors="ignore"), cur[changed]])
        if not changes:
            return pd.DataFrame(columns=["series_id", "observation_date", "vintage_date", "value"])
        out = tidy_vintages(pd.concat(changes, ignore_index=True))
        if start:
            out = out[out["observation_date"] >= start]
        return out

    def fetch_current(self, series_id) -> pd.DataFrame:
        s = self.api.get_one_series(series_id)
        return tidy_vintages(self._flatten(series_id, s, dt.date.today()))

    def unrevised(self, series_id) -> pd.DataFrame:
        """First-print values only. The honest input to any backtest."""
        s = self.api.get_nth_release(0, [series_id])[0]
        return self._flatten(series_id, s, dt.date.today())

    def describe(self, series_id) -> dict:
        md = self.api.get_one_series(series_id).metadata
        return {
            "title": md.get("FullDescription") or md.get("Description", ""),
            "units": md.get("DisplayUnit", ""),
            "frequency": md.get("Frequency", ""),
            "has_vintages": md.get("FirstRevisionTimeStamp") is not None,
        }

    @staticmethod
    def _flatten(series_id, series, vintage) -> pd.DataFrame:
        return pd.DataFrame({
            "series_id": series_id,
            "observation_date": pd.to_datetime(pd.Series(series.dates)).dt.tz_localize(None).dt.date,
            "vintage_date": vintage,
            "value": pd.to_numeric(pd.Series(series.values), errors="coerce"),
        }).dropna(subset=["value"])
