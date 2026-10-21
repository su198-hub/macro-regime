"""QuantGov RegData: yearly counts of binding restrictions in the Code of
Federal Regulations.

RegData counts words and phrases that create an obligation or a prohibition
("shall", "must", "may not", "prohibited", "required") in every part of the
CFR, once a year. The bulk download is a zip of several gigabytes, nearly all
of it industry-relevance detail; the one table needed here, restrictions per
part per year, is about 90 MB. So the archive is read in place over HTTP range
requests and only that member is fetched, then summed to one total a year.

A code in config/sources.yml is "<zip url>|<member path>". Data are public,
free to use with attribution to QuantGov (Mercatus Center).
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pandas as pd
import requests


class _RangeFile(io.RawIOBase):
    """A remote file read on demand with HTTP range requests, so zipfile can
    seek to the central directory and one member without the whole archive."""

    def __init__(self, url: str, session: requests.Session):
        self.url, self.session, self.pos = url, session, 0
        head = session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        r.raise_for_status()
        data = r.content
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


class QuantgovSource:
    name = "quantgov"

    def __init__(self):
        self.session = requests.Session()

    def close(self):
        self.session.close()

    def _annual(self, code: str) -> pd.Series:
        url, member = code.split("|", 1)
        raw = io.BufferedReader(_RangeFile(url, self.session), buffer_size=8 << 20)
        with zipfile.ZipFile(raw) as z, z.open(member) as f:
            parts = pd.read_csv(f, usecols=["year", "restrictions"])
        return parts.groupby("year")["restrictions"].sum().astype(float)

    def fetch_with_vintages(self, series_id, start=None) -> pd.DataFrame:
        # One release holds every year as RegData currently counts it; there
        # is no revision history to recover, so it is stored as one vintage.
        return self.fetch_current(series_id)

    def fetch_current(self, series_id) -> pd.DataFrame:
        annual = self._annual(series_id)
        # Dated at year end: a year's count describes the Code as it stood
        # during that year, and is published the following year.
        return pd.DataFrame({
            "observation_date": [dt.date(int(y), 12, 31) for y in annual.index],
            "vintage_date": dt.date.today(),
            "value": annual.to_numpy(),
        })

    def describe(self, series_id) -> dict:
        return {"title": "RegData: restrictions in the Code of Federal Regulations, total",
                "units": "count", "frequency": "annual", "has_vintages": False}
