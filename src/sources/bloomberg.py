"""Bloomberg Desktop API adapter.

Talks to the terminal running on this machine (bbcomm, localhost:8194), so it
works only where a Bloomberg terminal is logged in. No credentials are handled
here: the terminal's own login is the entitlement.

blpapi is not on PyPI, so it is not in requirements.txt (the hosted app could
not install it, and has no terminal to talk to anyway). Install it locally with:

    pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

Codes are a Bloomberg ticker, optionally with a field after a bar, and
optionally a forecast period after a second bar:

    "US CDS USD SR 5Y D14 Corp"          PX_LAST by default
    "LF98OAS Index|PX_LAST"
    "SPX Index|BEST_EPS|3FY"             consensus for the third fiscal year out

The third part sets BEST_FPERIOD_OVERRIDE, which is how the terminal is asked
for a consensus estimate at a fixed distance ahead rather than for the current
fiscal year. Without it, BEST_EPS means "this year" and the horizon it refers
to shortens as the year runs on, which would make the series a mix of horizons.

LICENSING. Desktop API data is for use on this machine. It must not be
redistributed, and `ingest.py publish` pushes a snapshot to a public branch, so
publish refuses to run while any Bloomberg series is in the store.

ON VINTAGES. What this adapter is for — CDS spreads, index spreads — are traded
prices, never revised, so each value is dated at its own observation. Same
treatment as CAPE and the breakevens. A consensus estimate is the same shape of
thing for a different reason: the number on a given date is what analysts
thought on that date, and later changes of mind arrive as new observations
rather than as corrections to old ones.
"""

from __future__ import annotations

import datetime as dt
import os

import pandas as pd

from src.sources.base import tidy_vintages

DEFAULT_FIELD = "PX_LAST"
COLUMNS = ["series_id", "observation_date", "vintage_date", "value"]


class BloombergSource:
    name = "bloomberg"

    def __init__(self, host: str | None = None, port: int | None = None,
                 session=None, timeout_ms: int = 30000):
        self.host = host or os.environ.get("BBG_HOST", "localhost")
        self.port = int(port or os.environ.get("BBG_PORT", 8194))
        self.timeout_ms = timeout_ms
        self._session = session
        self._refdata = None

    # ---------- transport: the only part that touches blpapi ----------

    def _service(self):
        if self._refdata is not None:
            return self._refdata
        import blpapi
        if self._session is None:
            opts = blpapi.SessionOptions()
            opts.setServerHost(self.host)
            opts.setServerPort(self.port)
            self._session = blpapi.Session(opts)
            if not self._session.start():
                raise RuntimeError(
                    f"No Bloomberg terminal answering on {self.host}:{self.port}. "
                    f"Log in to the terminal on this machine and try again.")
        if not self._session.openService("//blp/refdata"):
            raise RuntimeError("Bloomberg refdata service would not open.")
        self._refdata = self._session.getService("//blp/refdata")
        return self._refdata

    def _drain(self) -> list:
        import blpapi
        msgs = []
        while True:
            ev = self._session.nextEvent(self.timeout_ms)
            msgs.extend(m for m in ev)
            if ev.eventType() == blpapi.Event.RESPONSE:
                return msgs
            if ev.eventType() == blpapi.Event.TIMEOUT:
                raise RuntimeError("Bloomberg did not answer in time.")

    def _history(self, ticker: str, field: str, start: dt.date | None,
                 period: str | None = None) -> list[tuple[dt.date, float]]:
        svc = self._service()
        r = svc.createRequest("HistoricalDataRequest")
        r.getElement("securities").appendValue(ticker)
        r.getElement("fields").appendValue(field)
        r.set("startDate", (start or dt.date(1900, 1, 1)).strftime("%Y%m%d"))
        r.set("endDate", dt.date.today().strftime("%Y%m%d"))
        if period:
            o = r.getElement("overrides").appendElement()
            o.setElement("fieldId", "BEST_FPERIOD_OVERRIDE")
            o.setElement("value", period)
        self._session.sendRequest(r)
        rows = []
        for m in self._drain():
            if not m.hasElement("securityData"):
                continue
            sd = m.getElement("securityData")
            if sd.hasElement("securityError"):
                raise RuntimeError(f"Bloomberg: {ticker}: "
                                   f"{sd.getElement('securityError').getElementAsString('message')}")
            fd = sd.getElement("fieldData")
            for i in range(fd.numValues()):
                p = fd.getValueAsElement(i)
                if p.hasElement(field):
                    d = p.getElementAsDatetime("date")
                    rows.append((dt.date(d.year, d.month, d.day), p.getElementAsFloat(field)))
        return rows

    def _reference(self, ticker: str, fields: list[str]) -> dict:
        svc = self._service()
        r = svc.createRequest("ReferenceDataRequest")
        r.getElement("securities").appendValue(ticker)
        for f in fields:
            r.getElement("fields").appendValue(f)
        self._session.sendRequest(r)
        out: dict = {}
        for m in self._drain():
            if not m.hasElement("securityData"):
                continue
            arr = m.getElement("securityData")
            for i in range(arr.numValues()):
                fd = arr.getValueAsElement(i).getElement("fieldData")
                out.update({f: fd.getElementAsString(f) for f in fields if fd.hasElement(f)})
        return out

    # ---------- the Source interface ----------

    @staticmethod
    def split(code: str) -> tuple[str, str, str | None]:
        ticker, _, rest = code.partition("|")
        field, _, period = rest.partition("|")
        return (ticker.strip(), (field.strip() or DEFAULT_FIELD),
                period.strip() or None)

    @staticmethod
    def _frame(series_id: str, rows, vintage: dt.date | None) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        obs = [d for d, _ in rows]
        return pd.DataFrame({
            "series_id": series_id,
            "observation_date": obs,
            # A traded price is never revised: it was known on its own date.
            "vintage_date": [vintage] * len(rows) if vintage else obs,
            "value": [float(v) for _, v in rows],
        })

    def fetch_with_vintages(self, series_id: str, start: dt.date | None = None) -> pd.DataFrame:
        ticker, field, period = self.split(series_id)
        rows = self._history(ticker, field, start, period)
        return tidy_vintages(self._frame(series_id, rows, None))

    def fetch_current(self, series_id: str) -> pd.DataFrame:
        ticker, field, period = self.split(series_id)
        recent = dt.date.today() - dt.timedelta(days=14)
        rows = self._history(ticker, field, recent, period)
        return self._frame(series_id, rows, dt.date.today())

    def describe(self, series_id: str) -> dict:
        ticker, field, period = self.split(series_id)
        # LONG_COMP_NAME first: NAME is abbreviated to fit a terminal column and
        # loses the distinction that matters here. S5RETL comes back as "S&P 500
        # CONS DISCR", which is the parent sector, not the retail industry group
        # the ticker actually is.
        info = self._reference(ticker, ["LONG_COMP_NAME", "NAME"])
        units = f"{field} @{period}" if period else field
        return {"title": info.get("LONG_COMP_NAME") or info.get("NAME", ticker),
                "units": units, "frequency": "D", "has_vintages": True}

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.stop()
            except Exception:
                pass
