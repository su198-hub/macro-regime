"""S&P Capital IQ Pro adapter.

Credentials come from the environment and are never read from a file in the
repo:

    CIQ_USERNAME        your Capital IQ Pro login
    CIQ_PASSWORD        its password
    CIQ_ENDPOINT        optional; overrides the default GDS endpoint
    CIQ_FUNCTION        optional; overrides the historical-series function

On Windows PowerShell, for one session:

    $env:CIQ_USERNAME = "you@firm.com"
    $env:CIQ_PASSWORD = "..."

The API is entitled separately from a Pro seat at most institutions, so the
first thing to do is not to backfill but to run:

    python ingest.py ciq-probe --code "IQ_CDS_SPREAD_5YR" --id "IQ12345"

which makes one small call and prints the raw response. Two reasons. It tells
you in ten seconds whether the entitlement exists, with the vendor's own error
message rather than a guess. And S&P has several request shapes across GDS, the
newer MI API and Xpressfeed; the parser below targets GDS, and the probe shows
which one this account actually answers with so the seam in `_rows` can be
pointed at it without hunting.

ON VINTAGES. A CDS spread or a bond spread is a traded price: it is never
revised, so one vintage per observation dated at the observation itself is
correct, not a gap in the record. That is the same treatment CAPE and the
breakevens already get. Company fundamentals are a different matter — CIQ
restates them, and a point-in-time read needs their as-reported data rather
than the current view. `fetch_with_vintages` refuses fundamentals mnemonics
rather than quietly returning a restated history dressed as vintages, because
that would put revised numbers into months that never saw them and every
backtest downstream would be wrong in a way nobody could see.
"""

from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd
import requests

from src.sources.base import tidy_vintages

# The GDS REST service. Overridable because S&P has moved it before and
# entitlements differ by contract.
DEFAULT_ENDPOINT = ("https://api-ciq.marketintelligence.spglobal.com"
                    "/gdsapi/rest/v3/clientservice.json")
DEFAULT_FUNCTION = "GDSHE"   # historical series

# Mnemonics whose history CIQ restates. Allowed for a current read, refused for
# a vintage read: see the note above.
RESTATED_HINTS = ("IQ_TOTAL_REV", "IQ_CAPEX", "IQ_EBITDA", "IQ_NI",
                  "IQ_TOTAL_DEBT", "IQ_GROSS_MARGIN", "IQ_EBIT")


class CiqAuthError(RuntimeError):
    """Bad credentials, or the API is not on this seat's entitlement."""


class CiqSource:
    name = "ciq"

    def __init__(self, username: str | None = None, password: str | None = None,
                 endpoint: str | None = None, function: str | None = None,
                 pause: float = 0.3, session=None):
        self.username = username or os.environ.get("CIQ_USERNAME")
        self.password = password or os.environ.get("CIQ_PASSWORD")
        if not (self.username and self.password):
            raise RuntimeError(
                "No Capital IQ credentials. Set CIQ_USERNAME and CIQ_PASSWORD "
                "in the environment; do not put them in a file in the repo."
            )
        self.endpoint = endpoint or os.environ.get("CIQ_ENDPOINT") or DEFAULT_ENDPOINT
        self.function = function or os.environ.get("CIQ_FUNCTION") or DEFAULT_FUNCTION
        self.pause = pause
        self.session = session or requests.Session()
        self.session.auth = (self.username, self.password)

    # ---------- transport ----------

    def request(self, body: dict) -> dict:
        """One POST, with the vendor's own error surfaced rather than masked."""
        for attempt in range(4):
            r = self.session.post(self.endpoint, json=body, timeout=90,
                                  headers={"Content-Type": "application/json"})
            if r.status_code in (401, 403):
                raise CiqAuthError(
                    f"Capital IQ refused the request ({r.status_code}). Either the "
                    f"credentials are wrong or the API is not entitled on this seat "
                    f"— a Pro login alone usually is not enough. Response: "
                    f"{r.text[:300]}"
                )
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            time.sleep(self.pause)
            return r.json()
        raise RuntimeError("Capital IQ rate limit is not clearing")

    def _body(self, identifier: str, mnemonic: str,
              start: dt.date | None, end: dt.date | None) -> dict:
        props: dict[str, str] = {"periodType": "IQ_DAILY"}
        if start:
            props["startDate"] = start.strftime("%m/%d/%Y")
        props["endDate"] = (end or dt.date.today()).strftime("%m/%d/%Y")
        return {"inputRequests": [{"function": self.function,
                                   "identifier": identifier,
                                   "mnemonic": mnemonic,
                                   "properties": props}]}

    # ---------- parsing ----------

    @staticmethod
    def _rows(payload: dict) -> list[dict]:
        """Pull (date, value) pairs out of a GDS response.

        THE SEAM. GDS nests results as GDSSDKResponse -> Rows -> Row -> Values,
        and the exact nesting differs between functions and between the GDS and
        MI APIs. Run `ingest.py ciq-probe` once with real credentials, look at
        the printed response, and adjust here if the shape differs. Everything
        else in this file is shape-independent.
        """
        out: list[dict] = []
        for block in payload.get("GDSSDKResponse") or []:
            err = block.get("ErrMsg")
            if err:
                raise RuntimeError(f"Capital IQ: {err}")
            headers = [str(h).lower() for h in (block.get("Headers") or [])]
            for row in block.get("Rows") or []:
                values = row.get("Row") if isinstance(row, dict) else row
                if not values:
                    continue
                if len(values) >= 2:
                    out.append({"date": values[0], "value": values[1]})
                elif headers and len(values) == 1:
                    out.append({"date": headers[0], "value": values[0]})
        return out

    @staticmethod
    def _frame(series_id: str, rows: list[dict], vintage: dt.date | None) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=["series_id", "observation_date", "vintage_date", "value"])
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # GDS returns MM/DD/YYYY. Try that first so the common case is parsed
        # unambiguously — inference alone reads 01/02 as 2 January in some
        # locales and 1 February in others, which would silently shift a whole
        # series by weeks. Fall back to inference if the probe shows a
        # different format.
        parsed = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(df["date"], errors="coerce", format="mixed")
        df["date"] = parsed
        df = df.dropna(subset=["value", "date"])
        if df.empty:
            return pd.DataFrame(
                columns=["series_id", "observation_date", "vintage_date", "value"])
        obs = df["date"].dt.date
        return pd.DataFrame({
            "series_id": series_id,
            "observation_date": obs,
            # A traded price is never revised, so it was known on its own date.
            "vintage_date": [vintage] * len(df) if vintage else obs,
            "value": df["value"].astype(float),
        })

    # ---------- the Source interface ----------

    @staticmethod
    def split(code: str) -> tuple[str, str]:
        """'IQ12345|IQ_CDS_SPREAD_5YR' -> identifier, mnemonic."""
        if "|" not in code:
            raise ValueError(
                f"Capital IQ codes are 'identifier|mnemonic', got {code!r}. "
                f"The identifier names the entity and the mnemonic names the "
                f"field; one without the other cannot be resolved."
            )
        identifier, mnemonic = (part.strip() for part in code.split("|", 1))
        return identifier, mnemonic

    def fetch_with_vintages(self, series_id: str, start: dt.date | None = None) -> pd.DataFrame:
        identifier, mnemonic = self.split(series_id)
        if any(h in mnemonic.upper() for h in RESTATED_HINTS):
            raise ValueError(
                f"{mnemonic} is a restated fundamental. Capital IQ's default view "
                f"carries today's numbers back through history, so treating it as a "
                f"vintage record would put revised values into months that never saw "
                f"them. Fetch it with fetch_current, or wire the as-reported "
                f"endpoint before backfilling."
            )
        payload = self.request(self._body(identifier, mnemonic, start, None))
        return tidy_vintages(self._frame(series_id, self._rows(payload), None))

    def fetch_current(self, series_id: str) -> pd.DataFrame:
        identifier, mnemonic = self.split(series_id)
        payload = self.request(self._body(identifier, mnemonic, None, None))
        return self._frame(series_id, self._rows(payload), dt.date.today())

    def describe(self, series_id: str) -> dict:
        identifier, mnemonic = self.split(series_id)
        return {
            "title": f"{identifier} {mnemonic}",
            "units": "",
            "frequency": "D",
            # True for market prices, which is what this adapter is for; a
            # restated fundamental never reaches here.
            "has_vintages": True,
        }

    def close(self) -> None:
        self.session.close()
