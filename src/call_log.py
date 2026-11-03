"""The live call log: one row per publish, never rewritten.

Every grade of the model so far is on history the model was tuned against.
The only record that settles how good it is starts when calls are written
down before their outcome is known, so `ingest.py publish` appends the call it
publishes to calls.csv on the data branch, carrying the rows already there
across the force-push, and keeps a copy on the desk machine.

A row is what the dashboard showed on the day: the month scored, the regime
called and the odds behind it, each driver's reading, the newest provisional
month if there was one, and fingerprints of the config and code, so a call can
be traced to the model that made it. The outcome for a row is known about
three years and four months after its month.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
from pathlib import Path

import pandas as pd

FILE = "calls.csv"
KEY = "published_at"


def config_hash(*paths: str | Path) -> str:
    """A fingerprint of the config files' contents, line endings aside, so a
    checkout on Windows and one on Linux agree."""
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def row(results: dict, published_at: str, framework: str, config: str, commit: str) -> dict:
    """The call in `results` (drivers.compute plus regimes.run) as one log row."""
    probs = results["probabilities"].dropna(how="all")
    if probs.empty:
        raise ValueError("no month could be scored, so there is no call to log")
    month = probs.index[-1]
    call = results["calls"].loc[month]
    drivers = results["drivers"]
    out = {KEY: published_at, "month": pd.Timestamp(month).strftime("%Y-%m"),
           "framework": framework, "config_hash": config, "code_commit": commit,
           "called": call["called"], "leading": call["leading"]}
    out.update({f"p_{r}": round(float(p), 4) for r, p in probs.loc[month].items()})
    out.update({f"d_{d}": round(float(drivers.at[month, d]), 4)
                for d in drivers.columns if not d.endswith("__coverage")})
    prov = (results.get("provisional") or [None])[-1]
    out["provisional_month"] = pd.Timestamp(prov["month"]).strftime("%Y-%m") if prov else ""
    return out


def read(text: str | None) -> pd.DataFrame:
    """A log read back as text, blanks kept blank, so rows compare as written."""
    if not (text and text.strip()):
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def merge(*logs: pd.DataFrame, new: dict | None = None) -> pd.DataFrame:
    """Every row from every copy, once each, oldest first. A row is never
    changed: if two copies disagree on a publish, the first copy wins."""
    parts = [x for x in logs if not x.empty]
    if new is not None:
        parts.append(pd.DataFrame([{k: str(v) for k, v in new.items()}]))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).drop_duplicates(KEY, keep="first")
    return out.sort_values(KEY, kind="stable").reset_index(drop=True)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
