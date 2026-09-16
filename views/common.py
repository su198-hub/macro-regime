"""Loaders shared by every page. Cached, so switching pages does not recompute.

Where the data comes from, in order:
  1. MACRO_REGIME_DATA_URL set: a snapshot published by `ingest.py publish`.
     This is how the hosted app gets real data it cannot fetch itself.
  2. Otherwise MACRO_REGIME_DB (default data/regime.duckdb), a local store,
     seeded with demo data only if MACRO_REGIME_SEED_DEMO=1 and it is empty.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import countries as ctry
from src import overrides as ov
from src.drivers import compute, expected_release, indicator_inputs, load_config
from src.regimes import fit, load_regimes, probabilities, run
from src.store import Store

REPO_URL = "https://github.com/su198-hub/macro-regime"
SNAPSHOT_CHECK_SECONDS = 600
COUNTRY_KEY = "mr_country"


@st.cache_data(ttl=300, show_spinner=False)
def registry() -> dict:
    return ctry.load_registry()


def country() -> dict:
    """The country being viewed: code, label and its three config paths."""
    return ctry.resolve(registry(), st.session_state.get(COUNTRY_KEY))


# Paths of the country in view. Read through these, never hard-coded, so a
# second country needs no change here.
def INDICATORS() -> str:  # noqa: N802 - reads as a constant at call sites
    return country()["indicators"]


def REGIMES() -> str:     # noqa: N802
    return country()["regimes"]


def data_url() -> str | None:
    return os.environ.get("MACRO_REGIME_DATA_URL") or None


@st.cache_data(ttl=SNAPSHOT_CHECK_SECONDS, show_spinner=False)
def snapshot_manifest() -> dict | None:
    """The published snapshot's manifest, or None if not configured or unreachable."""
    url = data_url()
    if not url:
        return None
    from src.snapshot import read_manifest
    try:
        return read_manifest(url)
    except Exception:
        return None


def data_version() -> str:
    """Changes whenever a new snapshot is published, so caches rebuild."""
    m = snapshot_manifest()
    return m["published_at"] if m else "local"


@st.cache_resource(show_spinner="Loading the latest published data…", max_entries=1)
def _store_for(version: str):
    if version != "local":
        from src.snapshot import load_snapshot
        safe = version.replace(":", "").replace("+", "_")
        path = Path(tempfile.gettempdir()) / "macro-regime" / f"snapshot-{safe}.duckdb"
        try:
            return load_snapshot(data_url(), path)
        except Exception as exc:
            st.warning(f"Could not load the published data ({exc}). Showing local data instead.")

    # Point MACRO_REGIME_DB outside OneDrive; sync locks the file mid-write.
    store = Store(os.environ.get("MACRO_REGIME_DB", "data/regime.duckdb"))
    # Hosted deploys start from an empty disk. Opt-in only, so synthetic data
    # never lands in a real store by accident.
    if os.environ.get("MACRO_REGIME_SEED_DEMO") == "1" and store.coverage().empty:
        from ingest import load_demo
        # Demo data seeds every live country, so the menu works without a vendor.
        reg = ctry.load_registry()
        for code in ctry.live_codes(reg):
            load_demo(store, load_config(ctry.resolve(reg, code)["indicators"]))
    return store


def get_store():
    return _store_for(data_version())


OVERRIDE_KEY = "mr_overrides"


def base_config() -> tuple[dict, dict]:
    """The published defaults for the country in view, before any override."""
    here = country()
    return load_config(here["indicators"]), load_regimes(here["regimes"])


def get_overrides() -> dict:
    """This session's overrides for the country in view.

    Kept per country: a weight on US consumer spending means nothing to Japan,
    and switching country should not carry one country's settings into another.
    """
    cfg, reg = base_config()
    held = st.session_state.get(OVERRIDE_KEY) or {}
    return ov.clean(held.get(country()["code"]), cfg, reg)


def set_overrides(new: dict | None) -> None:
    cfg, reg = base_config()
    held = dict(st.session_state.get(OVERRIDE_KEY) or {})
    held[country()["code"]] = ov.clean(new, cfg, reg)
    st.session_state[OVERRIDE_KEY] = held
    # Results are cached per country and override signature, so nothing to
    # clear here.


# Control room widgets that carry a key: their value lives in session_state and
# survives a reset unless it is cleared with the overrides, which would let a
# reset silently re-apply on the next Apply.
CONTROL_WIDGETS = ("sal::", "w::", "c::", "s::")


def reset_overrides() -> None:
    set_overrides(None)
    for k in [k for k in st.session_state
              if isinstance(k, str) and k.startswith(CONTROL_WIDGETS)]:
        del st.session_state[k]


def overrides_active() -> int:
    return ov.count(get_overrides())


# One entry per override set a reader tries, plus the default. Enough to flip
# back and forth without recomputing, small enough not to hold the whole
# history many times over.
@st.cache_data(ttl=900, max_entries=8, show_spinner="Rescoring the model…")
def _results_for(vintage: dt.date, version: str, overrides: str = "{}",
                 indicators: str = "", regimes: str = ""):
    store = _store_for(version)
    cfg, reg = ov.apply(load_config(indicators), load_regimes(regimes), ov.parse(overrides))
    res = compute(store, cfg, vintage)
    if res["drivers"].empty:
        return None
    res.update(run(res["drivers"], reg), config=cfg, regimes=reg)
    gate_on = str(reg["settings"].get("fit_gate", "none")) == "closer_than_neutral"
    for reading in res["provisional"]:
        frame = pd.DataFrame([reading["drivers"]], index=[reading["month"]])
        p = probabilities(frame, reg).iloc[0]
        reading["probabilities"] = p
        reading["leading"] = p.idxmax()
        f = fit(frame, reg).iloc[0]
        reading["fits"] = bool(f["fits"]) if gate_on else True
        reading["fit"] = f["fit"] if gate_on else "clear"
        reading["fit_distance"], reading["fit_threshold"] = f["distance"], f["threshold"]
        reading["schedule"] = release_schedule(res, reading, vintage)
    return res


FREQ_LABEL = {252: "Daily", 52: "Weekly", 12: "Monthly", 4: "Quarterly", 1: "Annual"}


def release_schedule(res: dict, reading: dict, vintage: dt.date) -> pd.DataFrame:
    """The provisional month's status table, with frequency and expected release.

    Expected release is each source series' usual lag after the period ends,
    taken from its own recent vintages; for an indicator built from several
    series, the latest of them.
    """
    inputs = indicator_inputs(res["config"])
    freq, lags = res["frequency"], res["release_lags"]
    last_obs = res.get("last_obs", {})
    month_start = reading["month"] - pd.offsets.MonthBegin(1)
    status = reading["status"].copy()
    expected, frequency = [], []
    today = pd.Timestamp(vintage)
    for row in status.itertuples():
        series = inputs.get(row.key, [])
        fastest = max((freq.get(s, 12) for s in series), default=12)
        frequency.append(FREQ_LABEL.get(fastest, "Monthly"))
        if row.status in ("carried", "pending") and series:
            # Only inputs that actually hold the month back: monthly or faster
            # series without an observation for it. Quarterly and annual inputs
            # are carried forward and never do.
            waiting_on = [s for s in series if freq.get(s, 12) >= 12
                          and (pd.isna(last_obs.get(s)) or last_obs[s] < month_start)]
            dates = [expected_release(reading["month"], freq.get(s, 12), lags.get(s, np.nan))
                     for s in (waiting_on or series) if not pd.isna(lags.get(s, np.nan))]
            when = max(dates) if dates else pd.NaT
            expected.append(when if pd.isna(when) or when > today else today)
        else:
            expected.append(pd.NaT)
    status["frequency"] = frequency
    status["expected"] = expected
    return status


def get_results(vintage: dt.date, use_overrides: bool = True):
    """Model output for a vintage, for the country in view, under this
    session's settings.

    use_overrides=False gives the published defaults, for side-by-side
    comparison in the control room.
    """
    signature = ov.signature(get_overrides()) if use_overrides else "{}"
    here = country()
    return _results_for(vintage, data_version(), signature,
                        here["indicators"], here["regimes"])


def is_demo(store) -> bool:
    return "demo" in store.sources()


def published_note() -> str:
    """'Data published 15 September 2026 at 07:23 UTC' on a snapshot, else ''.

    The time matters: a second publish on the same day otherwise looks
    identical, and there is no way to tell the app has picked it up.
    """
    m = snapshot_manifest()
    if not m:
        return ""
    when = dt.datetime.fromisoformat(m["published_at"]).astimezone(dt.timezone.utc)
    vintage = dt.date.fromisoformat(m["latest_vintage"])
    return (f"Data published {when:%B} {when.day}, {when.year} at {when:%H:%M} UTC; "
            f"latest vintage {vintage:%B} {vintage.day}.")


def country_picker(container=None, label_visibility: str = "visible") -> dict:
    """The country menu. One live country still gets a menu, so the shape of
    the page does not change when the second one lands.

    Planned countries are named underneath rather than listed as options: an
    option that leads to an empty dashboard is worse than a sentence.
    """
    reg = registry()
    live = ctry.live_codes(reg)
    where = container or st
    where.selectbox("Country", live, key=COUNTRY_KEY,
                    format_func=lambda c: ctry.label(reg, c),
                    label_visibility=label_visibility,
                    help="Each country is scored from its own indicator set and centers.")
    return country()


def planned_note() -> str:
    """'Japan, Germany and the United Kingdom are being onboarded.' or ''."""
    names = [f"{p.get('article', '')} {p.get('label', p['code'])}".strip()
             for p in ctry.planned(registry())]
    if not names:
        return ""
    from src.ui import join_words
    verb = "is" if len(names) == 1 else "are"
    return f"{join_words(names)} {verb} being onboarded."


def settings_banner() -> None:
    """Say so, on every page, when the reader is not looking at the defaults."""
    n = overrides_active()
    if not n:
        return
    st.html(f'<div class="mr-custom"><b>Custom settings.</b> {n} '
            f'{"assumption" if n == 1 else "assumptions"} changed from the published '
            f'defaults, in this browser session only. Everything below is scored with '
            f'them.</div>')
    if st.button("Reset to defaults", type="tertiary", icon=":material/restart_alt:",
                 key=f"reset_{st.session_state.get('_mr_page', 'x')}"):
        reset_overrides()
        st.rerun()


def refresh_button() -> None:
    """Let a viewer skip the wait for the periodic snapshot check."""
    if not data_url():
        return
    if st.button("Check for new data", icon=":material/refresh:", type="tertiary"):
        snapshot_manifest.clear()
        st.rerun()


def source_sentence(vendors: set[str]) -> str:
    """Plain-language data source, as HTML, for footers and the methodology."""
    parts = []
    if "demo" in vendors:
        return ("Synthetic demo series generated by <code>ingest.py demo</code>, with "
                "invented dynamics, so the numbers mean nothing.")
    if "macrobond" in vendors:
        parts.append("Macrobond, carrying each series from its original publisher: BEA, "
                     "BLS, Census, the Federal Reserve Board, the Chicago and Cleveland Feds, "
                     "CBO, OMB and the University of Michigan")
    if "fred" in vendors:
        parts.append("FRED and ALFRED, Federal Reserve Bank of St. Louis")
    if not parts:
        return "No data loaded."
    return "; ".join(parts) + ". Each value is the latest vintage published on or before the chosen date."
