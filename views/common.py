"""Loaders shared by every page. Cached, so switching pages does not recompute.

Where the data comes from, in order:
  1. MACRO_REGIME_DATA_URL set: a snapshot published by `ingest.py publish`.
     This is how the hosted app gets real data it cannot fetch itself.
  2. Otherwise MACRO_REGIME_DB (default data/regime.duckdb), a local store,
     seeded with demo data only if MACRO_REGIME_SEED_DEMO=1 and it is empty.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import countries as ctry
from src import overrides as ov
from src.drivers import compute, expected_release, indicator_inputs, load_config, required_series
from src.regimes import fit, load_regimes, probabilities, run
from src.store import Store

REPO_URL = "https://github.com/su198-hub/macro-regime"
SNAPSHOT_CHECK_SECONDS = 600
COUNTRY_KEY = "mr_country"
FRAMEWORK_KEY = "mr_framework"


@st.cache_data(ttl=300, show_spinner=False)
def registry() -> dict:
    return ctry.load_registry()


def country() -> dict:
    """The country being viewed: code, label, framework and its config paths."""
    return ctry.resolve(registry(), st.session_state.get(COUNTRY_KEY),
                        framework=st.session_state.get(FRAMEWORK_KEY))


def settings_key() -> str:
    """Where this country and framework's settings are kept: the country code
    for the default framework, so existing sessions and links keep working, and
    code:framework for any other, because a weight on one framework's
    indicator means nothing in another's."""
    here = country()
    return here["code"] if here["framework"] == here["default_framework"] else f'{here["code"]}:{here["framework"]}'


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
    last = _last_good()
    try:
        last["manifest"] = read_manifest(url)
    except Exception:
        # Unreachable for a moment: keep serving the snapshot already loaded
        # rather than switching every page to local data until the next check.
        pass
    return last.get("manifest")


@st.cache_resource(show_spinner=False)
def _last_good() -> dict:
    """Shared across sessions: the last manifest read, and when a snapshot
    download last failed."""
    return {}


def data_version() -> str:
    """Changes whenever a new snapshot is published, so caches rebuild.

    "local" while the published snapshot cannot be loaded, so that results
    computed from the fallback are cached under the fallback's name and not
    under the snapshot's, where they would stick until the next publish.
    """
    m = snapshot_manifest()
    if not m:
        return "local"
    version = m["published_at"]
    return version if _snapshot_ready(version) else "local"


SNAPSHOT_RETRY_SECONDS = 120


@st.cache_resource(show_spinner="Loading the latest published data…", max_entries=1)
def _snapshot_store(version: str):
    """The store rebuilt from one publish. Raises if it cannot be loaded, so a
    failure is never cached."""
    from src.snapshot import load_snapshot
    safe = version.replace(":", "").replace("+", "_")
    path = Path(tempfile.gettempdir()) / "macro-regime" / f"snapshot-{safe}.duckdb"
    return load_snapshot(data_url(), path, manifest=snapshot_manifest())


def _snapshot_ready(version: str) -> bool:
    """Whether this publish's store is loaded, loading it if need be. After a
    failure it waits SNAPSHOT_RETRY_SECONDS before trying again, so a dead
    network does not stall every rerun."""
    failed = _last_good().get("failed")
    if failed and failed[0] == version and time.monotonic() - failed[1] < SNAPSHOT_RETRY_SECONDS:
        return False
    try:
        _snapshot_store(version)
    except Exception as exc:
        _last_good()["failed"] = (version, time.monotonic(), str(exc))
        return False
    _last_good().pop("failed", None)
    return True


def _store_for(version: str):
    return _snapshot_store(version) if version != "local" else _local_store()


@st.cache_resource(show_spinner=False)
def _local_store():
    # Point MACRO_REGIME_DB outside OneDrive; sync locks the file mid-write.
    store = Store(os.environ.get("MACRO_REGIME_DB", "data/regime.duckdb"))
    # Hosted deploys start from an empty disk. Opt-in only, so synthetic data
    # never lands in a real store by accident.
    if os.environ.get("MACRO_REGIME_SEED_DEMO") == "1" and store.coverage().empty:
        from ingest import load_demo, load_demo_truth
        # Demo data seeds every live country and every framework, so both
        # menus work without a vendor, and the answer key's inputs, so the
        # regime explorer does too.
        reg = ctry.load_registry()
        for code in ctry.live_codes(reg):
            for path in ctry.resolve(reg, code)["all_indicators"]:
                load_demo(store, load_config(path))
        load_demo_truth(store)
    return store


def get_store():
    version = data_version()
    failed = _last_good().get("failed")
    if version == "local" and failed and data_url():
        st.warning(f"Could not load the published data ({failed[2]}). Showing local data for "
                   f"now; it is tried again every two minutes.")
    return _store_for(version)


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
    return ov.clean(held.get(settings_key()), cfg, reg)


def set_overrides(new: dict | None) -> None:
    cfg, reg = base_config()
    held = dict(st.session_state.get(OVERRIDE_KEY) or {})
    held[settings_key()] = ov.clean(new, cfg, reg)
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
                 indicators: str = "", regimes: str = "", config_stamp: str = ""):
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
                        here["indicators"], here["regimes"],
                        config_stamp(here["indicators"], here["regimes"]))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _earliest_scoreable(version: str, indicators: str, regimes: str, config_stamp: str):
    """The first date on which data published by then scores a regime, or
    None if even today's cannot.

    Whether a date can be scored changes only when one of the model's series
    gains its first vintage, so the search runs over those dates. The store
    holds a series from its first recorded revision, which for many series is
    recent: before it, a rewind has no reading for that series at all.
    """
    store = _store_for(version)
    cfg, reg = load_config(indicators), load_regimes(regimes)
    rows = store.con.execute(
        "SELECT MIN(vintage_date) FROM observations WHERE series_id IN (SELECT unnest(?)) "
        "GROUP BY series_id", [required_series(cfg)]).fetchall()
    today = dt.date.today()
    dates = sorted({r[0] for r in rows if r[0] <= today} | {today})

    def scores(day: dt.date) -> bool:
        d = compute(store, cfg, day)["drivers"]
        return not d.empty and not probabilities(d, reg).dropna(how="all").empty

    if not scores(today):
        return None
    lo, hi = 0, len(dates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if scores(dates[mid]):
            hi = mid
        else:
            lo = mid + 1
    return dates[lo]


def earliest_scoreable() -> dt.date | None:
    """The first date the country and framework in view can be rewound to,
    under the published defaults."""
    here = country()
    return _earliest_scoreable(data_version(), here["indicators"], here["regimes"],
                               config_stamp(here["indicators"], here["regimes"]))


def config_stamp(*paths: str) -> str:
    """A fingerprint of the config files' contents, for cache keys.

    The cached results are keyed on what the files say, not just their names:
    after a deploy that renamed a signpost, results cached under the old file
    would otherwise be served beside the new config for up to the cache's
    lifetime, and the page would look up a signpost that is not there.
    """
    h = hashlib.sha1()
    for path in paths:
        try:
            h.update(Path(path).read_bytes())
        except OSError:
            h.update(path.encode())
    return h.hexdigest()[:12]


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
    """The country menu, and the framework menu under it when the country has
    more than one. One live country still gets a menu, so the shape of the page
    does not change when the second one lands.

    Planned countries are named underneath rather than listed as options: an
    option that leads to an empty dashboard is worse than a sentence.
    """
    reg = registry()
    live = ctry.live_codes(reg)
    where = container or st
    # The choice lives under a plain key and the menu under its own: Streamlit
    # drops a widget's state on any page that does not draw it, which reset
    # the framework to the default whenever a reader visited another page.
    _seed(COUNTRY_KEY, live)
    where.selectbox("Country", live, key=_widget(COUNTRY_KEY),
                    on_change=_keep, args=(COUNTRY_KEY,),
                    format_func=lambda c: ctry.label(reg, c),
                    label_visibility=label_visibility,
                    help="Each country is scored from its own indicator set and centers.")
    here = country()
    if len(here["frameworks"]) > 1:
        options = list(here["frameworks"])
        if st.session_state.get(FRAMEWORK_KEY) not in options:
            st.session_state[FRAMEWORK_KEY] = here["framework"]
        _seed(FRAMEWORK_KEY, options)
        where.selectbox("Framework", options, key=_widget(FRAMEWORK_KEY),
                        on_change=_keep, args=(FRAMEWORK_KEY,),
                        format_func=lambda f: here["frameworks"][f],
                        label_visibility=label_visibility,
                        help="Which signposts the call is scored from, and where each scenario sits on them.")
    return country()


def _widget(key: str) -> str:
    return f"_w_{key}"


def _seed(key: str, options: list) -> None:
    """Show the kept choice in the menu, which may be new on this page."""
    if st.session_state.get(key) in options:
        st.session_state[_widget(key)] = st.session_state[key]


def _keep(key: str) -> None:
    st.session_state[key] = st.session_state[_widget(key)]


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


def model_vendors(store, series_ids: list[str]) -> set[str]:
    """The vendors behind these series, for the source lines.

    Asked of the store's connection rather than through a new Store method on
    purpose: the store is a cached resource, so a hosted session that began
    before a deploy keeps an instance of the old class, and a method added
    since does not exist on it. That is how this broke once.
    """
    rows = store.con.execute(
        "SELECT DISTINCT source FROM series_meta WHERE series_id IN (SELECT unnest(?))",
        [list(series_ids)]).fetchall()
    return {r[0] for r in rows}


def source_sentence(vendors: set[str]) -> str:
    """Plain-language data source, as HTML, for footers and the methodology."""
    parts = []
    if "demo" in vendors:
        return ("Synthetic demo series generated by <code>ingest.py demo</code>, with "
                "invented dynamics, so the numbers mean nothing.")
    if "macrobond" in vendors:
        parts.append("Macrobond, carrying each series from its original publisher: BEA, "
                     "BLS, Census, the Treasury, the Federal Reserve Board, the New York, "
                     "Philadelphia and Cleveland Feds, CBO, S&amp;P with Robert Shiller, "
                     "the University of Michigan, the BIS, the World Bank, the World Justice "
                     "Project, the Fraser Institute, and Baker, Bloom and Davis for the policy "
                     "uncertainty indexes")
    if "fred" in vendors:
        parts.append("FRED and ALFRED, Federal Reserve Bank of St. Louis")
    if "bloomberg" in vendors:
        parts.append("Bloomberg, for the equity, credit and analyst-consensus series it "
                     "licenses, published here with the vendor's approval")
    if "quantgov" in vendors:
        parts.append("QuantGov RegData (Mercatus Center), for restrictions in the Code of "
                     "Federal Regulations")
    if not parts:
        return "No data loaded."
    return "; ".join(parts) + ". Each value is the latest vintage published on or before the chosen date."
