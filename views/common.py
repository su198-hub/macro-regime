"""Loaders shared by every page. Cached, so switching pages does not recompute."""

from __future__ import annotations

import datetime as dt
import os

import streamlit as st

from src.drivers import compute, load_config
from src.regimes import load_regimes, run
from src.store import Store

INDICATORS = "config/indicators.yml"
REGIMES = "config/regimes.yml"
REPO_URL = "https://github.com/su198-hub/macro-regime"


@st.cache_resource
def get_store():
    # Point MACRO_REGIME_DB outside OneDrive; sync locks the file mid-write.
    store = Store(os.environ.get("MACRO_REGIME_DB", "data/regime.duckdb"))
    # Hosted deploys start from an empty disk. Opt-in only, so synthetic data
    # never lands in a real store by accident.
    if os.environ.get("MACRO_REGIME_SEED_DEMO") == "1" and store.coverage().empty:
        from ingest import load_demo
        load_demo(store, load_config(INDICATORS))
    return store


@st.cache_data(ttl=900)
def get_results(vintage: dt.date):
    store = get_store()
    cfg = load_config(INDICATORS)
    reg = load_regimes(REGIMES)
    res = compute(store, cfg, vintage)
    if res["drivers"].empty:
        return None
    res.update(run(res["drivers"], reg), config=cfg, regimes=reg)
    return res


def is_demo(store) -> bool:
    return store.con.execute(
        "SELECT COUNT(*) FROM series_meta WHERE source = 'demo'").fetchone()[0] > 0
