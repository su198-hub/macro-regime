"""Macro regime dashboard.

    streamlit run app.py

The vintage selector in the sidebar is the point of the whole thing. Move it
back a year and the dashboard shows what you would have seen then, not what
you know now.
"""

from __future__ import annotations

import datetime as dt
import os

import altair as alt
import pandas as pd
import streamlit as st

from src.drivers import compute, load_config
from src.regimes import contributions, load_regimes, run
from src.store import Store

st.set_page_config(page_title="Macro regime", layout="wide")

PALETTE = {
    "goldilocks": "#3f7d5a",
    "high_growth_high_inflation": "#b8792b",
    "stagflation": "#9c4a34",
    "hard_landing": "#3a5a80",
    "transitional": "#8a8880",
}


@st.cache_resource
def get_store():
    # Point MACRO_REGIME_DB outside OneDrive; sync locks the file mid-write.
    store = Store(os.environ.get("MACRO_REGIME_DB", "data/regime.duckdb"))
    # Hosted deploys start from an empty disk. Opt-in only, so synthetic data
    # never lands in a real store by accident.
    if os.environ.get("MACRO_REGIME_SEED_DEMO") == "1" and store.coverage().empty:
        from ingest import load_demo
        load_demo(store, load_config("config/indicators.yml"))
    return store


@st.cache_data(ttl=900)
def get_results(vintage: dt.date):
    store = get_store()
    cfg = load_config("config/indicators.yml")
    reg = load_regimes("config/regimes.yml")
    res = compute(store, cfg, vintage)
    if res["drivers"].empty:
        return None
    res.update(run(res["drivers"], reg), config=cfg, regimes=reg)
    return res


store = get_store()

st.sidebar.header("View")
vintage = st.sidebar.date_input(
    "Data as known on", value=dt.date.today(),
    help="Rewind to see the call you would have made at the time.",
)
results = get_results(vintage)

if results is None:
    st.title("No data yet")
    st.write("Load something first:")
    st.code("python ingest.py demo        # synthetic, runs immediately\n"
            "python ingest.py backfill    # real vintages, needs FRED_API_KEY")
    st.stop()

drivers = results["drivers"]
probs = results["probabilities"].dropna(how="all")
calls = results["calls"]
reg = results["regimes"]
driver_cols = [c for c in drivers.columns if not c.endswith("__coverage")]

if probs.empty:
    st.warning("Not enough overlapping coverage to score a regime at this "
               "vintage. Check the Coverage tab.")
    st.stop()

latest = probs.index[-1]
call = calls.loc[latest]
label = reg["regimes"].get(call["called"], {}).get("label", "Transitional")

st.title(label)
st.caption(f"As of {latest:%B %Y} · data known on {vintage} · "
           f"config {results.get('config_hash', 'n/a')}")

is_demo = store.con.execute(
    "SELECT COUNT(*) FROM series_meta WHERE source = 'demo'").fetchone()[0]
if is_demo:
    st.warning("Demo data. These series are synthetic, so the call and the "
               "numbers mean nothing yet.")

cols = st.columns(len(probs.columns))
for col, name in zip(cols, probs.columns):
    col.metric(reg["regimes"][name]["label"], f"{probs[name].iloc[-1]:.0%}")

if call["called"] != call["leading"]:
    lead = reg["regimes"][call["leading"]]["label"]
    st.info(f"{lead} is leading this month but has not held long enough to "
            f"change the call. It needs "
            f"{reg['settings']['persistence_months']} consecutive months.")

tab_now, tab_drivers, tab_ind, tab_judge, tab_cov = st.tabs(
    ["Regime history", "Drivers", "Indicators", "Judgement", "Coverage"]
)

with tab_now:
    recent = probs.tail(180).reset_index().melt(
        id_vars="observation_date", var_name="regime", value_name="probability")
    recent["label"] = recent["regime"].map(
        lambda r: reg["regimes"][r]["label"])
    st.altair_chart(
        alt.Chart(recent).mark_area().encode(
            x=alt.X("observation_date:T", title=None),
            y=alt.Y("probability:Q", stack="normalize", title=None,
                    axis=alt.Axis(format="%")),
            color=alt.Color("label:N", title=None,
                            scale=alt.Scale(
                                domain=[reg["regimes"][r]["label"]
                                        for r in probs.columns],
                                range=[PALETTE[r] for r in probs.columns])),
            tooltip=["observation_date:T", "label:N",
                     alt.Tooltip("probability:Q", format=".0%")],
        ).properties(height=300),
        width="stretch",
    )
    st.subheader("What is pulling the call")
    st.caption("Squared distance from each archetype, per driver. "
               "Lower means closer. The biggest number in the called row is "
               "the driver arguing against the call.")
    st.dataframe(contributions(drivers, reg, latest), width="stretch")

with tab_drivers:
    hist = drivers[driver_cols].tail(180).reset_index().melt(
        id_vars="observation_date", var_name="driver", value_name="score")
    hist["driver"] = hist["driver"].str.replace("_", " ").str.capitalize()
    st.altair_chart(
        alt.Chart(hist).mark_line().encode(
            x=alt.X("observation_date:T", title=None),
            y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("driver:N", title=None),
        ).properties(height=300),
        width="stretch",
    )
    st.caption("Positive means hot, tight or restrictive depending on the "
               "driver. See config/indicators.yml for the sign convention.")
    cov_cols = [f"{c}__coverage" for c in driver_cols
                if f"{c}__coverage" in drivers.columns]
    if cov_cols:
        thin = drivers[cov_cols].loc[latest]
        thin = thin[thin < 0.75]
        if not thin.empty:
            names = ", ".join(i.replace("__coverage", "").replace("_", " ")
                              for i in thin.index)
            st.warning(f"Running on partial inputs this month: {names}. "
                       "Weights were renormalised over what was available.")

with tab_ind:
    ind = results["indicators"]
    driver_pick = st.selectbox("Driver", driver_cols,
                               format_func=lambda d: d.replace("_", " ").capitalize())
    cols_for = [c for c in ind.columns if c.startswith(f"{driver_pick}::")]
    block = ind[cols_for].tail(120)
    block.columns = [c.split("::")[1].replace("_", " ") for c in block.columns]
    st.dataframe(
        block.tail(18).iloc[::-1].style.background_gradient(
            cmap="RdYlBu_r", vmin=-2, vmax=2).format("{:.2f}"),
        width="stretch",
    )
    st.caption("Direction already applied, so positive always means the "
               "indicator is pushing its driver up.")

with tab_judge:
    st.caption("Analyst observations sit in the same database as the series, "
               "so you can ask later whether judgement led or lagged the data.")
    with st.form("judgement", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        j_driver = c1.selectbox("Driver", driver_cols)
        j_dir = c2.selectbox("Direction", [1, 0, -1],
                             format_func={1: "Pushing up", 0: "Neutral",
                                          -1: "Pushing down"}.get)
        j_conf = c3.slider("Confidence", 0.0, 1.0, 0.5, 0.1)
        j_analyst = st.text_input("Your name")
        j_source = st.text_input("Source", placeholder="FOMC minutes, call notes")
        j_note = st.text_area("Observation")
        if st.form_submit_button("Save observation"):
            if not j_note.strip():
                st.error("Add an observation before saving.")
            else:
                store.add_judgement(dt.date.today(), j_driver, j_dir, j_conf,
                                    j_analyst, j_note, j_source)
                st.success("Saved.")
    log = store.con.execute(
        "SELECT as_of, driver, direction, confidence, analyst, source, note "
        "FROM judgement ORDER BY as_of DESC LIMIT 100"
    ).df()
    if log.empty:
        st.write("Nothing logged yet.")
    else:
        st.dataframe(log, width="stretch", hide_index=True)

with tab_cov:
    cov = store.coverage()
    st.dataframe(cov, width="stretch", hide_index=True)
    thin = cov[cov["vintages"] <= 1]["series_id"].tolist()
    if thin:
        st.warning("Single vintage only, so any backtest using these is "
                   "as-revised rather than point-in-time: " + ", ".join(thin))
