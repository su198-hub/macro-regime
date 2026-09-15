"""Dashboard page: the call, the signposts and what sits behind them.

The "data as known on" control is the point of the whole thing. Move it back a
year and the dashboard shows what you would have seen then, not what you know
now.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from src import ui
from src.drivers import DRIVER_SCALE, driver_breakdown
from src.regimes import contributions
from views.common import (get_results, get_store, is_demo as store_is_demo, published_note,
                          source_sentence)

# Diverging blue to red through a neutral grey, for signed indicator scores.
DIVERGING = LinearSegmentedColormap.from_list(
    "signed", ["#2a78d6", "#f0efec", "#e34948"])

store = get_store()

# ---------- masthead ----------

head_left, head_right = st.columns([3, 1], vertical_alignment="bottom")
head_left.html(
    '<p class="mr-eyebrow">United States</p>'
    '<h1 class="mr-title">Macro regime monitor</h1>'
    '<p class="mr-sub">Five drivers scored monthly from point-in-time data and '
    'mapped to four regimes.</p>')
vintage = head_right.date_input(
    "Data as known on", value=dt.date.today(), format="DD/MM/YYYY",
    help="Rewind to see the call you would have made at the time, using only "
         "data published by that date.")
st.html('<hr class="mr-rule">')

results = get_results(vintage)
if results is None:
    st.html('<h2 class="mr-h2">No data yet</h2>'
            '<p class="mr-caption">Load something first, then refresh.</p>')
    st.code("python ingest.py demo        # synthetic, runs immediately\n"
            "python ingest.py backfill    # real vintages, needs FRED_API_KEY")
    st.stop()

drivers = results["drivers"]
probs = results["probabilities"].dropna(how="all")
calls = results["calls"]
reg = results["regimes"]
cfg = results["config"]
settings = reg["settings"]
driver_cols = [c for c in drivers.columns if not c.endswith("__coverage")]
driver_label = {n: cfg["drivers"].get(n, {}).get("label", n) for n in driver_cols}
regime_label = {n: s["label"] for n, s in reg["regimes"].items()}

if probs.empty:
    st.warning("Not enough overlapping coverage to score a regime at this "
               "vintage. Check the coverage table below.")
    st.stop()

latest = probs.index[-1]
call = calls.loc[latest]
called = call["called"]
leading = call["leading"]
year_ago = drivers.index[drivers.index <= latest - pd.DateOffset(months=12)]
then = year_ago[-1] if len(year_ago) else None
is_demo = store_is_demo(store)

# ---------- headline ----------

called_rows = calls.loc[:latest, "called"]
held = ui.run_length(called_rows)
since = called_rows.index[-held] if held else latest
ranked = probs.loc[latest].sort_values(ascending=False)
runner_up = ranked.index[1] if len(ranked) > 1 else None

if called == "transitional":
    call_name, swatch = "Transitional", ui.TRANSITIONAL
    lede = (f"No regime clears the {settings['min_confidence']:.0%} confidence "
            f"floor. {regime_label[leading]} leads with {ranked.iloc[0]:.0%}.")
else:
    call_name, swatch = regime_label[called], reg["regimes"][called]["color"]
    lede = f"{regime_label[leading]} leads with {ranked.iloc[0]:.0%} probability"
    if runner_up is not None:
        gap = (ranked.iloc[0] - ranked.iloc[1]) * 100
        lede += f", {gap:.0f} points ahead of {regime_label[runner_up]}"
    lede += f". Called since {since:%B %Y}."

if called not in ("transitional", leading):
    streak = ui.run_length(calls.loc[:latest, "leading"])
    lede += (f" {regime_label[leading]} has led for {streak} of the "
             f"{settings['persistence_months']} consecutive months needed to "
             f"change the call.")

call_col, prob_col = st.columns([1.15, 1], gap="large")
call_col.html(
    f'<p class="mr-eyebrow">Regime call for {latest:%B %Y}</p>'
    f'<div class="mr-call"><span class="mr-call-swatch" '
    f'style="background:{swatch}"></span>{ui.esc(call_name)}</div>'
    f'<p class="mr-lede">{ui.esc(lede)}</p>'
    f'<p class="mr-lede-muted">Data as known on {vintage:%d %B %Y}. Latest month '
    f'with every driver scored: {latest:%B %Y}. {ui.esc(published_note())}</p>'
    + ('<div class="mr-demo">Demo data. These series are synthetic, so the call '
       'and the numbers mean nothing yet.</div>' if is_demo else ""))
prob_col.html(ui.probability_panel(probs.loc[latest], reg, called))

# ---------- signposts ----------

st.html('<hr class="mr-rule"><h2 class="mr-h2">Scenario drivers and signposts</h2>'
        '<p class="mr-caption">Each driver runs from one extreme to the other. '
        'Regime codes sit where that regime expects the driver to be; the star '
        'is where it is now. Hover any mark for the exact score. '
        '<a href="/methodology#m-signposts" target="_self">How to read this chart</a>.</p>')
st.html(ui.signpost_html(drivers, cfg, reg, latest, then))

# ---------- history ----------

st.html('<hr class="mr-rule"><h2 class="mr-h2">Regime probabilities over time</h2>'
        '<p class="mr-caption">The band on top shows the regime actually called '
        f'after the {settings["persistence_months"]}-month persistence rule. The '
        'bands below stack to 100%. Hover the chart for each month.</p>')
span = st.segmented_control("Range", ["5 years", "15 years", "All"],
                            default="15 years", label_visibility="collapsed")
months = {"5 years": 60, "15 years": 180}.get(span or "15 years")
window = probs if months is None else probs.tail(months)
st.altair_chart(ui.history_chart(window, calls, reg), width="stretch")
with st.expander("Show as a table"):
    table = window.iloc[::-1].rename(columns=regime_label)
    table.insert(0, "Called", calls["called"].reindex(window.index).iloc[::-1]
                 .map(lambda c: regime_label.get(c, "Transitional")))
    table.index = table.index.strftime("%b %Y")
    st.dataframe(table.style.format("{:.0%}", subset=list(regime_label.values())),
                 width="stretch", height=320)

# ---------- detail ----------

st.html('<hr class="mr-rule"><h2 class="mr-h2">Behind the call</h2>')
tab_pull, tab_drivers, tab_judge, tab_cov = st.tabs(
    ["What is pulling the call", "Driver history", "Judgement", "Coverage"])

ORDINAL = ["largest", "second largest", "third largest", "fourth largest", "smallest"]

with tab_pull:
    st.caption("Weighted squared distance from each regime's archetype, per "
               "driver, this month. Lower means closer. In the called regime's "
               "row, the largest number is the driver arguing against the call. "
               "Click any number to see the data behind it.")
    raw_contrib = contributions(drivers, reg, latest)
    contrib = raw_contrib.rename(index=regime_label, columns=driver_label)
    contrib["Total"] = contrib.sum(axis=1)
    event = st.dataframe(
        contrib.style.format("{:.2f}").background_gradient(
            cmap=LinearSegmentedColormap.from_list("seq", ["#ffffff", "#86b6ef"]),
            subset=list(driver_label.values()), axis=None, vmin=0),
        width="stretch", on_select="rerun", selection_mode="single-cell", key="pull_cell")

    # Resolve the clicked cell to (regime, driver). Nothing clicked, or the
    # Total column, falls back to the biggest gap in that regime's row, which
    # by default is the driver arguing hardest against the call.
    label_to_driver = {v: k for k, v in driver_label.items()}
    focus_regime = called if called in reg["regimes"] else leading
    focus_driver = None
    cells = event.selection.cells if event and event.selection else []
    if cells:
        row, col = cells[0]
        focus_regime = raw_contrib.index[int(row)]
        focus_driver = label_to_driver.get(col)
    if focus_driver is None:
        focus_driver = raw_contrib.loc[focus_regime].idxmax()

    spec = reg["regimes"][focus_regime]
    arche = float(spec["archetype"][focus_driver])
    score = float(drivers.at[latest, focus_driver])
    sal = float(reg["driver_salience"].get(focus_driver, 1.0))
    cell = float(raw_contrib.at[focus_regime, focus_driver])
    rank = int((raw_contrib.loc[focus_regime] > cell).sum())
    st.html(
        f'<div style="margin-top:1.2rem"><p class="mr-eyebrow">'
        f'{"Selected" if cells else "Largest gap for the called regime"}</p>'
        f'<h3 class="m-h3" style="margin-top:0">{ui.esc(spec["label"])} and '
        f'{ui.esc(driver_label[focus_driver].lower())}: {cell:.2f}</h3>'
        f'<p class="mr-lede">{ui.esc(driver_label[focus_driver])} scores '
        f'{ui.signed(score)} in {latest:%B %Y}. {ui.esc(spec["label"])} expects '
        f'{ui.signed(arche)}, a gap of {abs(score - arche):.2f}. Squared and weighted by '
        f'salience {sal:g}, that adds {cell:.2f} to the distance, the '
        f'{ORDINAL[min(rank, 4)]} of its five drivers.</p></div>')
    st.altair_chart(ui.gap_chart(drivers, focus_driver, arche, spec["label"],
                                 spec["color"]), width="stretch")

    parts = driver_breakdown(cfg, focus_driver, results["inputs"],
                             results["indicators"], latest)
    st.html(f'<p class="mr-probs-head" style="margin-top:0.6rem">What makes up the '
            f'{ui.esc(driver_label[focus_driver].lower())} score</p>'
            + ui.breakdown_table(parts, cfg["drivers"][focus_driver]["indicators"], score)
            + f'<p class="mr-caption">Score is the reading measured against its centre, '
              f'with direction applied, so positive always pushes the driver up. '
              f'Contribution is weight × score ÷ {DRIVER_SCALE:g}; the column sums to '
              f'the driver score. Red pushes up, blue pulls down.</p>')

    with st.expander(f"Indicator scores over the last 18 months: "
                     f"{driver_label[focus_driver].lower()}"):
        ind = results["indicators"]
        cols_for = [c for c in ind.columns if c.startswith(f"{focus_driver}::")]
        block = ind[cols_for].tail(18).iloc[::-1]
        ind_label = {i["id"]: i.get("label") or i["id"].replace("_", " ").capitalize()
                     for i in cfg["drivers"][focus_driver]["indicators"]}
        block.columns = [ind_label.get(c.split("::")[1], c) for c in block.columns]
        block.index = block.index.strftime("%b %Y")
        st.dataframe(
            block.style.background_gradient(cmap=DIVERGING, vmin=-2, vmax=2)
            .format("{:+.2f}", na_rep="–"),
            width="stretch")

with tab_drivers:
    st.caption("Scores run from −1 to +1. Positive means hot, tight or "
               "restrictive depending on the driver.")
    st.altair_chart(ui.drivers_chart(drivers.tail(180), cfg), width="content")
    cov_cols = [f"{c}__coverage" for c in driver_cols
                if f"{c}__coverage" in drivers.columns]
    if cov_cols:
        thin = drivers[cov_cols].loc[latest]
        thin = thin[thin < 0.75]
        if not thin.empty:
            names = ", ".join(driver_label[i.replace("__coverage", "")].lower()
                              for i in thin.index)
            st.warning(f"Running on partial inputs this month: {names}. "
                       "Weights were renormalised over what was available.")

with tab_judge:
    st.caption("Analyst observations sit in the same database as the series, "
               "so you can ask later whether judgement led or lagged the data.")
    with st.form("judgement", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        j_driver = c1.selectbox("Driver", driver_cols,
                                format_func=lambda d: driver_label[d])
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

# ---------- sources ----------

source = source_sentence(store.sources())
if published_note():
    source += " " + published_note()
st.html(
    f'<div class="mr-foot"><b>Sources:</b> {source}<br>'
    f'<b>Method:</b> each driver is a weighted mean of normalised indicators '
    f'(<code>config/indicators.yml</code>). Regime probabilities come from '
    f'distance to each archetype (<code>config/regimes.yml</code>), softmaxed at '
    f'temperature {settings["temperature"]}, with a '
    f'{settings["persistence_months"]}-month persistence rule before a call '
    f'changes. The archetypes have not yet been validated against a labelled '
    f'regime history.<br>'
    f'<b>Config:</b> {results.get("config_hash", "n/a")}</div>')
st.page_link("views/methodology.py", label="Read the full methodology",
             icon=":material/menu_book:")
