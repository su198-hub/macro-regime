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
from src.drivers import DRIVER_SCALE, driver_breakdown, load_config
from src.regimes import contributions, load_regimes
from views.common import (INDICATORS, REGIMES, get_results, get_store, is_demo as store_is_demo,
                          published_note, refresh_button, settings_banner, source_sentence)

# Diverging blue to red through a neutral gray, for signed indicator scores.
DIVERGING = LinearSegmentedColormap.from_list(
    "signed", ["#2a78d6", "#f0f0f0", "#e34948"])
VENDOR_NAME = {"macrobond": "Macrobond", "fred": "FRED", "demo": "synthetic demo data"}

store = get_store()

# ---------- masthead ----------

st.html('<hr class="mr-mast-rule">')
head_left, head_right = st.columns([3, 1], vertical_alignment="bottom")
_cfg_counts = (len(load_config(INDICATORS)["drivers"]), len(load_regimes(REGIMES)["regimes"]))
head_left.html(
    '<div class="mr-mast bare"><h1 class="mr-title">Macro Regime Monitor</h1>'
    f'<p class="mr-sub">U.S. economy · {ui.count_word(_cfg_counts[0])} drivers scored monthly '
    f'from point-in-time data and mapped to {ui.count_word(_cfg_counts[1])} regimes</p></div>')
vintage = head_right.date_input(
    "Data as known on", value=dt.date.today(), format="MM/DD/YYYY",
    help="Rewind to see the call you would have made at the time, using only "
         "data published by that date.")
st.html('<hr class="mr-rule">')
st.session_state["_mr_page"] = "dashboard"
settings_banner()

results = get_results(vintage)
if results is None:
    st.html('<h2 class="mr-h2">No data yet</h2>'
            '<p class="mr-caption">Load something first, then refresh.</p>')
    st.code("python ingest.py demo        # synthetic, runs immediately\n"
            "python ingest.py backfill    # real vintages")
    st.stop()

drivers = results["drivers"]
probs = results["probabilities"].dropna(how="all")
calls = results["calls"]
reg = results["regimes"]
cfg = results["config"]
settings = reg["settings"]
gate_on = str(settings.get("fit_gate", "none")) == "closer_than_neutral"
driver_cols = [c for c in drivers.columns if not c.endswith("__coverage")]
driver_label = {n: cfg["drivers"].get(n, {}).get("label", n) for n in driver_cols}
regime_label = {n: s["label"] for n, s in reg["regimes"].items()}
vendors = store.sources()
source_line = (f'<p class="mr-source">Source: '
               f'{", ".join(VENDOR_NAME.get(v, v) for v in sorted(vendors))}; '
               f'Macro Regime Monitor calculations.</p>')

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

# The newest month not yet confirmed, read from what has been released so far.
prov = results["provisional"][-1] if results.get("provisional") else None
prov_box = ""
if prov is not None:
    ind_label = {f"{d}::{i['id']}": i.get("label") or i["id"]
                 for d, spec in cfg["drivers"].items() for i in spec["indicators"]}
    anchor_release = {f"{d}::{i['id']}": i.get("release")
                      for d, spec in cfg["drivers"].items() for i in spec["indicators"]}
    sched = prov["schedule"]
    waiting = sched[sched["status"].isin(["carried", "pending"])]
    anchors = waiting[waiting["anchor"]]
    confirm_by = anchors["expected"].max() if len(anchors) else waiting["expected"].max()
    upcoming, seen = [], set()
    for r in waiting.dropna(subset=["expected"]).sort_values("expected").itertuples():
        release = anchor_release.get(r.key) or ind_label[r.key].split(",")[0]
        if release in seen:
            continue
        seen.add(release)
        upcoming.append({"release": release, "date": r.expected,
                         "driver": driver_label.get(r.driver, r.driver), "confirms": bool(r.anchor)})
    # Name the releases users watch for, such as "PCE report", not the indicators.
    confirm_with = list(dict.fromkeys(anchor_release.get(k) or ind_label[k].lower()
                                      for k in anchors["key"]))
    prov_box = ui.provisional_box(prov, reg, called, confirm_by, confirm_with, upcoming[:6])

# ---------- headline ----------

called_rows = calls.loc[:latest, "called"]
held = ui.run_length(called_rows)
since = called_rows.index[-held] if held else latest
ranked = probs.loc[latest].sort_values(ascending=False)
runner_up = ranked.index[1] if len(ranked) > 1 else None

call_name, swatch = ui.call_label(called, reg), ui.call_color(called, reg)
state = call.get("state", leading)
fits_now = bool(call.get("fits", True))
grade = call.get("fit") if gate_on else None
distance, threshold, limit = call.get("distance"), call.get("threshold"), call.get("limit")
fit_numbers = (f"distance {distance:.2f} against {threshold:.2f} for a neutral economy"
               if gate_on and pd.notna(distance) and pd.notna(threshold) else "")
if fit_numbers and grade != "clear" and pd.notna(limit):
    fit_numbers += f", limit {limit:.2f}"

if called == "transitional":
    lede = (f"No regime clears the {settings['min_confidence']:.0%} confidence "
            f"floor. {regime_label[leading]} leads with {ranked.iloc[0]:.0%}.")
elif called == "unclassified" and fits_now:
    lede = (f"{regime_label[leading]} fits this month ({ranked.iloc[0]:.0%}; {fit_numbers}), "
            f"but has not fit for long enough to be called. No clear regime since {since:%B %Y}.")
elif called == "unclassified":
    lede = (f"Nearest is {regime_label[leading]} ({ranked.iloc[0]:.0%}), but conditions are "
            f"too far from it to call"
            + (f" ({fit_numbers})" if fit_numbers else "") + f". No clear regime since {since:%B %Y}.")
else:
    lede = f"{regime_label[leading]} leads with {ranked.iloc[0]:.0%} probability"
    if runner_up is not None:
        gap = (ranked.iloc[0] - ranked.iloc[1]) * 100
        lede += f", {gap:.0f} points ahead of {regime_label[runner_up]}"
    lede += f". Called since {since:%B %Y}."
    # The fit is measured against this month's leader, so name it when the
    # call is still held on another regime by the persistence rule.
    subject = "it" if leading == called else regime_label[leading]
    if fit_numbers and grade == "clear":
        lede += (f" A clear fit: conditions sit closer to {subject} than a neutral economy "
                 f"would ({fit_numbers}).")
    elif fit_numbers and grade == "weak":
        lede += (f" A weak fit: conditions sit a little further from {subject} than a neutral "
                 f"economy would, but within the tolerance ({fit_numbers}).")
    elif fit_numbers:
        lede += f" This month conditions are too far from {subject} to fit ({fit_numbers})."

if called != "transitional" and state != called and pd.notna(state):
    streak = ui.run_length(calls.loc[:latest, "state"])
    lede += (f" {ui.call_label(state, reg)} for {streak} of the "
             f"{settings['persistence_months']} consecutive months needed to "
             f"change the call.")

call_col, prob_col = st.columns([1.15, 1], gap="large")
call_col.html(
    f'<p class="mr-eyebrow">Regime call, {latest:%B %Y} · confirmed</p>'
    f'<div class="mr-call"><span class="mr-call-swatch" '
    f'style="background:{swatch}"></span>{ui.esc(call_name)}</div>'
    f'<p class="mr-lede">{ui.esc(lede)}</p>'
    f'<p class="mr-lede-muted">Data as known on {ui.long_date(vintage)}. {latest:%B %Y} is the '
    f'latest month with its core data released. {ui.esc(published_note())}</p>'
    + ('<p class="mr-demo"><b>Demo data.</b> These series are synthetic, so the call '
       'and the numbers mean nothing yet.</p>' if is_demo else ""))
prob_col.html(ui.probability_panel(
    probs.loc[latest], reg, called, f"{latest:%b}",
    prov["probabilities"] if prov is not None else None,
    f"{prov['month']:%b}" if prov is not None else ""))
if prov_box:
    st.html(prov_box)

# ---------- signposts ----------

sp_meta = f"Monthly · {latest:%b %Y} confirmed" + (
    f" · {prov['month']:%b %Y} provisional" if prov is not None else "")
st.html(ui.section_head(
    "Scenario drivers and signposts", sp_meta,
    "Each driver runs from one extreme to the other. Regime codes sit where that regime expects "
    "the driver to be; the solid star is the confirmed month"
    + (", the outlined star the provisional one" if prov is not None else "")
    + ". Hover any mark for the exact score."))
st.html(ui.signpost_html(drivers, cfg, reg, latest, then, prov)
        + source_line.replace("</p>", ' · <a href="/methodology#m-signposts" target="_self">'
                                      'How to read this chart</a></p>'))

# ---------- history ----------

span = st.session_state.get("history_span") or "15 years"
months = {"5 years": 60, "15 years": 180}.get(span)
window = probs if months is None else probs.tail(months)
st.html(ui.section_head(
    "Regime probabilities over time",
    f"Monthly · {window.index[0]:%b %Y} to {latest:%b %Y}",
    "Probability of each regime, % of total, confirmed months. The band on top is the regime "
    f"called after the {settings['persistence_months']}-month persistence rule"
    + ("; a paler shade means a weak fit, and light gray no clear regime." if gate_on else ".")))
st.segmented_control("Range", ["5 years", "15 years", "All"], default="15 years",
                     label_visibility="collapsed", key="history_span")
st.altair_chart(ui.history_chart(window, calls, reg), width="stretch")
st.html(source_line)
export = probs.rename(columns=regime_label).assign(
    Called=calls["called"].reindex(probs.index).map(lambda c: ui.call_label(c, reg)),
    **({"Fit to nearest regime": ui.fit_grade(calls).reindex(probs.index).map(ui.FIT_WORDS)}
       if gate_on else {}))
export.index = export.index.strftime("%Y-%m")
export.index.name = "Month"
st.download_button("Download full history (CSV)", export.to_csv().encode(),
                   file_name=f"regime_probabilities_{latest:%Y%m}.csv", mime="text/csv",
                   type="tertiary", icon=":material/download:")
with st.expander("Show as a table"):
    table = window.iloc[::-1].rename(columns=regime_label)
    table.insert(0, "Called", calls["called"].reindex(window.index).iloc[::-1]
                 .map(lambda c: ui.call_label(c, reg)))
    table.index = table.index.strftime("%b %Y")
    st.dataframe(table.style.format("{:.0%}", subset=list(regime_label.values())),
                 width="stretch", height=320)

# ---------- detail ----------

# Any confirmed month can be examined, not just the latest: the same breakdown
# read at a past month is how you check whether a call made sense at the time.
years = sorted({d.year for d in probs.index}, reverse=True)
st.html(ui.section_head("Behind the call", "Pick any confirmed month",
                        "What was pulling the call, in the month you choose."))
pick_year, pick_month, _spacer = st.columns([1, 1, 3], gap="medium")
focus_year = pick_year.selectbox("Year", years, key="focus_year")
in_year = [d for d in probs.index if d.year == focus_year]
# No key: changing the year rebuilds the options and falls back to that year's
# last month, which is what a reader flipping through years expects.
focus = pick_month.selectbox("Month", in_year, index=len(in_year) - 1,
                             format_func=lambda d: f"{d:%B}")
focus_call = calls.loc[focus]
focus_grade = focus_call.get("fit") if gate_on else None
st.caption(
    f"{focus:%B %Y}: called {ui.call_label(focus_call['called'], reg).lower()}, "
    f"with {regime_label[focus_call['leading']].lower()} leading at "
    f"{probs.at[focus, focus_call['leading']]:.0%}"
    + (f", {ui.FIT_WORDS[focus_grade].lower()} fit to it" if isinstance(focus_grade, str) else "")
    + ("." if focus == latest else
       f". This is {ui.count_word(len(probs.loc[focus:latest]) - 1)} months before the "
       f"latest confirmed month."))

tab_pull, tab_status, tab_drivers, tab_judge, tab_cov = st.tabs(
    ["What is pulling the call", "Data status", "Driver history", "Judgment", "Coverage"])

with tab_status:
    if prov is None:
        st.caption(f"Every driver's core data for {latest:%B %Y} is in, and too little has "
                   f"been released for the following month to read it provisionally.")
    else:
        st.caption(
            f"Where each input stands for {prov['month']:%B %Y}, the provisional month. Released "
            f"data is used as published. Anything not released yet carries its latest value, for "
            f"up to {cfg['meta'].get('provisional', {}).get('fill_months', 2)} months. Expected dates "
            f"come from each series' own recent release timing, so treat them as estimates.")
        st.html(ui.status_table(prov["schedule"], cfg, prov["month"]))

with tab_pull:
    st.caption(f"Weighted squared distance from each regime's archetype, per driver, in "
               f"{focus:%B %Y}. Lower means closer. In the called regime's row, the largest "
               f"number is the driver arguing against the call. Click any number to see the "
               f"data behind it.")
    raw_contrib = contributions(drivers, reg, focus)
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
    focus_regime = (focus_call["called"] if focus_call["called"] in reg["regimes"]
                    else focus_call["leading"])
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
    score = float(drivers.at[focus, focus_driver])
    sal = float(reg["driver_salience"].get(focus_driver, 1.0))
    cell = float(raw_contrib.at[focus_regime, focus_driver])
    rank = int((raw_contrib.loc[focus_regime] > cell).sum())
    st.html(
        f'<div style="margin-top:1.2rem"><p class="mr-eyebrow">'
        f'{"Selected" if cells else "Largest gap for the called regime"}</p>'
        f'<h3 class="m-h3" style="margin-top:0">{ui.esc(spec["label"])} and '
        f'{ui.esc(driver_label[focus_driver].lower())}: {cell:.2f}</h3>'
        f'<p class="mr-lede">{ui.esc(driver_label[focus_driver])} scores '
        f'{ui.signed(score)} in {focus:%B %Y}. {ui.esc(spec["label"])} expects '
        f'{ui.signed(arche)}, a gap of {abs(score - arche):.2f}. Squared and weighted by '
        f'salience {sal:g}, that adds {cell:.2f} to the distance, the '
        f'{ui.ordinal_size(rank, len(driver_cols))} of its {ui.count_word(len(driver_cols))} '
        f'drivers.</p></div>')
    st.altair_chart(ui.gap_chart(drivers, focus_driver, arche, spec["label"],
                                 spec["color"]), width="stretch")

    parts = driver_breakdown(cfg, focus_driver, results["inputs"],
                             results["indicators"], focus)
    st.html(f'<p class="mr-probs-head" style="margin-top:0.6rem">What makes up the '
            f'{ui.esc(driver_label[focus_driver].lower())} score</p>'
            + ui.breakdown_table(parts, cfg["drivers"][focus_driver]["indicators"], score)
            + f'<p class="mr-caption">Score is the reading measured against its center, '
              f'with direction applied, so positive always pushes the driver up. '
              f'Contribution is weight × score ÷ {DRIVER_SCALE:g}; the column sums to '
              f'the driver score. Red pushes up, blue pulls down.</p>')

    with st.expander(f"Indicator scores over the 18 months to {focus:%B %Y}: "
                     f"{driver_label[focus_driver].lower()}"):
        ind = results["indicators"]
        cols_for = [c for c in ind.columns if c.startswith(f"{focus_driver}::")]
        block = ind[cols_for].loc[:focus].tail(18).iloc[::-1]
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
                       "Weights were renormalized over what was available.")

with tab_judge:
    st.caption("Analyst observations sit in the same database as the series, "
               "so you can ask later whether judgment led or lagged the data.")
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

source = source_sentence(vendors)
if published_note():
    source += " " + published_note()
st.html(
    f'<div class="mr-foot"><b>Sources:</b> {source}<br>'
    f'<b>Method:</b> each driver is a weighted mean of normalized indicators '
    f'(<code>config/indicators.yml</code>). Regime probabilities come from '
    f'distance to each archetype (<code>config/regimes.yml</code>), softmaxed at '
    f'temperature {settings["temperature"]}, with a '
    f'{settings["persistence_months"]}-month persistence rule before a call '
    f'changes. The archetypes have not yet been validated against a labeled '
    f'regime history.<br>'
    f'<b>Config:</b> {results.get("config_hash", "n/a")}</div>')
st.page_link("views/methodology.py", label="Read the full methodology",
             icon=":material/menu_book:")
st.page_link("views/control.py", label="Change these assumptions in the control room",
             icon=":material/tune:")
refresh_button()
