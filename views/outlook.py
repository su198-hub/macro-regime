"""The three-year outlook (/outlook): v2's call, read for where growth and
inflation are heading.

It reads the same v2 drivers and scenario positions as the dashboard, over
the four regimes the answer key can grade, whichever framework the reader has
picked elsewhere.

  1. The call: the most likely regime for the next three years, and the odds.
  2. Where growth and inflation are heading: each regime's typical outcome on
     a growth x inflation map, and the probability-weighted landing point.
  3. What is pushing it: each driver's pull on that landing point.
  4. The drivers, as on the dashboard.
  5. A policy what-if: move monetary or fiscal policy, watch the call move.
  6. The track record, graded as the call would have been made at the time.
"""

from __future__ import annotations

import copy
import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import countries as ctry
from src import outlook as ol
from src import regimes as R
from src import truth
from src import ui
from src.drivers import compute, config_hash, driver_groups, group_scores, load_config
from views.common import data_version, get_store, registry

st.html('<h1 class="mr-title">Three-year outlook</h1>'
        '<p class="mr-sub">The most likely regime for the next three years, where growth and '
        'inflation are heading, and why. Read from the v2 drivers.</p>')

base = ctry.resolve(registry(), "US", framework="v2")
CFG = load_config(base["indicators"])
REG_ALL = R.load_regimes(base["regimes"])
# The call covers the four regimes the answer key can grade.
REG = copy.deepcopy(REG_ALL)
REG["regimes"] = {k: v for k, v in REG["regimes"].items() if k in truth.QUADRANT.values()}
NAMES = list(CFG["drivers"])
LABEL = {k: v["label"] for k, v in REG["regimes"].items()}
COLOR = {k: v["color"] for k, v in REG["regimes"].items()}
DLABEL = {k: v.get("label", k) for k, v in CFG["drivers"].items()}


# Cached on the data and on both configs: caches outlive a deploy, and a
# config-only change must not leave old results beside new text.
CONFIG_KEY = config_hash(CFG) + str(hash(repr(REG)))


@st.cache_data(ttl=900, show_spinner="Scoring the drivers…")
def model(version: str, config_key: str) -> dict | None:
    store = get_store()
    res = compute(store, CFG, dt.date.today())
    if res["drivers"].empty or not set(NAMES) <= set(res["drivers"].columns):
        return None
    scores = ol.monthly_drivers(res["drivers"], NAMES)
    wide = store.latest(list(truth.STORE_IDS.values()))
    if scores.empty or wide.empty or not set(truth.STORE_IDS.values()) <= set(wide.columns):
        return None
    fwd = truth.build(truth.from_store(wide), **truth.METHODS["next3"])
    train = ol.training_frame(scores, fwd)
    # Each regime's typical next three years: the average growth and
    # inflation gaps that followed the months in its run-up.
    outcomes = (fwd.loc[train.index, ["growth", "inflation"]].join(train["label"])
                .groupby("label").mean().reindex(list(REG["regimes"])))
    return {"drivers": res["drivers"], "indicators": res["indicators"], "scores": scores,
            "train": train, "fwd": fwd, "outcomes": outcomes}


def probabilities(x: pd.DataFrame) -> pd.DataFrame:
    return R.probabilities(x[NAMES], REG)


def landing(p: pd.Series, outcomes: pd.DataFrame) -> pd.Series:
    return (outcomes.T * p.reindex(outcomes.index)).T.sum()


M = model(data_version(), CONFIG_KEY)
if M is None:
    st.info("The outlook needs the driver scores and the history behind the regime labels, "
            "and one of them has not loaded. Refresh in a few minutes.")
    st.stop()
scores, outcomes = M["scores"], M["outcomes"]
latest = scores.index[-1]
now = scores.loc[[latest]]
p_now = probabilities(now).iloc[0]
year_ago = latest - 12
p_then = probabilities(scores.loc[[year_ago]]).iloc[0] if year_ago in scores.index else None
land_now = landing(p_now, outcomes)
land_then = landing(p_then, outcomes) if p_then is not None else None
ranked = p_now.sort_values(ascending=False)
lead, second = ranked.index[0], ranked.index[1]
# The call, as the dashboard makes it: a new leader takes over only after
# leading for the persistence rule's run of months.
_calls = R.call_regime(probabilities(scores), REG)
called = _calls.at[latest, "called"]
_lead_run = int((_calls["leading"][::-1] == lead).cummin().sum())
_need = int(REG["settings"]["persistence_months"])


def direction(g: float, i: float) -> str:
    gw = "above" if g >= 0 else "below"
    iw = "above" if i >= 0 else "below"
    return (f"growth about {abs(g):.1f} points a year {gw} potential and inflation about "
            f"{abs(i):.1f} points {iw} expectations")


# ---------- 1. the call ----------

when = latest.to_timestamp(how="end")
call_col, prob_col = st.columns([1.15, 1], gap="large")
if called == lead:
    _lede = (f'{LABEL[lead]} is the most likely regime for the next three years, at '
             f'{p_now[lead]:.0%}; {LABEL[second].lower()} is next at {p_now[second]:.0%}. ')
else:
    _lede = (f'{LABEL[lead]} now leads, at {p_now[lead]:.0%} against {p_now[called]:.0%} for '
             f'{LABEL[called].lower()}, but has led for {_lead_run} of the {_need} months needed to '
             f'change the call, so the call stays {LABEL[called].lower()} for now. ')
call_col.html(
    f'<p class="mr-eyebrow">Next three years, read in {when:%B %Y}</p>'
    f'<div class="mr-call"><span class="mr-call-swatch" style="background:{COLOR[called]}"></span>'
    f'{ui.esc(LABEL[called])}</div>'
    f'<p class="mr-lede">{ui.esc(_lede)}'
    f'Weighing all four by their odds, the drivers point to {direction(*land_now)}.</p>'
    + (f'<p class="mr-lede-muted">A year ago the same reading pointed to '
       f'{direction(*land_then)}.</p>' if land_then is not None else ""))
prob_col.html(ui.probability_panel(p_now, REG, called, f"{when:%b}"))

# ---------- 2. where growth and inflation are heading ----------

st.html(ui.section_head(
    "Where growth and inflation are heading", f"Next three years · read in {when:%b %Y}",
    "Each regime's dot is what followed its run-ups since 1968: the average yearly growth gap "
    "(growth less potential) and inflation gap (core PCE less expected inflation) over the next "
    "three years. The black diamond weighs the four by today's odds; the hollow one by the odds "
    "a year ago."))
pts = outcomes.reset_index().rename(columns={"label": "regime"})
pts["name"] = pts["regime"].map(LABEL)
pts["odds"] = pts["regime"].map(p_now).map(lambda v: f"{v:.0%}")
lim = float(np.ceil(max(outcomes.abs().max().max(), abs(land_now).max()) * 1.25 * 2) / 2)
quad = pd.DataFrame([
    {"x1": 0, "x2": lim, "y1": -lim, "y2": 0, "regime": "goldilocks"},
    {"x1": 0, "x2": lim, "y1": 0, "y2": lim, "regime": "high_growth_high_inflation"},
    {"x1": -lim, "x2": 0, "y1": 0, "y2": lim, "regime": "stagflation"},
    {"x1": -lim, "x2": 0, "y1": -lim, "y2": 0, "regime": "hard_landing"}])
scale = alt.Scale(domain=list(COLOR), range=list(COLOR.values()))
xs, ys = alt.Scale(domain=[-lim, lim]), alt.Scale(domain=[-lim, lim])
bg = alt.Chart(quad).mark_rect(opacity=0.08).encode(
    x=alt.X("x1:Q", scale=xs, title="Growth gap, points a year", axis=alt.Axis(tickCount=8, grid=False)),
    x2="x2", y=alt.Y("y1:Q", scale=ys, title="Inflation gap, points", axis=alt.Axis(tickCount=8, grid=False)),
    y2="y2", color=alt.Color("regime:N", scale=scale, legend=None))
rules = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#9aa3b2").encode(x="v:Q") + \
    alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#9aa3b2").encode(y="v:Q")
dots = alt.Chart(pts).mark_circle(size=260, opacity=0.95).encode(
    x=alt.X("growth:Q", scale=xs), y=alt.Y("inflation:Q", scale=ys),
    color=alt.Color("regime:N", scale=scale, legend=None),
    tooltip=[alt.Tooltip("name:N", title="Regime"), alt.Tooltip("growth:Q", format="+.2f", title="Growth gap"),
             alt.Tooltip("inflation:Q", format="+.2f", title="Inflation gap"), alt.Tooltip("odds:N", title="Odds today")])
names = alt.Chart(pts).mark_text(dy=-18, fontSize=12, fontWeight="bold").encode(
    x=alt.X("growth:Q", scale=xs), y=alt.Y("inflation:Q", scale=ys), text="name:N",
    color=alt.Color("regime:N", scale=scale, legend=None))
lp = pd.DataFrame([{"growth": land_now["growth"], "inflation": land_now["inflation"],
                    "what": f"Next 3 years, read in {when:%b %Y}"}]
                  + ([{"growth": land_then["growth"], "inflation": land_then["inflation"],
                       "what": "Read a year earlier"}] if land_then is not None else []))
land_marks = alt.Chart(lp).mark_point(shape="diamond", size=320, strokeWidth=2.5).encode(
    x=alt.X("growth:Q", scale=xs), y=alt.Y("inflation:Q", scale=ys),
    fill=alt.condition(alt.datum.what == "Read a year earlier", alt.value("white"), alt.value(ui.INK)),
    color=alt.value(ui.INK),
    tooltip=[alt.Tooltip("what:N", title=""), alt.Tooltip("growth:Q", format="+.2f", title="Growth gap"),
             alt.Tooltip("inflation:Q", format="+.2f", title="Inflation gap")])
arrow = (alt.Chart(lp).mark_line(color=ui.INK, strokeDash=[4, 3]).encode(
    x=alt.X("growth:Q", scale=xs), y=alt.Y("inflation:Q", scale=ys), order="what:N")
    if land_then is not None else alt.Chart(lp).mark_point(opacity=0))
land_label = alt.Chart(lp.iloc[[0]]).mark_text(align="left", dx=14, dy=-12, fontSize=12, fontWeight="bold",
                                                color=ui.INK).encode(
    x=alt.X("growth:Q", scale=xs), y=alt.Y("inflation:Q", scale=ys), text=alt.value("Next three years"))
st.altair_chart(ui.style((bg + rules + arrow + dots + names + land_marks + land_label).properties(height=420)),
                use_container_width=True)

# ---------- 3. what is pushing it ----------

st.html(ui.section_head(
    "What is pushing it", "",
    "How far each driver moves the landing point: the point with the driver as it reads "
    "today, less the point with that driver set to neutral. Positive pushes growth or "
    "inflation up."))
rows = []
for n in NAMES:
    alt_x = now.copy()
    alt_x[n] = 0.0
    moved = land_now - landing(probabilities(alt_x).iloc[0], outcomes)
    rows.append([DLABEL[n], ui.signed(float(now[n].iloc[0])),
                 ui.signed(float(moved["growth"])), ui.signed(float(moved["inflation"]))])
st.html(ui.table(["Driver", f"Reading, {when:%b %Y}", "Pushes growth", "Pushes inflation"],
                 rows, numeric={1, 2, 3}))

# ---------- 4. the drivers ----------

st.html(ui.section_head(
    "The drivers", f"Monthly · {when:%b %Y}",
    "Each row runs from one extreme to the other. The codes are the four regimes, in order, at "
    "where the drivers sat in their run-ups; the star is this month, the circle a year ago."))
drivers = M["drivers"]
latest_ts, then_ts = when.normalize() + pd.offsets.MonthEnd(0), None
idx = pd.DatetimeIndex(drivers.index)
latest_ts = idx[idx.to_period("M") == latest][0]
then_match = idx[idx.to_period("M") == year_ago]
then_ts = then_match[0] if len(then_match) else None
blocks = {}
for d in NAMES:
    groups = driver_groups(CFG, d)
    if len(groups) > 1:
        gs = group_scores(CFG, d, M["indicators"])
        if latest_ts in gs.index:
            blocks[d] = {groups[g].get("label") or g: gs.at[latest_ts, g] for g in gs.columns}
st.html(ui.signpost_html(drivers, CFG, REG, latest_ts, then_ts, None, blocks=blocks))

# ---------- 5. policy what-if ----------

st.html(ui.section_head(
    "What if policy moves", "",
    "Set monetary and fiscal policy where you think they are heading; the odds and the landing "
    "point update. Everything else stays as it reads today."))
c1, c2 = st.columns(2)
mon = c1.slider("Monetary policy (dovish − / hawkish +)", -1.0, 1.0,
                float(round(now["monetary"].iloc[0], 2)), 0.05, key="prop_mon")
fis = c2.slider("Fiscal policy (restrictive − / supportive +)", -1.0, 1.0,
                float(round(now["fiscal"].iloc[0], 2)), 0.05, key="prop_fis")
wx = now.copy()
# A slider shows today's reading rounded; left there, it means today's reading.
for col, val in (("monetary", mon), ("fiscal", fis)):
    if val != round(float(now[col].iloc[0]), 2):
        wx[col] = val
p_w = probabilities(wx).iloc[0]
land_w = landing(p_w, outcomes)
st.html(ui.table(
    ["", "As it reads today", "With these settings"],
    [[LABEL[r], f"{p_now[r]:.0%}", f"{p_w[r]:.0%}"] for r in REG["regimes"]]
    + [["Growth gap, landing", ui.signed(land_now["growth"]), ui.signed(land_w["growth"])],
       ["Inflation gap, landing", ui.signed(land_now["inflation"]), ui.signed(land_w["inflation"])]],
    numeric={1, 2}))

# ---------- 6. track record ----------

st.html(ui.section_head(
    "Track record", "Graded against what the next three years turned out to be",
    "Held out: each episode called from positions learned without it or any month sharing its "
    "outcome window. Real time: each month called only from outcomes already known then. "
    "Share of each regime's months called right, averaged over the four regimes."))


@st.cache_data(ttl=6 * 3600, show_spinner="Grading every month since 1976…")
def track(version: str, config_key: str) -> dict:
    from scripts.realign_positions import grade, held_out, priors, real_time
    train, fwd = M["train"], M["fwd"]
    window = train.index[train.index >= pd.Period("1976-01", "M")]
    labels = fwd["label"].reindex(window)
    prior = priors(REG, NAMES)
    out = {}
    for mode, fn in (("Held out", held_out), ("Real time", real_time)):
        P, F = fn(train, prior, REG, NAMES)
        g = grade(P, F, REG, train, window, labels)
        calls = R.call_regime(P.reindex(window).dropna(how="all"), REG)["called"].reindex(window)
        out[mode] = {"g": g, "calls": calls}
    from src.backtest import NBER  # noqa: F401  (kept for the recession shading below)
    nowcast = truth.build(truth.from_store(get_store().latest(list(truth.STORE_IDS.values()))),
                          **truth.METHODS["now"])["label"]
    out["Regime continues"] = {"g": ol.grade(ol.persistence(nowcast, window), labels, train.loc[window])}
    out["labels"] = labels
    return out


if st.toggle("Show the track record", key="prop_track"):
    T = track(data_version(), CONFIG_KEY)
    rows = []
    for name in ("Held out", "Real time", "Regime continues"):
        g = T[name]["g"]
        rows.append([name, f"{g['balanced']:.0%}",
                     *[f"{g['by_regime'].get(r, float('nan')):.0%}" for r in REG["regimes"]],
                     f"{g['hard_landing_caught']} of {g['hard_landing_runups']}"])
    st.html(ui.table(["", "Regimes right, average"] + [LABEL[r] for r in REG["regimes"]]
                     + ["Hard-landing run-ups caught"], rows, numeric=set(range(1, 7))))
    st.caption("Placebo, run 22 October 2026: with the driver readings slid out of line with what "
               "followed, none of 40 held-out and none of 20 real-time runs matched the scores "
               "then, 68% and 66% (medians 31% and 25%).")
    band = pd.DataFrame({"Real-time call": T["Real time"]["calls"],
                         "What the next 3 years were": T["labels"]})
    long = band.reset_index(names="month").melt("month", var_name="row", value_name="regime").dropna()
    long["month"] = long["month"].dt.to_timestamp()
    long["name"] = long["regime"].map(LABEL)
    chart = alt.Chart(long).mark_rect().encode(
        x=alt.X("month:T", title=None, timeUnit="yearmonth"),
        y=alt.Y("row:N", title=None, sort=["Real-time call", "What the next 3 years were"]),
        color=alt.Color("regime:N", scale=scale, legend=alt.Legend(title=None, orient="top",
                        labelExpr=" + ".join(f"(datum.label == '{k}' ? '{v}' : '')" for k, v in LABEL.items()))),
        tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("row:N", title=""),
                 alt.Tooltip("name:N", title="Regime")])
    st.altair_chart(ui.style(chart.properties(height=110)), use_container_width=True)

# ---------- how it works ----------

st.html(ui.section_head("How it works"))
st.html(
    '<div class="m-body"><ul>'
    '<li><b>Five drivers, read monthly:</b> growth, inflation, monetary policy, fiscal policy, '
    'reforms. Each is a weighted average of indicators chosen for their economic link and '
    'checked against history.</li>'
    '<li><b>Each regime has a position:</b> where the five drivers sat in the three years '
    'before it, learned from fifteen run-ups since 1968 (three or four per regime) and nudged '
    'toward a judgment of where the regime sits. Policy is part of the position: hard '
    'landings came after tight money, high inflation after loose money.</li>'
    '<li><b>The call</b> compares this month\'s drivers with each position; the closer, the '
    'likelier. Fiscal policy and reforms count at half weight. The call is held for three '
    'months before it changes.</li>'
    '<li><b>Where growth and inflation are heading</b> weighs what followed each regime\'s '
    'run-ups by today\'s odds.</li>'
    '<li><b>Before 1979</b> the inflation anchor is the Fed\'s FRB/US estimate of long-run '
    'expectations, not a survey; everything is on today\'s revised data.</li>'
    '</ul></div>')
