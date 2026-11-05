"""The regime explorer (/outlook, its address from when it was the three-year
outlook): what each regime looked like, any month since 1968, and v2's call
for where growth and inflation are heading.

It reads the same v2 drivers and scenario positions as the dashboard, over
the four regimes the answer key can grade, whichever framework the reader has
picked elsewhere.

  0. Regime shapes, first: where the drivers sat going into each regime and
     during it, one radar per regime beside today's, and what changed.
  0b. Go back in time: each call, held out, drawn over the
     months it was about, above what they turned out to be, and what led into
     each turning point. A
     radar shows what the model saw when it called a turn, against the
     typical run-ups to what came and to its other pick, and the text says
     which drivers and indicators decided it.
  0c. How well the call has done: held out against simple benchmarks, by
     regime, where the misses go, whether the odds hold, what each driver adds.

  Folded below them:
  1. The call: the most likely regime for the next three years, and the odds.
  2. Where growth and inflation are heading: each regime's typical outcome on
     a growth x inflation map, and the probability-weighted landing point.
  3. What is pushing it: each driver's pull on that landing point.
  4. The drivers, as on the dashboard.
  5. A policy what-if: move monetary stance or fiscal policy, watch the call move.
"""

from __future__ import annotations

import copy
import datetime as dt
import pathlib
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src import countries as ctry
from src import outlook as ol
from src import regimes as R
from src import truth
from src import ui
from src.drivers import DRIVER_SCALE, compute, config_hash, driver_groups, group_scores, load_config
from views.common import data_version, get_store, registry

st.html('<h1 class="mr-title">Regime explorer</h1>'
        '<p class="mr-sub">What each regime looked like going in and during, any month since 1968, '
        'and the call for the next three years. Read from the v2 drivers.</p>')

base = ctry.resolve(registry(), "US", framework="v2")
CFG = load_config(base["indicators"])
REG_ALL = R.load_regimes(base["regimes"])
# The call covers the four regimes the answer key can grade.
REG = copy.deepcopy(REG_ALL)
REG["regimes"] = {k: v for k, v in REG["regimes"].items() if k in truth.QUADRANT.values()}
NAMES = list(CFG["drivers"])
LABEL = {k: v["label"] for k, v in REG["regimes"].items()}
# Headings use the short code where the full name runs long (HGHI).
HEAD = {k: v["short"] if len(v["label"]) > 15 else v["label"] for k, v in REG["regimes"].items()}
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
    # During a regime: months whose growth and inflation over the 18 months
    # around them (15 before, 3 after) fit it, the answer key's reading of
    # the present.
    nowcast = truth.build(truth.from_store(wide), **truth.METHODS["now"])
    during = ol.training_frame(scores, nowcast)
    return {"drivers": res["drivers"], "indicators": res["indicators"], "scores": scores,
            "train": train, "fwd": fwd, "outcomes": outcomes, "during": during,
            "core_in": ol.core_positions(train, fwd, NAMES),
            "core_during": ol.core_positions(during, nowcast, NAMES),
            "inputs": res["inputs"], "nowcast": nowcast}


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


when = latest.to_timestamp(how="end")

# ---------- 0. regime shapes ----------

st.html(ui.section_head(
    "Regime shapes", f"Since 1968 · {when:%b %Y}",
    f"How the {ui.NUMBER_WORDS[len(NAMES)]} drivers looked going into each regime and during it, "
    "and what changes between the two."))
months = st.segmented_control("Months", ["All", "Strongest quarter"], default="All",
                              key="shape_months") or "All"
strong = months == "Strongest quarter"
shape = {
    "going": M["core_in"] if strong else ol.positions(M["train"], NAMES),
    "during": M["core_during"] if strong else ol.positions(M["during"], NAMES),
}
NAME = {"going": "Going in", "during": "During"}
KEY_NAME = {"going": "Going in: the three years before", "during": "During"}
# Going in is filled, during dashed.
LOOK = {"going": ({"fill": 0.18}, "solid"), "during": ({"fill": 0, "dash": True}, "dashed")}
# Each spoke names its driver and what its outer end means, kept short so
# six fit round a small chart.
POLE = {"inflation": "Above anchor"}
SHORT = {d: (DLABEL[d].split(" and ")[0],
             POLE.get(d, CFG["drivers"][d]["signpost"]["high"]["label"].split(" / ")[0]))
         for d in NAMES}
today_panel = ui.radar_svg(NAMES, SHORT, [{"name": f"{when:%b %Y}", "values": now.iloc[0].to_dict(),
                                            "color": ui.INK, "fill": 0.10}],
                          title=f"Today ({when:%b %Y})", ring_labels=True)
panels = []
for r in REG["regimes"]:
    series = [{"name": NAME[kind], "values": shape[kind].loc[r].to_dict(), "color": COLOR[r], **LOOK[kind][0]}
              for kind in NAME if r in shape[kind].index]
    panels.append(ui.radar_svg(NAMES, SHORT, series, title=HEAD[r], title_color=COLOR[r]))
st.html(ui.radar_grid(
    panels, [(KEY_NAME[k], ui.INK_2, LOOK[k][1]) for k in NAME], lead=today_panel,
    key=[("Scale", "The centre is −1, the middle ring neutral and the outer ring +1, toward the "
                   "end named under each driver. Hover a chart for its scores."),
         ("Going in and during", "Going in is the three years before each regime, its run-ups, "
                                 "which the call learns from. During is the months whose growth "
                                 "and inflation over the 18 months around them (15 before, 3 "
                                 "after) fit the regime, so it trails events by about six months."),
         ("Drivers look ahead", "So during a regime they already read the policy response and the "
                                "next phase; going in is the shape that warns of it."),
         ("Direction over size", "An average of many months sits closer to neutral than any one "
                                 "month. Strongest quarter keeps only each regime’s deepest months.")]))
# How each regime changes from going in to during: a card with what happened
# and all the drivers, biggest change first.
_diff = (shape["during"] - shape["going"]).reindex(list(REG["regimes"]))
cards = []
for r in REG["regimes"]:
    if r not in _diff.index or _diff.loc[r].isna().all():
        continue
    notes = REG["regimes"][r].get("notes") or {}
    top = _diff.loc[r].abs().sort_values(ascending=False).index
    cards.append({"title": HEAD[r], "color": COLOR[r], "note": notes.get("during", ""),
                  "points": notes.get("shifts", []), "ring": "dashed " + COLOR[r],
                  "rows": [(SHORT[d][0], float(shape["going"].at[r, d]),
                            float(shape["during"].at[r, d])) for d in top]})
st.html(ui.gap_cards(cards, NAME["going"], NAME["during"]))
_runs = ol.spells(M["train"]["label"])["label"].value_counts()
_spells = ol.spells(M["during"]["label"])["label"].value_counts()
_count = {"going": lambda r: f"{int(_runs.get(r, 0))} run-ups",
          "during": lambda r: f"{int(_spells.get(r, 0))} spells"}
rows = []
for r in REG["regimes"]:
    rows.append(ui.Raw(f'<b style="color:{COLOR[r]}">{ui.esc(HEAD[r])}</b>'))
    for kind in NAME:
        if r in shape[kind].index:
            rows.append([NAME[kind]] + [ui.signed(shape[kind].at[r, d]) for d in NAMES]
                        + [_count[kind](r)])
with st.expander("All the numbers"):
    st.html(ui.table(["Strongest quarter of months" if strong else "All months"]
                     + [SHORT[d][0] for d in NAMES] + ["Episodes"],
                     rows, numeric=set(range(1, len(NAMES) + 2))))

# ---------- 0b. go back in time ----------

st.html(ui.section_head("Go back in time", f"1968 to {when:%b %Y}"))
hx_box = st.container()
_HX_CFG = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config" / "history_events.yml",
                              encoding="utf-8"))
EVENTS, MISSES = _HX_CFG["events"], _HX_CFG.get("misses", [])
HX_MONTHS = list(M["scores"].index)
HX_LABEL = {m_: m_.strftime("%b %Y") for m_ in HX_MONTHS}
HX_BY_LABEL = {v: k for k, v in HX_LABEL.items()}
EV_BY_MONTH = {pd.Period(e["month"], "M"): e for e in EVENTS if pd.Period(e["month"], "M") in HX_LABEL}
# In running text a regime is its code if it has one short enough to be a
# heading (HGHI), otherwise its name in lower case.
INTEXT = {k: HEAD[k] if HEAD[k].isupper() else LABEL[k].lower() for k in LABEL}


# Two ways to make the calls the section grades. Held out leans on later
# history, so it shows what the drivers say with every other run-up known;
# real time is what the model could have said then, which before 2000 meant
# no hard landing to learn from.
HX_MODES = {
    "Held out": "each month called from positions learned from every other episode, leaving out its "
                "own and any month sharing its three years. It uses later history: what the drivers "
                "say with the other run-ups known, not what the model could have said then.",
    "Real time": "each month called from positions learned only from outcomes known then, from 1976. "
                 "The model had no hard landing to learn from until 2002."}
# Fixed at held out. The real-time paths stay: to offer the choice again, make
# this hx_box.radio("How the calls were made", list(HX_MODES), horizontal=True).
hx_mode = "Held out"
HELD = hx_mode == "Held out"
hx_box.html(f'<p class="mr-caption"><b>{hx_mode}:</b> {HX_MODES[hx_mode]}</p>')


@st.cache_data(ttl=6 * 3600, show_spinner="Calling every month…")
def predicted(version: str, config_key: str, mode: str) -> dict:
    """The call each month. Held out: each episode from positions learned
    without it or any month sharing its outcomes, and months past the last
    known outcome from positions learned from all of them. Real time: from
    positions learned only from what followed months already known then, from
    1976 (enough run-ups to learn from) to today."""
    from scripts.realign_positions import held_out, priors, real_time, with_positions
    prior = priors(REG, NAMES)
    if mode == "Held out":
        P, _ = held_out(M["train"], prior, REG, NAMES)
        later = M["scores"].loc[M["scores"].index > M["train"].index[-1], NAMES]
        if len(later):
            today = with_positions(REG, ol.realigned_positions(M["train"], prior, NAMES))
            P = pd.concat([P, R.probabilities(later, today)])
    else:
        P, _ = real_time(M["train"], prior, REG, NAMES, scores=M["scores"])
    return {"probs": P, "called": R.call_regime(P, REG)["called"]}


PRED = predicted(data_version(), CONFIG_KEY, hx_mode)
going_lab, during_lab = M["train"]["label"], M["during"]["label"]
# Misses written up in history_events.yml, as (first month, last month, entry),
# kept only while the model still makes the call in most of their months.
MISS_SPANS = []
for _e in MISSES:
    _a, _b = pd.Period(_e["from"], "M"), pd.Period(_e["to"], "M")
    _c = PRED["called"].loc[_a:_b]
    if len(_c) and (_c == _e["called"]).mean() >= 0.5:
        MISS_SPANS.append((_a, _b, _e))

# A call covers the next three years on average, so its middle is about a
# year and a half out; living through covers the 18 months to three months
# ahead, so its middle is about six months back. Drawn two years after it was
# made, a call sits over the months it was about: shifted two years, the
# answer key agrees with living through in 86% of months, against 71% at three
# years (checked 29 Oct 2026, 1968 to 2023).
SHIFT = 24
EARLY = 6  # months a call may run ahead of the run-up before the text calls it early
CALL_SPAN = 36  # a call covers three years, so one dropped before a turn was about it only if made within this
ROW_TURN, ROW_PRED, ROW_DUR = "Turning points", "Called 2 years earlier", "Living through"
ROWS = [ROW_TURN, ROW_PRED, ROW_DUR]
TURNS = ol.spells(during_lab).iloc[1:].reset_index(drop=True)


def _turn_of(month: pd.Period):
    """The turning point an event led into: the first from six months before
    it to two years after it. Otherwise the event sits inside a spell."""
    later = TURNS[TURNS["start"] >= month - 6]
    if len(later) and (later["start"].iloc[0] - month).n <= 24:
        return later["start"].iloc[0]
    return None


LEADS_INTO: dict = {}
for _m, _e in EV_BY_MONTH.items():
    _t = _turn_of(_m)
    if _t is not None:
        LEADS_INTO.setdefault(_t, []).append((_m, _e))
# Today first, then each turning point.
NOW_M = HX_MONTHS[-1]
NOW_CALL = PRED["called"].get(NOW_M)
TURN_OPTS = {f'Today · {HX_LABEL[NOW_M]}'
             + (f' · calling {HEAD[NOW_CALL]}' if NOW_CALL in HEAD else ""): HX_LABEL[NOW_M]}
# False alarms slot in among the turning points by month.
_jumps = [(t_, f'{HX_LABEL[t_]} · into {HEAD[r_]}'
           + (f' · {LEADS_INTO[t_][-1][1]["title"]}' if t_ in LEADS_INTO else ""))
          for t_, r_ in zip(TURNS["start"], TURNS["label"]) if t_ in HX_LABEL]
_jumps += [(a_, f'{HX_LABEL[a_]} · {e["title"]}') for a_, _, e in MISS_SPANS if a_ in HX_LABEL]
TURN_OPTS.update({lab: HX_LABEL[m_] for m_, lab in sorted(_jumps, key=lambda j: j[0])})
if st.session_state.get("hx_month") not in HX_BY_LABEL:
    st.session_state["hx_month"] = HX_LABEL.get(pd.Period("2008-03", "M"), HX_LABEL[HX_MONTHS[-1]])


def _hx_jump() -> None:
    pick = st.session_state.get("hx_turn")
    if pick in TURN_OPTS:
        st.session_state["hx_month"] = TURN_OPTS[pick]


c_ev, c_mo = hx_box.columns([1, 1.6])
c_ev.selectbox("Jump to today, a turning point or a false alarm", ["Choose…"] + list(TURN_OPTS), key="hx_turn",
               on_change=_hx_jump)
c_mo.select_slider("Or pick a month", options=[HX_LABEL[m_] for m_ in HX_MONTHS], key="hx_month")
hx = HX_BY_LABEL[st.session_state["hx_month"]]


def _spell_of(labels: pd.Series, month: pd.Period):
    for _, sp in ol.spells(labels).iterrows():
        if sp["start"] <= month <= sp["end"]:
            return sp
    return None


def _bands(labels: pd.Series, row: str, shift: int = 0) -> pd.DataFrame:
    sp = ol.spells(labels)

    def span(s_, e_):
        months = f"{s_.strftime('%b %Y')} to {e_.strftime('%b %Y')}"
        return f"Called {months}, for the 3 years after" if shift else months

    return pd.DataFrame({"start": [(s_ + shift).to_timestamp() for s_ in sp["start"]],
                         "end": [(e_ + shift + 1).to_timestamp() for e_ in sp["end"]],
                         "regime": sp["label"].to_numpy(), "name": sp["label"].map(LABEL).to_numpy(),
                         "row": row, "span": [span(s_, e_) for s_, e_ in zip(sp["start"], sp["end"])]})


# The timeline: each call drawn over the months it was about, above what those
# months turned out to be, so the two rows read against each other. Calls made
# in the last two years run past today. Markers sit at each turning point.
_pred = PRED["called"][PRED["called"].isin(list(COLOR))]
bands = pd.concat([_bands(_pred, ROW_PRED, SHIFT), _bands(during_lab, ROW_DUR)], ignore_index=True)
_last = max(HX_MONTHS[-1], _pred.index[-1] + SHIFT) if len(_pred) else HX_MONTHS[-1]
# A few months' room past the last band, so today's second stamp is not cut off.
xs_t = alt.Scale(domain=[HX_MONTHS[0].to_timestamp(), (_last + 4).to_timestamp()])
y_rows = alt.Y("row:N", sort=ROWS, title=None, scale=alt.Scale(domain=ROWS),
               axis=alt.Axis(labelLimit=220))
layers = [alt.Chart(bands).mark_rect().encode(
    x=alt.X("start:T", scale=xs_t, title=None), x2="end:T", y=y_rows,
    color=alt.Color("regime:N", scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())),
                    legend=None),
    tooltip=[alt.Tooltip("row:N", title=""), alt.Tooltip("name:N", title="Regime"),
             alt.Tooltip("span:N", title="Months")])]
sp_dur = _spell_of(during_lab, hx)
if sp_dur is not None:
    layers.append(alt.Chart(pd.DataFrame([{"start": sp_dur["start"].to_timestamp(),
                                           "end": (sp_dur["end"] + 1).to_timestamp(), "row": ROW_DUR}]))
                  .mark_rect(fill=None, stroke=ui.INK, strokeWidth=2).encode(
                      x=alt.X("start:T", scale=xs_t), x2="end:T", y=y_rows))
# Today's stamp: a diamond on the turning points row, and another where
# today's call is drawn, two years on, on the called row.
_now_odds = PRED["probs"].loc[NOW_M] if NOW_M in PRED["probs"].index else None
_now_tip = (f"Called {LABEL[NOW_CALL].lower()} for the next three years ({_now_odds[NOW_CALL]:.0%})"
            if NOW_CALL in LABEL and _now_odds is not None else "–")
_today = pd.DataFrame({"m": [(NOW_M + 1).to_timestamp(), (NOW_M + SHIFT).to_timestamp()],
                       "t": ["Today", ""], "row": [ROW_TURN, ROW_PRED],
                       "when": [f"Today, {HX_LABEL[NOW_M]}", f"Today's call, drawn two years on"],
                       "what": [_now_tip, _now_tip]})
layers.append(alt.Chart(_today.iloc[:1]).mark_rule(color=ui.MUTED, strokeDash=[3, 3]).encode(
    x=alt.X("m:T", scale=xs_t)))
layers.append(alt.Chart(_today).mark_point(shape="diamond", size=110, filled=True, color=ui.INK,
                                           stroke="#fff", strokeWidth=1.5, opacity=1).encode(
    x=alt.X("m:T", scale=xs_t), y=y_rows,
    tooltip=[alt.Tooltip("when:N", title=""), alt.Tooltip("what:N", title="Call")]))
layers.append(alt.Chart(_today.iloc[:1]).mark_text(align="left", dx=8, fontSize=11, fontWeight="bold",
                                                   color=ui.INK).encode(
    x=alt.X("m:T", scale=xs_t), y=y_rows, text="t:N"))
layers.append(alt.Chart(pd.DataFrame({"m": [hx.to_timestamp()]})).mark_rule(
    color=ui.INK, strokeWidth=2).encode(x=alt.X("m:T", scale=xs_t)))
# The chosen month's call sits two years on, so mark it there too, and inside
# a false alarm outline the alarm's calls, as the spell is outlined below.
if hx in _pred.index:
    layers.append(alt.Chart(pd.DataFrame([{"m": (hx + SHIFT).to_timestamp(), "row": ROW_PRED,
                                           "when": f"{HX_LABEL[hx]}'s call, drawn two years on"}]))
                  .mark_tick(color=ui.INK, thickness=3, size=26, opacity=1).encode(
                      x=alt.X("m:T", scale=xs_t), y=y_rows, tooltip=alt.Tooltip("when:N", title="")))
for a_, b_, _e in MISS_SPANS:
    if a_ <= hx <= b_:
        layers.append(alt.Chart(pd.DataFrame([{"start": (a_ + SHIFT).to_timestamp(),
                                               "end": (b_ + SHIFT + 1).to_timestamp(), "row": ROW_PRED}]))
                      .mark_rect(fill=None, stroke=ui.INK, strokeWidth=2).encode(
                          x=alt.X("start:T", scale=xs_t), x2="end:T", y=y_rows))
turn_df = pd.DataFrame([{"month": t_.to_timestamp(), "regime": r_, "row": ROW_TURN,
                         "when": f"{t_.strftime('%b %Y')} · into {LABEL[r_].lower()}",
                         "events": "; ".join(e["title"] for _, e in LEADS_INTO.get(t_, [])) or "–"}
                        for t_, r_ in zip(TURNS["start"], TURNS["label"])])
layers.append(alt.Chart(turn_df).mark_point(shape="triangle-down", size=70, filled=True, opacity=1).encode(
    x=alt.X("month:T", scale=xs_t), y=y_rows,
    color=alt.Color("regime:N", scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())),
                    legend=None),
    tooltip=[alt.Tooltip("when:N", title=""), alt.Tooltip("events:N", title="Led in by")]))
# A pin in the middle of each miss, where its calls are drawn.
if MISS_SPANS:
    miss_df = pd.DataFrame([{"m": (a_ + (b_ - a_).n // 2 + SHIFT).to_timestamp(), "row": ROW_PRED,
                             "what": e["title"],
                             "when": f'Called {a_.strftime("%b %Y")} to {b_.strftime("%b %Y")}; '
                                     f'pick a month in it for why'} for a_, b_, e in MISS_SPANS])
    layers.append(alt.Chart(miss_df).mark_point(shape="circle", size=90, filled=True, color=ui.INK,
                                                stroke="#fff", strokeWidth=1.5, opacity=1).encode(
        x=alt.X("m:T", scale=xs_t), y=y_rows,
        tooltip=[alt.Tooltip("what:N", title=""), alt.Tooltip("when:N", title="")]))
hx_box.altair_chart(ui.style(alt.layer(*layers).properties(height=130)), width="stretch")

# The chosen month beside the averages it feeds.
r_dur = during_lab.get(hx)


def _positions_at(month: pd.Period) -> pd.DataFrame:
    """Each regime's typical run-up as the call had it in `month`. Held out:
    learned without the month's episode and anything sharing its outcomes.
    Real time: learned only from what followed months already known then, and
    before 1976 today's."""
    from scripts.realign_positions import priors
    prior, train = priors(REG, NAMES), M["train"]
    if HELD:
        sp = _spell_of(going_lab, month) if month <= going_lab.index[-1] else None
        rest = train if sp is None else train[~ol.purged(train.index, sp["start"], sp["end"])]
        return ol.realigned_positions(rest, prior, NAMES)
    if month not in PRED["probs"].index:
        return pd.DataFrame({r_: {d: float(s_["archetype"][d]) for d in NAMES}
                             for r_, s_ in REG["regimes"].items()}).T
    known = train[train.index <= month - ol.HORIZON - ol.LABEL_LAG]
    return ol.realigned_positions(known, prior, NAMES)


def _turn_call(t0: pd.Period, new: str):
    """The run of calls for `new` that a turn into it is credited to: the last
    to start before the turn, if it held to the turn or, dropped before it,
    was made within a call's span of it. None if the model did not call it."""
    runs = ol.spells(_pred)
    runs = runs[(runs["label"] == new) & (runs["start"] < t0)]
    if runs.empty:
        return None
    run = runs.iloc[-1]
    return run if run["end"] >= t0 - 1 or (t0 - run["start"]).n <= CALL_SPAN else None


def _instead(t0: pd.Period, new: str) -> pd.Period:
    """Where to read a turn the model did not call: two years before it, or if
    an old call for `new` was still standing then, the month it was dropped."""
    made = t0 - SHIFT
    return _spell_of(_pred, made)["end"] + 1 if _pred.get(made) == new else made


# The radar pictures the call the text explains. At a turning point: the
# drivers when the model called it, or what it called instead, against the
# typical run-ups, as the call had them, to the regime that came and to the
# model's other pick. The drivers at the turn are in the table, not drawn:
# three shapes read, four did not. Elsewhere, and before the calls start, the
# month's own call.
# Inside a miss, the month's own call is the story, not the turn its spell began with.
in_miss = [e for a_, b_, e in MISS_SPANS if a_ <= hx <= b_]
t_turn = (sp_dur["start"] if not in_miss and sp_dur is not None and sp_dur["start"] in set(TURNS["start"])
          else None)
turn_run = _turn_call(t_turn, sp_dur["label"]) if t_turn is not None else None
seen_at = (None if t_turn is None else turn_run["start"] if turn_run is not None
           else _instead(t_turn, sp_dur["label"]))
if seen_at is not None and seen_at in PRED["probs"].index and PRED["called"].get(seen_at) in COLOR:
    m_f, came, pick = seen_at, sp_dur["label"], PRED["called"][seen_at]
    rival_f = pick if pick != came else PRED["probs"].loc[seen_at].drop(came).idxmax()
    refs = [(came, "what came"), (rival_f, "called" if pick != came else "runner-up")]
    rd_title = (f"When it called the turn: {HX_LABEL[m_f]}" if turn_run is not None
                else f"{(t_turn - m_f).n} months before the turn: {HX_LABEL[m_f]}")
    at_turn = t_turn
else:
    m_f, at_turn = hx, None
    odds_f = (PRED["probs"].loc[hx] if hx in PRED["probs"].index
              else R.probabilities(M["scores"].loc[[hx]], REG).iloc[0])
    pick = PRED["called"].get(hx) if PRED["called"].get(hx) in COLOR else odds_f.idxmax()
    refs = [(pick, "called"), (odds_f.drop(pick).idxmax(), "runner-up")]
    rd_title = HX_LABEL[hx]
pos_f = _positions_at(m_f)
# The two typical run-ups as light tinted areas, the month's drivers as one
# black line over them.
series = [{"name": f"Typical run-up to {INTEXT[r_]} ({role})", "values": pos_f.loc[r_].to_dict(),
           "color": COLOR[r_], "fill": 0.12, "width": 1.5} for r_, role in refs]
key_hx = [(f"Typical run-up to {INTEXT[r_]} ({role})", COLOR[r_], "solid") for r_, role in refs]
series.append({"name": f"Drivers, {HX_LABEL[m_f]}", "values": M["scores"].loc[m_f].to_dict(),
               "color": ui.INK, "fill": 0, "width": 2.5})
key_hx.insert(0, (f"Drivers, {HX_LABEL[m_f]}", ui.INK, "solid"))
c_chart, c_text = hx_box.columns([1, 1.3], gap="large")
c_chart.html(ui.radar_grid([ui.radar_svg(NAMES, SHORT, series, title=rd_title)], key_hx))
_light = [SHORT[d][0].lower() for d in NAMES if float(REG["driver_salience"].get(d, 1.0)) < 1]
c_chart.html('<p class="mr-caption">The model calls the regime whose typical run-up the black shape '
             'sits closest to' + (f', with {ui.join_words(_light)} counting less' if _light else '') + '. '
             + ('Typical run-ups are learned from every episode known, none of which this month’s '
                'three years belong to. ' if HELD and m_f > going_lab.index[-1] else
                'Typical run-ups are learned from every other episode, leaving out this one. ' if HELD
                else 'Typical run-ups are learned only from what was known then. '
                if m_f in PRED["probs"].index else
                'Before 1976 the typical run-ups are today’s, learned with hindsight. ')
             + 'Driver readings use today’s revised data.</p>')

def _per(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index).to_period("M")
    return out


ind_p, raw_p = _per(M["indicators"]), _per(M["inputs"])


def _gaps(row: pd.Series, horizon: str) -> str:
    """Growth and core inflation in plain numbers, against potential and expectations."""
    g, i_ = row["growth"], row["inflation"]
    pct = lambda v: f"{v:.1f}%".replace("-", "−")
    return (f"growth {horizon} {pct(row['gdp_growth'])} a year, {abs(g):.1f} points "
            f"{'above' if g >= 0 else 'below'} potential, and core PCE inflation "
            f"{pct(row['core_pce'])}, {abs(i_):.1f} points {'above' if i_ >= 0 else 'below'} "
            f"expectations")


def _held(row, label: str) -> str:
    """When the numbers on their own point elsewhere: the label holds until a
    change is clear and has lasted a year."""
    if row is None:
        return ""
    if row.get("raw") in INTEXT and row.get("raw") != label:
        return (f" On these numbers alone it would be {ui.esc(INTEXT[row['raw']])}; the label holds "
                f"until a change is clear and has lasted a year.")
    # A gap on the other side of zero but inside the margin a label needs
    # before it switches sides.
    side = {r_: k_ for k_, r_ in truth.QUADRANT.items()}.get(label)
    if side is None:
        return ""
    notes = []
    for name, v, up in (("Growth", row["growth"], side[0]), ("Inflation", row["inflation"], side[1])):
        if (v >= 0) != up:
            notes.append(f"{name} is only {abs(v):.1f} points {'above' if v >= 0 else 'below'} "
                         f"{'potential' if name == 'Growth' else 'expectations'}, too small a margin "
                         f"to count as a change of side.")
    return (" " + " ".join(notes)) if notes else ""


SALIENCE = pd.Series({d: float(REG["driver_salience"].get(d, 1.0)) for d in NAMES})


def _tag(r: str) -> str:
    return f'<span style="color:{COLOR[r]}">{ui.esc(INTEXT[r])}</span>'


def _side(v: float, what: str = "") -> str:
    return f"{abs(v):.1f} points {'above' if v >= 0 else 'below'}" + (f" {what}" if what else "")


# A driver reading in a word: toward its +1 end, its -1 end, or about neutral.
READS = {"growth": ("strong", "weak"), "inflation": ("hot", "cool"), "monetary": ("tight", "loose"),
         "credibility": ("strong", "weak"),
         "fiscal": ("supportive", "unsupportive"), "reforms": ("proactive", "passive")}


def _reads(d: str, v: float) -> str:
    return "about neutral" if abs(v) < 0.1 else READS.get(d, ("high", "low"))[0 if v > 0 else 1]


def _behind(d: str, month: pd.Period) -> list[tuple[dict, float, float, float]]:
    """Each indicator's part in driver `d` in `month`, largest first, as
    (indicator, part, score, reading). The parts sum to the driver's score
    before it is capped at ±1: the same weighted mean, block by block."""
    if month not in ind_p.index:
        return []
    row, inds = ind_p.loc[month], {i_["id"]: i_ for i_ in CFG["drivers"][d]["indicators"]}
    blocks = driver_groups(CFG, d)
    has = {g: [i for i in b["indicators"] if pd.notna(row.get(f"{d}::{i}"))] for g, b in blocks.items()}
    den = sum(blocks[g]["weight"] for g in blocks if has[g])
    parts = []
    for g, b in blocks.items():
        gden = sum(float(inds[i]["weight"]) for i in has[g])
        for i in has[g]:
            z = float(row[f"{d}::{i}"])
            parts.append((inds[i], b["weight"] / den * float(inds[i]["weight"]) / gden * z / DRIVER_SCALE,
                          z, raw_p.at[month, f"{d}::{i}"] if f"{d}::{i}" in raw_p.columns else np.nan))
    return sorted(parts, key=lambda p_: -abs(p_[1]))


KEEP_CASE = {"Cleveland", "Fed", "Fraser", "US"}  # first words that stay capitalized


def _name(i_: dict) -> str:
    """An indicator's label mid-sentence: lower case unless it opens with an
    acronym or a number (CBO, 10-year)."""
    lab = i_["label"]
    return ui.esc(lab if lab.split()[0] in KEEP_CASE or not lab[:2].istitle() else lab[0].lower() + lab[1:])


def _reading(i_: dict, v: float) -> str:
    return ui.format_reading(v, i_.get("transform", "level"), i_.get("unit"))


def _norm(i_: dict, z: float) -> str:
    """Where a reading sat against the indicator's norm, in words."""
    n = i_.get("normalize") or {}
    if n.get("method") != "gap":
        return ""
    raw_z = z * float(i_.get("direction", 1))
    side = "close to" if abs(raw_z) < 0.5 else ("well " if abs(raw_z) >= 1.5 else "") + (
        "above" if raw_z > 0 else "below")
    return f", {side} its norm of {_reading(i_, float(n['center']))}"


def _explain(d: str, month: pd.Period, sign: float, most: int = 2) -> list[str]:
    """The indicators that pushed driver `d` toward `sign` in `month`, each a
    bullet: its reading against its norm, then what that told us. Up to
    `most`, each at least a third of the largest push."""
    parts = [p_ for p_ in _behind(d, month) if p_[1] * sign > 0]
    keep = [p_ for p_ in parts if abs(p_[1]) >= abs(parts[0][1]) / 3][:most] if parts else []
    # Labels end in a tag such as "(falling +)" for the chart; the sentence says it.
    return [f"<b>{ui.esc(re.sub(r' [(][^)]*[+][)]$', '', i_['label']))}: {_reading(i_, v)}</b>{_norm(i_, z)}. "
            + ui.esc((i_.get("reads") or {}).get("high" if z > 0 else "low", ""))
            for i_, _, z, v in keep]


def _call_why(made: pd.Period, call: str, rival: str, drivers: int = 3, per: int = 2) -> list[str]:
    """The drivers that most favored `call` over `rival` in `made`, against
    each regime's typical run-up as the call had it, each with the indicators
    behind it nested under it. Up to `drivers`, with up to `per` indicators."""
    pos = _positions_at(made)
    x = M["scores"].loc[made, NAMES]
    pull = SALIENCE * ((x - pos.loc[rival]) ** 2 - (x - pos.loc[call]) ** 2)
    top = [d for d in pull.sort_values(ascending=False).index if pull[d] > 0][:drivers]
    out = []
    for d in top:
        # Which way the driver had to lean to favor the call over the rival.
        sub = _explain(d, made, np.sign(pos.at[call, d] - pos.at[rival, d]), per)
        out.append(f"<b>{ui.esc(SHORT[d][0])} read {_reads(d, x[d])}</b> ({ui.signed(x[d])}), nearer a typical "
                   f"{ui.esc(INTEXT[call])} run-up ({ui.signed(pos.at[call, d])}) than a typical "
                   f"{ui.esc(INTEXT[rival])} one ({ui.signed(pos.at[rival, d])})."
                   + (f'<ul>{"".join(f"<li>{t}</li>" for t in sub)}</ul>' if sub else ""))
    return out


def _months(n: int) -> str:
    return f"{n} month{'s' if n != 1 else ''}"


def _moved(d: str, a: pd.Period, b: pd.Period) -> str:
    """The indicator whose part in driver `d` changed most from `a` to `b`."""
    pa = {p_[0]["id"]: p_ for p_ in _behind(d, a)}
    pb = {p_[0]["id"]: p_ for p_ in _behind(d, b)}
    both = [i for i in pa if i in pb]
    if not both:
        return ""
    i = max(both, key=lambda k: abs(pb[k][1] - pa[k][1]))
    i_ = pa[i][0]
    return f", led by {_name(i_)} ({_reading(i_, pa[i][3])} to {_reading(i_, pb[i][3])})"


def _item(head: str, body: str = "", sub=()) -> str:
    """One bullet: a bold lead, its sentence, and any bullets nested under it."""
    inner = "".join(f"<li>{s}</li>" for s in sub)
    return f'<li><b>{head}</b>{body}{f"<ul>{inner}</ul>" if inner else ""}</li>'


def _event(m_: pd.Period, e: dict) -> str:
    return f'<b>{ui.esc(e["title"])}</b>, {m_.strftime("%b %Y")}. {ui.esc(e["note"])}'


# Two sections, each a few bullets: the turning point that began this spell
# (what changed, the events that led in, what the model had called two years
# before), then this month (what it was living through).
sections = []
turn_events = []
if t_turn is not None:
    t0, new = sp_dur["start"], sp_dur["label"]
    prev_sp = _spell_of(during_lab, t0 - 1)
    nc = M["nowcast"]
    before = nc.loc[prev_sp["start"]:prev_sp["end"], ["growth", "inflation"]].mean()
    after = nc.loc[t0:sp_dur["end"], ["growth", "inflation"]].mean()
    turn_events = LEADS_INTO.get(t0, [])
    items = [_item(f'Into {_tag(new)}, from {_tag(prev_sp["label"])}.',
                   f' Growth went from {_side(before["growth"], "potential")} to '
                   f'{_side(after["growth"])}; core PCE inflation from '
                   f'{_side(before["inflation"], "expectations")} to {_side(after["inflation"])} '
                   f'(averages over each spell).'
                   + (" What led in:" if turn_events else ""),
                   [_event(m_, e) for m_, e in turn_events])]
    made = seen_at
    call = PRED["called"].get(made)
    if turn_run is not None:
        sw, last = turn_run["start"], turn_run["end"]
        odds = PRED["probs"].loc[sw]
        rival = odds.drop(new).idxmax()
        moved = (f' The model moved to {_tag(new)} in {sw.strftime("%b %Y")} ({odds[new]:.0%}, against '
                 f'{ui.esc(INTEXT[rival])} {odds[rival]:.0%})')
        if last >= t0 - 1:
            moved += ' and held it to the turn. What it saw when it called it:'
        else:
            off = last + 1
            moved += (f', then dropped it for {_tag(PRED["called"][off])} in {off.strftime("%b %Y")}, '
                      f'{_months((t0 - off).n)} before the turn. What it saw when it called it:')
        items.append(_item(f'Called it {_months((t0 - sw).n)} in advance.', moved, _call_why(sw, new, rival)))
        # Early: the call ran back well before the run-up the answer key gives,
        # through months whose next three years went otherwise.
        k_sp = _spell_of(going_lab, t0 - SHIFT)
        if k_sp is not None and k_sp["label"] == new and (k_sp["start"] - sw).n > EARLY:
            k0 = k_sp["start"]
            was = going_lab.loc[sw:k0 - 1].value_counts().idxmax()
            items.append(_item('Too early at first.',
                               f' Calls made from {sw.strftime("%b %Y")} to {(k0 - 1).strftime("%b %Y")} were '
                               f'wrong: the three years after them turned out {_tag(was)}. The drivers caught '
                               f'the build-up, not its timing: the same signs can sit for years before a turn.'))
    elif made in PRED["probs"].index and call in COLOR:
        odds = PRED["probs"].loc[made]
        x0, x1 = M["scores"].loc[made, NAMES], M["scores"].loc[t0, NAMES]
        d = (x1 - x0).abs().idxmax()
        items.append(_item('Not called in advance.',
                           f' {_months((t0 - made).n)} before, in {made.strftime("%b %Y")}, the model called '
                           f'{_tag(call)} ({odds[call]:.0%}; {ui.esc(INTEXT[new])} {odds[new]:.0%}), because:',
                           _call_why(made, call, new)
                           + [f"<b>What changed after:</b> {ui.esc(SHORT[d][0].lower())} moved most "
                              f"by the turn, from {ui.signed(x0[d])} to {ui.signed(x1[d])}"
                              f"{_moved(d, made, t0)}."]))
    else:
        items.append(_item('Not called in advance.',
                           ' The data start in 1968.' if HELD else
                           ' The model’s calls start in 1976, once there are past regimes to learn from.'))
    sections.append((f'Turning point, {t0.strftime("%b %Y")}', items))
# This month: what it was living through, with its own events.
items = [_item(f'{ui.esc(e["title"])}.', f' {ui.esc(e["note"])}') for e in in_miss]
shown = {m_ for m_, _ in turn_events}
ev = EV_BY_MONTH.get(hx)
sub = []
if ev and hx not in shown:
    sub.append(_event(hx, ev))
    shown.add(hx)
if sp_dur is not None:
    also = [(m_, e) for m_, e in EV_BY_MONTH.items()
            if sp_dur["start"] <= m_ <= sp_dur["end"] and m_ not in shown and _turn_of(m_) is None]
    if also:
        sub.append("Also in this spell: " + "; ".join(f'{ui.esc(e["title"])} ({m_.strftime("%b %Y")})'
                                                     for m_, e in also) + ".")
if r_dur is not None and sp_dur is not None:
    nc = M["nowcast"].loc[hx] if hx in M["nowcast"].index else None
    body = f": over the 18 months around it, {_gaps(nc, 'ran')}." if nc is not None else "."
    sub.insert(0, f'Part of the {sp_dur["start"].strftime("%b %Y")} to {sp_dur["end"].strftime("%b %Y")} '
                  f'spell.{_held(nc, r_dur)}')
    items.append(_item(f"Living through {_tag(r_dur)}", body, sub))
elif sub:
    items.append(_item("Events", "", sub))
if not items and hx > during_lab.index[-1]:
    _last = during_lab.index[-1]
    items.append(_item("Too recent to label",
                       f": living through reads the three months after a month, so the latest is "
                       f'{_last.strftime("%b %Y")}, {_tag(during_lab.iloc[-1])}.'))
if items:
    sections.append((f"This month, {HX_LABEL[hx]}", items))
c_text.html('<div class="hx">' + "".join(f'<div class="hx-h">{t}</div><ul>{"".join(i)}</ul>'
                                         for t, i in sections) + "</div>")
# The radar's shapes as numbers.
_cols = [m_f] + ([at_turn] if at_turn is not None else [])
c_text.html(ui.table(
    [""] + [ui.Raw(f"Drivers,<br>{HX_LABEL[m_]}") if m_ == m_f else ui.Raw(f"At the turn,<br>{HX_LABEL[m_]}")
            for m_ in _cols]
    + [ui.Raw(f"Typical run-up,<br>{ui.esc(HEAD[r_])} ({role})") for r_, role in refs],
    [[SHORT[d][0]] + [ui.signed(M["scores"].at[m_, d]) for m_ in _cols]
     + [ui.signed(pos_f.at[r_, d]) for r_, _ in refs] for d in NAMES],
    numeric=set(range(1, 1 + len(_cols) + len(refs)))))

# The indicators behind each driver in the month the radar shows, as read (in
# their own units) and as scored, and at the turn when there is one. Folded:
# thirty-odd rows.


def _cell(frame: pd.DataFrame, row, col: str, fmt) -> str:
    v = frame.at[row, col] if row in frame.index and col in frame.columns else None
    return "–" if v is None or pd.isna(v) else fmt(v)


rows_ind = []
for d in NAMES:
    rows_ind.append(ui.Raw(f"<b>{ui.esc(DLABEL[d])}</b> · driver "
                           + " → ".join(ui.signed(M["scores"].at[m_, d]) for m_ in _cols)))
    total = sum(float(i_["weight"]) for i_ in CFG["drivers"][d]["indicators"])
    for i_ in sorted(CFG["drivers"][d]["indicators"], key=lambda x: -float(x["weight"])):
        col, tr = f"{d}::{i_['id']}", i_.get("transform", "level")
        read = lambda v, tr=tr, un=i_.get("unit"): ui.format_reading(v, tr, un)
        rows_ind.append([
            ui.Raw(f'{ui.esc(i_["label"])}<br><span style="color:{ui.MUTED};font-size:0.8rem">'
                   f'{ui.esc(ui.TRANSFORM_TEXT.get(tr, tr))}</span>'),
            f"{float(i_['weight']) / total:.0%}",
            *[c_ for m_ in _cols for c_ in (_cell(raw_p, m_, col, read), _cell(ind_p, m_, col, ui.signed))]])
with hx_box.expander("The indicators behind each driver"):
    st.html('<p class="mr-caption">Each indicator as read, in its own units, and as scored: '
            'standard deviations from its norm, capped at ±3, signed so positive points to the '
            'driver\u2019s +1 end. Weight is its share of the driver. A dash is an indicator not '
            'yet measured that month.</p>')
    st.html(ui.table(
        ["Indicator", "Weight"] + [ui.Raw(f"{k},<br>{HX_LABEL[m_]}") for m_ in _cols for k in ("Reading", "Score")],
        rows_ind, numeric=set(range(1, 2 + 2 * len(_cols)))))

# ---------- 0c. how well the call has done ----------


@st.cache_data(ttl=6 * 3600, show_spinner="Grading every month since 1976…")
def record(version: str, config_key: str) -> dict | None:
    """The call's record, recomputed from the store (src/evaluation.py)."""
    from src import evaluation
    if M["train"]["label"].nunique() < 4:
        return None
    return evaluation.evaluate(M["train"], M["nowcast"]["label"], REG, NAMES)


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def regime_name(c: str) -> str:
    return LABEL.get(c) or ui.call_label(c, REG)


EV = record(data_version(), CONFIG_KEY)
if EV is not None:
    a, b = EV["window"]
    G, BR = EV["methods"], EV["by_regime"]
    lo, hi = EV["bootstrap"]["balanced"]
    st.html(ui.section_head(
        "How well the call has done", f'Held out · {a.strftime("%b %Y")} to {b.strftime("%b %Y")}',
        f'Every month from {a.strftime("%B %Y")} to {b.strftime("%B %Y")}, the last whose next three '
        f'years are known, graded against what those three years turned out to be: '
        f'{EV["months"]} months, {EV["bootstrap"]["episodes"]} episodes. Each episode is called '
        f'from positions learned without it or any month sharing its three years. Recomputed '
        f'from the live data.'))

    st.html('<h3 class="m-h3">Against simple benchmarks</h3>')

    def bench_row(name: str, g: dict) -> list[str]:
        return [name, pct(g["balanced"]), pct(g["months"]),
                f'{g["hard_landing_caught"]} of {g["hard_landing_runups"]}',
                "–" if pd.isna(g["log_loss"]) else f'{g["log_loss"]:.2f}']

    rows = [[ui.Raw(f"<b>{ui.esc(c)}</b>") for c in bench_row("Held out", G["held_out"])]]
    rows += [["Chance", "25%", "25%", "–", "1.39"],
             bench_row("Current regime continues", G["continues"]),
             bench_row("Sign of the growth and inflation drivers", G["signs"])]
    st.html(ui.table(["", "Regimes right, average", "Months right", "Hard-landing run-ups caught",
                      "Log loss"], rows, numeric={1, 2, 3, 4}))
    st.html(
        '<div class="m-body"><ul>'
        '<li><b>Chance</b> picks one regime in four.</li>'
        '<li><b>Current regime continues</b> calls, each month, the regime the economy was in '
        'three months before as the regime for the next three years. The answer key reads the '
        'present from growth and inflation over the 18 months around a month, three of them '
        'after it, so three months back is the latest it can label. The call changes whenever '
        'the economy does, three months late: the forecast of someone who expects no '
        'change.</li>'
        '<li><b>Sign of the growth and inflation drivers</b> reads those two scores by sign '
        'alone: both above zero is high growth with high inflation, growth above and inflation '
        'below is goldilocks, and so on. It is the model\'s two main inputs without the learned '
        f'positions or the other {ui.NUMBER_WORDS[len(NAMES) - 2]} drivers, so the gap to the '
        'held-out grade is what those add.</li>'
        '<li><b>Regimes right, average</b> is the share of each regime\'s months called right, '
        'averaged over the four, so goldilocks, which followed '
        f'{BR.at["goldilocks", "base_rate"]:.0%} of months, does not dominate. <b>Months right</b> '
        'is the plain share. A hard-landing run-up counts as caught when at least half its '
        'months were called hard landing. <b>Log loss</b> grades the odds rather than the call: '
        'lower is better, and an even split scores 1.39.</li>'
        '<li><b>The grade uses today\'s revised data</b>, with each indicator\'s center and scale '
        'set on the full history, so it is not the call as it would have been made at the time. '
        'The live call log, kept from November 2026, is.</li></ul></div>')

    st.html('<h3 class="m-h3">How much to trust the headline</h3>')
    st.html(
        f'<div class="m-body"><p>Neighboring months share most of their answer, so the '
        f'{EV["months"]} months are really {EV["bootstrap"]["episodes"]} episodes. Resampling '
        f'whole episodes puts the held-out average between <b>{lo:.0%} and {hi:.0%}</b> (5th to '
        '95th percentile). With the driver readings slid out of line with what followed, none of '
        '40 runs matched it (median 33%, best 62%; run 2 November 2026). The grade still '
        'flatters the model somewhat: the indicators, weights and settings were chosen with it '
        'in view.</p></div>')

    st.html('<h3 class="m-h3">By regime</h3>')
    AX = EV["axes"]
    st.html(ui.table(
        ["Held out", "Share of months that followed", "Months called", "Precision", "Recall", "F1"],
        [[LABEL[r], pct(BR.at[r, "base_rate"]), f'{int(BR.at[r, "called"])}',
          pct(BR.at[r, "precision"]), pct(BR.at[r, "recall"]), f'{BR.at[r, "f1"]:.2f}']
         for r in BR.index]
        + [[n, pct(AX.at[k, "base_rate"]), f'{int(AX.at[k, "called"])}',
            pct(AX.at[k, "precision"]), pct(AX.at[k, "recall"]), f'{AX.at[k, "f1"]:.2f}']
           for k, n in (("growth_below", "Growth below potential"),
                        ("inflation_above", "Inflation above expectations"))],
        numeric={1, 2, 3, 4, 5}))
    worst = BR["precision"].idxmin()
    keen = BR["recall"].idxmax()
    plo, phi = EV["bootstrap"]["precision"][worst]
    st.html(
        '<div class="m-body"><p><b>Precision</b>: of the months the model called a regime, the '
        'share it followed. <b>Recall</b>: of the months a regime followed, the share the model '
        'called it. <b>F1</b> is high only when both are. The bottom two rows score the two '
        'halves of the call on their own.</p><ul>'
        f'<li>{ui.esc(LABEL[keen])} is the hardest to miss ({pct(BR.at[keen, "recall"])} '
        f'recall) but often a false alarm ({pct(BR.at[keen, "precision"])} precision).</li>'
        f'<li>{ui.esc(LABEL[worst])} is the least reliable when called: '
        f'{pct(BR.at[worst, "precision"])}, against the {pct(BR.at[worst, "base_rate"])} of '
        f'months it followed. Resampling episodes puts it anywhere from {plo:.0%} to {phi:.0%}.'
        + (' It is today\'s call.' if called == worst else '') + '</li>'
        + (f'<li>Today\'s call, {ui.esc(LABEL[called])}, has followed '
           f'{pct(BR.at[called, "precision"])} of the months it was called.</li>'
           if called in BR.index and called != worst else '')
        + '</ul></div>')

    st.html('<h3 class="m-h3">Where the misses go</h3>')
    conf = EV["confusion"].fillna(0).astype(int)
    # Called in the same order as followed, so the diagonal is the hits.
    conf = conf[[r for r in conf.index if r in conf.columns]
                + [c for c in conf.columns if c not in conf.index]]
    st.html(ui.table(["What followed ↓ · called →"] + [regime_name(c) for c in conf.columns],
                     [[LABEL[r]] + [f"{v}" for v in conf.loc[r]] for r in conf.index],
                     numeric=set(range(1, len(conf.columns) + 1))))
    st.html('<div class="m-body"><p>The longest unbroken wrong calls:</p><ul>'
            + "".join(f'<li>{x.start.strftime("%b %Y")} to {x.end.strftime("%b %Y")}, {x.months} '
                      f'months: called {ui.esc(regime_name(x.called).lower())}, followed by '
                      f'{ui.esc(LABEL[x.followed].lower())}.</li>'
                      for x in EV["misses"].itertuples())
            + '</ul><p>The timeline above pins the largest false alarms, with what went '
            'wrong.</p></div>')

    st.html('<h3 class="m-h3">Are the odds right?</h3>')
    cal = EV["calibration"]
    st.html(ui.table(
        ["Odds on the leading regime", "Months", "Average odds", "Leader followed"],
        [[f"{iv.left:.0%} to {iv.right:.0%}", f'{int(r.months)}', pct(r.stated), pct(r.right)]
         for iv, r in zip(cal.index, cal.itertuples())],
        numeric={1, 2, 3}))
    top_band = cal.iloc[-1]
    years = ui.join_words([str(y) for y in sorted(EV["confident_years"].index[:3])])
    st.html('<div class="m-body">'
            + (f'<p>Up to {cal.index[-1].left:.0%}, the leader followed about as often as its odds '
               f'said, or more. Above that it followed {pct(top_band.right)} of the time, against '
               f'odds of {pct(top_band.stated)}: {int(top_band.months)} months, most of them in '
               f'{years}. Read odds above {cal.index[-1].left:.0%} as no more reliable than lower '
               'ones.</p>'
               if top_band.right < top_band.stated - 0.1 else
               '<p>In every band the leader followed about as often as its odds said.</p>')
            + '</div>')

    st.html('<h3 class="m-h3">What each driver adds</h3>')
    full = G["held_out"]["balanced"]

    def change(v: float) -> str:
        # From the rounded scores, so the column agrees with the one beside it.
        d = round(v * 100) - round(full * 100)
        return "no change" if d == 0 else f'{d:+d} point{"s" * (abs(d) != 1)}'.replace("-", "−")

    st.html(ui.table(
        ["Held out, regimes right on average", "Score", "Change"],
        [["All drivers", pct(full), ""]]
        + [[f"Without {DLABEL[d].lower()}", pct(v), change(v)]
           for d, v in sorted(EV["without"].items(), key=lambda kv: kv[1])]
        + [["Growth and inflation only", pct(EV["growth_inflation_only"]),
            change(EV["growth_inflation_only"])]],
        numeric={1, 2}))
    st.html('<div class="m-body"><p>Each row takes drivers out, learns the positions again '
            'without them, and grades the call held out, as in the first table. A driver whose '
            'removal costs little is kept for what it measures rather than for its record over '
            f'three years, and {EV["bootstrap"]["episodes"]} episodes cannot tell a few points '
            'apart.</p></div>')

# The call and how it is made, folded under the sections above.
with st.expander("The three-year call: odds, landing point, drivers and what-if",
                 expanded=False):
    # ---------- 1. the call ----------

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
                    width="stretch")

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
        "Set monetary stance and fiscal policy where you think they are heading; the odds and the "
        "landing point update. Everything else, credibility included, stays as it reads today."))
    c1, c2 = st.columns(2)
    mon = c1.slider("Monetary stance (dovish − / hawkish +)", -1.0, 1.0,
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

    # ---------- how it works ----------

    st.html(ui.section_head("How it works"))
    st.html(
        '<div class="m-body"><ul>'
        f'<li><b>{ui.NUMBER_WORDS[len(NAMES)].capitalize()} drivers, read monthly:</b> '
        + ui.esc(ui.join_words([DLABEL[d] if i == 0 else DLABEL[d].lower() for i, d in enumerate(NAMES)]))
        + '. Each is a weighted average of indicators chosen for their economic link and '
        'checked against history.</li>'
        f'<li><b>Each regime has a position:</b> where the {ui.NUMBER_WORDS[len(NAMES)]} drivers sat '
        'in the three years before it, learned from fifteen run-ups since 1968 (three or four per '
        'regime) and nudged toward a judgment of where the regime sits. Policy is part of the '
        'position: hard landings came after tight policy with credibility intact, stagflation '
        'after tight policy without it, high inflation after policy that had lost its '
        'credibility.</li>'
        '<li><b>The call</b> compares this month\'s drivers with each position; the closer, the '
        'likelier. '
        + ui.esc(ui.join_words([DLABEL[d] if i == 0 else DLABEL[d].lower()
                                for i, d in enumerate(d for d in NAMES
                                                      if float(REG["driver_salience"].get(d, 1)) < 1)]))
        + ' count at half weight. A new call has to lead for '
        f'{int(REG["settings"]["persistence_months"])} months in a row before it changes.</li>'
        '<li><b>Where growth and inflation are heading</b> weighs what followed each regime\'s '
        'run-ups by today\'s odds.</li>'
        '<li><b>Before 1979</b> the inflation anchor is the Fed\'s FRB/US estimate of long-run '
        'expectations, not a survey; everything is on today\'s revised data.</li>'
        '</ul></div>')
