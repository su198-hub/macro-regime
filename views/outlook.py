"""The regime explorer (/outlook, its address from when it was the three-year
outlook): what each regime looked like, any month since 1968, and v2's call
for where growth and inflation are heading.

It reads the same v2 drivers and scenario positions as the dashboard, over
the four regimes the answer key can grade, whichever framework the reader has
picked elsewhere.

  0. Regime shapes, first: where the drivers sat going into each regime and
     during it, against theory, one radar per regime beside today's.
  0b. Go back in time: each call, held out or in real time, drawn over the
     months it was about, above what they turned out to be, and what led into
     each turning point. A
     radar shows what the call two years before a turn saw, against the
     typical run-ups to what came and to its other pick, and the text says
     which drivers and indicators decided it.

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
    "How the five drivers looked going into each regime and during it, set against theory: "
    "where each regime should sit by its definition. Theory is a direction, not a reading."))
PAIRS = {"Going in vs theory": ("going", "theory"), "Going in vs during": ("going", "during"),
         "During vs theory": ("during", "theory")}
c_pair, c_months = st.columns([1.4, 1])
pair_label = c_pair.segmented_control("Compare", list(PAIRS), default="Going in vs theory",
                                      key="shape_pair") or "Going in vs theory"
months = c_months.segmented_control("Months", ["All", "Strongest quarter"], default="All",
                                    key="shape_months") or "All"
strong = months == "Strongest quarter"
shape = {
    "going": M["core_in"] if strong else ol.positions(M["train"], NAMES),
    "during": M["core_during"] if strong else ol.positions(M["during"], NAMES),
    "theory": pd.DataFrame({r: v["theory"] for r, v in REG["regimes"].items()
                            if v.get("theory")}).T.reindex(columns=NAMES),
}
NAME = {"going": "Going in", "during": "During", "theory": "Theory"}
KEY_NAME = {"going": "Going in: the three years before", "during": "During", "theory": "Theory"}
shape_a, shape_b = PAIRS[pair_label]
# The first shape of the pair is filled, the second dashed; theory is always
# a thin grey dotted line, so data and judgment never look alike.
LOOK = {shape_a: ({"fill": 0.18}, "solid"), shape_b: ({"fill": 0, "dash": True}, "dashed")}
LOOK["theory"] = ({"fill": 0, "dash": "1.5 2.5", "width": 1.4, "grey": True}, "dotted")
# Each spoke names its driver and what its outer end means, kept short so
# five fit round a small chart.
POLE = {"inflation": "Above anchor"}
SHORT = {d: (DLABEL[d].split(" and ")[0],
             POLE.get(d, CFG["drivers"][d]["signpost"]["high"]["label"].split(" / ")[0]))
         for d in NAMES}
today_panel = ui.radar_svg(NAMES, SHORT, [{"name": f"{when:%b %Y}", "values": now.iloc[0].to_dict(),
                                            "color": ui.INK, "fill": 0.10}],
                          title=f"Today ({when:%b %Y})", ring_labels=True)
panels = []
for r in REG["regimes"]:
    series = []
    for kind in (shape_b, shape_a) if shape_b == "theory" else (shape_a, shape_b):
        if r in shape[kind].index:
            look = dict(LOOK[kind][0])
            color = ui.INK_2 if look.pop("grey", False) else COLOR[r]
            series.append({"name": NAME[kind], "values": shape[kind].loc[r].to_dict(),
                           "color": color, **look})
    panels.append(ui.radar_svg(NAMES, SHORT, series, title=HEAD[r], title_color=COLOR[r]))
st.html(ui.radar_grid(
    panels, [(KEY_NAME[k], ui.INK_2, LOOK[k][1]) for k in (shape_a, shape_b)], lead=today_panel,
    key=[("Scale", "The centre is −1, the middle ring neutral and the outer ring +1, toward the "
                   "end named under each driver. Hover a chart for its scores."),
         ("Going in and during", "Going in is the three years before each regime, its run-ups, "
                                 "which the call learns from. During is the months whose growth "
                                 "and inflation over the 18 months around them (15 before, 3 "
                                 "after) fit the regime, so it trails events by about six months."),
         ("Drivers look ahead", "So going in is what compares with theory; during a regime they "
                                "already read the policy response and the next phase."),
         ("Direction over size", "An average of many months sits closer to neutral than theory. "
                                 "Strongest quarter keeps only each regime’s deepest months.")]))
# Where the two shapes part ways: a card per regime with its two biggest
# gaps and a note, the theory note whenever theory is one of the pair.
_diff = (shape[shape_b] - shape[shape_a]).reindex(list(REG["regimes"]))
_note = "theory" if "theory" in (shape_a, shape_b) else "during"
_ring = "dotted " + ui.INK_2 if shape_b == "theory" else "dashed "
cards = []
for r in REG["regimes"]:
    if r not in _diff.index or _diff.loc[r].isna().all():
        continue
    top = _diff.loc[r].abs().sort_values(ascending=False).index[:2]
    cards.append({"title": HEAD[r], "color": COLOR[r],
                  "note": (REG["regimes"][r].get("notes") or {}).get(_note, ""),
                  "ring": _ring if shape_b == "theory" else _ring + COLOR[r],
                  "rows": [(SHORT[d][0], float(shape[shape_a].at[r, d]),
                            float(shape[shape_b].at[r, d])) for d in top]})
st.html(ui.gap_cards(cards, NAME[shape_a], NAME[shape_b]))
_runs = ol.spells(M["train"]["label"])["label"].value_counts()
_spells = ol.spells(M["during"]["label"])["label"].value_counts()
_count = {"going": lambda r: f"{int(_runs.get(r, 0))} run-ups",
          "during": lambda r: f"{int(_spells.get(r, 0))} spells", "theory": lambda r: ""}
rows = []
for r in REG["regimes"]:
    rows.append(ui.Raw(f'<b style="color:{COLOR[r]}">{ui.esc(HEAD[r])}</b>'))
    for kind in ("theory", "going", "during"):
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
EVENTS = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config" / "history_events.yml",
                             encoding="utf-8"))["events"]
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
hx_mode = hx_box.radio("How the calls were made", list(HX_MODES), horizontal=True, key="hx_mode")
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

# A call covers the next three years on average, so its middle is about a
# year and a half out; living through covers the 18 months to three months
# ahead, so its middle is about six months back. Drawn two years after it was
# made, a call sits over the months it was about: shifted two years, the
# answer key agrees with living through in 86% of months, against 71% at three
# years (checked 29 Oct 2026, 1968 to 2023).
SHIFT = 24
EARLY = 6  # months a call may run ahead of the run-up before the text calls it early
# A call that switches to the coming regime within this many months of two
# years before the turn, and before the turn itself, still counts as seen:
# the persistence rule alone can hold a call back six months.
LATE = 6
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
TURN_OPTS.update({f'{HX_LABEL[t_]} · into {HEAD[r_]}'
                  + (f' · {LEADS_INTO[t_][-1][1]["title"]}' if t_ in LEADS_INTO else ""): HX_LABEL[t_]
                  for t_, r_ in zip(TURNS["start"], TURNS["label"]) if t_ in HX_LABEL})
if st.session_state.get("hx_month") not in HX_BY_LABEL:
    st.session_state["hx_month"] = HX_LABEL.get(pd.Period("2008-03", "M"), HX_LABEL[HX_MONTHS[-1]])


def _hx_jump() -> None:
    pick = st.session_state.get("hx_turn")
    if pick in TURN_OPTS:
        st.session_state["hx_month"] = TURN_OPTS[pick]


c_ev, c_mo = hx_box.columns([1, 1.6])
c_ev.selectbox("Jump to today or a turning point", ["Choose…"] + list(TURN_OPTS), key="hx_turn",
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
turn_df = pd.DataFrame([{"month": t_.to_timestamp(), "regime": r_, "row": ROW_TURN,
                         "when": f"{t_.strftime('%b %Y')} · into {LABEL[r_].lower()}",
                         "events": "; ".join(e["title"] for _, e in LEADS_INTO.get(t_, [])) or "–"}
                        for t_, r_ in zip(TURNS["start"], TURNS["label"])])
layers.append(alt.Chart(turn_df).mark_point(shape="triangle-down", size=70, filled=True, opacity=1).encode(
    x=alt.X("month:T", scale=xs_t), y=y_rows,
    color=alt.Color("regime:N", scale=alt.Scale(domain=list(COLOR), range=list(COLOR.values())),
                    legend=None),
    tooltip=[alt.Tooltip("when:N", title=""), alt.Tooltip("events:N", title="Led in by")]))
hx_box.altair_chart(ui.style(alt.layer(*layers).properties(height=130)), use_container_width=True)

# The chosen month beside the averages it feeds.
r_in, r_dur = going_lab.get(hx), during_lab.get(hx)


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


# The radar pictures the call the text explains. At a turning point: the
# drivers two years before it, which the call read, against the typical
# run-ups, as the call had them, to the regime that came and to the model's other
# pick. The drivers at the turn are in the table, not drawn: three shapes
# read, four did not. Elsewhere, and before the calls start, the month's own
# call.
t_turn = sp_dur["start"] if sp_dur is not None and sp_dur["start"] in set(TURNS["start"]) else None
seen_at = t_turn - SHIFT if t_turn is not None else None
if seen_at is not None and seen_at in PRED["probs"].index and PRED["called"].get(seen_at) in COLOR:
    m_f, came, pick = seen_at, sp_dur["label"], PRED["called"][seen_at]
    rival_f = pick if pick != came else PRED["probs"].loc[seen_at].drop(came).idxmax()
    refs = [(came, "what came"), (rival_f, "called" if pick != came else "runner-up")]
    rd_title, at_turn = f"Two years before the turn: {HX_LABEL[m_f]}", t_turn
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


def _span(n: int) -> str:
    """A number of months in years and months."""
    y, m_ = divmod(n, 12)
    parts = ([f"{y} year{'s' if y != 1 else ''}"] if y else []) + ([f"{m_} month{'s' if m_ != 1 else ''}"] if m_ else [])
    return " and ".join(parts)


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


def _basis(month: pd.Period) -> str:
    """How the call in `month` was made, in words."""
    if not HELD:
        return "from what was known then"
    if month > going_lab.index[-1]:
        return "from positions learned from every episode known so far"
    return "from positions learned without this episode"


def _chip(right: bool) -> str:
    return f' <span class="hx-chip{"" if right else " miss"}">{"Right" if right else "Wrong"}</span>'


# Two sections, each a few bullets: the turning point that began this spell
# (what changed, the events that led in, what the model had called two years
# before), then this month (what it was living through, what it called).
sections = []
turn_events = []
if sp_dur is not None and sp_dur["start"] in set(TURNS["start"]):
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
    made = t0 - SHIFT
    call = PRED["called"].get(made)
    if made in PRED["probs"].index and call in COLOR:
        odds = PRED["probs"].loc[made]
        # When the call switched to the coming regime, if it did before the turn.
        switched = PRED["called"].loc[made:t0 - 1]
        switched = switched.index[switched == new]
        if call == new:
            rival = odds.drop(call).idxmax()
            items.append(_item('Seen two years earlier.',
                               f' In {made.strftime("%b %Y")} the model called {_tag(call)} for the next '
                               f'three years ({odds[call]:.0%}, against {ui.esc(INTEXT[rival])} '
                               f'{odds[rival]:.0%}), because:', _call_why(made, call, rival)))
            # Early: the same call ran back well before the run-up the answer
            # key gives, through months whose next three years went otherwise.
            c_sp, k_sp = _spell_of(PRED["called"].dropna(), made), _spell_of(going_lab, made)
            if (c_sp is not None and k_sp is not None and k_sp["label"] == new
                    and (k_sp["start"] - c_sp["start"]).n > EARLY):
                c0, k0 = c_sp["start"], k_sp["start"]
                was = going_lab.loc[c0:k0 - 1].value_counts().idxmax()
                items.append(_item('But early.',
                                   f' The model had called {_tag(new)} since {c0.strftime("%b %Y")}, '
                                   f'{_span((t0 - c0).n)} before the turn. Calls made from then to '
                                   f'{(k0 - 1).strftime("%b %Y")} were wrong: the three years after them '
                                   f'turned out {_tag(was)}. In {c0.strftime("%b %Y")} it read:',
                                   _call_why(c0, new, was, 2, 1)
                                   + ["<b>So</b> the drivers caught the build-up, not its timing: the "
                                      "same signs can sit for years before a turn."]))
        elif len(switched) and (switched[0] - made).n <= LATE:
            sw = switched[0]
            ahead = "the odds already favored it" if odds.idxmax() == new else "the odds were moving its way"
            items.append(_item(f'Seen, {_span((sw - made).n)} late.',
                               f' In {made.strftime("%b %Y")} the model still called {_tag(call)}, but '
                               f'{ahead} ({ui.esc(INTEXT[new])} {odds[new]:.0%}, {ui.esc(INTEXT[call])} '
                               f'{odds[call]:.0%}). A new regime has to lead for '
                               f'{int(REG["settings"]["persistence_months"])} months in a row before the '
                               f'call moves, so it switched to {_tag(new)} in {sw.strftime("%b %Y")}, '
                               f'{_span((t0 - sw).n)} before the turn. What the odds read in '
                               f'{made.strftime("%b %Y")}:', _call_why(made, new, call)))
        else:
            x0, x1 = M["scores"].loc[made, NAMES], M["scores"].loc[t0, NAMES]
            d = (x1 - x0).abs().idxmax()
            items.append(_item('Not seen two years earlier.',
                               f' In {made.strftime("%b %Y")} the model called {_tag(call)} '
                               f'({odds[call]:.0%}; {ui.esc(INTEXT[new])} {odds[new]:.0%}), because:',
                               _call_why(made, call, new)
                               + [f"<b>What changed after:</b> {ui.esc(SHORT[d][0].lower())} moved most "
                                  f"by the turn, from {ui.signed(x0[d])} to {ui.signed(x1[d])}"
                                  f"{_moved(d, made, t0)}."]))
    else:
        items.append(_item('No call two years earlier.',
                           ' The data start in 1968.' if HELD else
                           ' The model’s calls start in 1976, once there are past regimes to learn from.'))
    sections.append((f'Turning point, {t0.strftime("%b %Y")}', items))
# This month: what it was living through, with its own events, and the call made in it.
items = []
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
if hx in PRED["probs"].index:
    _odds = PRED["probs"].loc[hx].sort_values(ascending=False)
    _pc = PRED["called"].get(hx)
    _top = ", ".join(f'{ui.esc(INTEXT[k])} {v:.0%}' for k, v in _odds.head(2).items())
    _call = f"{_tag(_pc)} ({_top})" if _pc in COLOR else _top
    _out = (f" It turned out {_tag(r_in)}." + (_chip(r_in == _pc) if _pc in COLOR else "") if r_in in COLOR
            else f" Not known yet: what followed is known up to {going_lab.index[-1].strftime('%b %Y')}.")
    _why = (_call_why(hx, _pc, _odds.drop(_pc).idxmax(), 3, 1) if _pc in COLOR else [])
    items.append(_item("Called for the next three years",
                       f", {_basis(hx)}: {_call}.{_out}" + (" Why:" if _why else ""), _why))
else:
    _odds = R.probabilities(M["scores"].loc[[hx]], REG).iloc[0].sort_values(ascending=False)
    items.append(_item("The model on this month",
                       ", before 1976, so with today’s positions (learned with hindsight, not what it "
                       "would have said then): "
                       + ", ".join(f'{ui.esc(INTEXT[k])} {v:.0%}' for k, v in _odds.head(2).items()) + "."))
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

# The call and how it is made, folded under the two sections above.
with st.expander("The three-year call: odds, landing point, drivers, what-if and track record",
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
                   "then, 67% and 66% (medians 30% and 26%).")
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
