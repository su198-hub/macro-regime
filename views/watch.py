"""Turning points: what would end the current regime, and is it happening?

A first cut for discussion, built after the team asked how the monitor would
catch the next turn. It sits beside the three-year call and does not feed it.

The page is organized around the three ways a high-growth, high-inflation
spell can end. Stagflation has a tested alarm; goldilocks rests on
productivity; a bust has no precedent in the sample and is watched through the
AI build-out and the public finances. The three forces the team named (AI,
fiscal, the world order) are read as what decides between the exits.

Reads the store like every other page. Two series are its own, pulled by
`ingest.py watch`; everything else is already there for the model or the
energy page.
"""

import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import truth, ui
from src import watch as wt
from views.common import data_version, get_store, published_note, refresh_button

RED, GREEN, BLUE, AMBER = "#e34948", "#008300", "#2a78d6", "#eda100"
EXIT_COLOR = {"stagflation": RED, "goldilocks": GREEN, "bust": ui.INK_2, "hard_landing": AMBER}
EXIT_NAME = {"stagflation": "Stagflation", "goldilocks": "Goldilocks", "hard_landing": "Hard landing",
             "bust": "Bust"}

cfg = wt.load_config()
meta = cfg["meta"]
WARN = float(meta["warn"])
WINDOW = int(meta["window_months"])


@st.cache_data(ttl=900, show_spinner=False)
def load(version: str):
    store = get_store()
    wide = store.as_of(wt.required_series(cfg), dt.date.today())
    tr = store.latest(list(truth.STORE_IDS.values()))
    lab = None
    if not tr.empty and set(truth.STORE_IDS.values()) <= set(tr.columns):
        lab = truth.build(truth.from_store(tr), **truth.METHODS["now"])["label"].dropna()
    return wide.astype(float) if not wide.empty else wide, lab


st.html('<h1 class="mr-title">Turning points</h1>'
        '<p class="mr-sub">What would end this regime, and is it happening? A first cut for '
        'discussion. It sits beside the three-year call and does not feed it.</p>')
st.html('<hr class="mr-rule">')
refresh_button()

wide, lab = load(data_version())
if wide.empty or lab is None:
    st.html('<div class="mr-custom"><b>The data for this page is not in the store yet.</b> Pull it on '
            'the desk machine, then publish:<br><code>python ingest.py --db '
            '"%LOCALAPPDATA%\\macro-regime\\macrobond.duckdb" watch</code></div>')
    st.stop()

res = wt.compute(cfg, wide)
gauges, alarm = res["gauges"], res["alarm"].dropna()
absent = sorted({sid for _, comp in wt.components(cfg) for sid in comp["series"]} - set(wide.columns))
if absent or alarm.empty or wt.exits(lab).empty:
    st.html('<div class="mr-custom"><b>The gauges cannot be scored from this store.</b> '
            + (f'Missing series: {ui.esc(", ".join(absent))}.' if absent else
               'It holds too little history to find a past exit.') + '</div>')
    st.stop()
record = wt.track_record(res["alarm"], lab, WARN, lookback=int(meta["lookback_months"]),
                         horizon=int(meta["horizon_months"]))
record_rt = wt.track_record(wt.compute(cfg, wide, realtime=True)["alarm"], lab, WARN,
                            lookback=int(meta["lookback_months"]), horizon=int(meta["horizon_months"]))
ex = wt.exits(lab)
sp = wt.spells(lab)
in_regime = len(sp) and sp.iloc[-1]["label"] == wt.HGHI
G = cfg["gauges"]


def last(s: pd.Series):
    s = s.dropna()
    return (s.index[-1], float(s.iloc[-1])) if len(s) else (None, np.nan)


def when(p) -> str:
    return "" if p is None else pd.Period(p, "M").strftime("%B %Y")


def num(x: float, fmt: str = "+.1f") -> str:
    return "–" if x is None or pd.isna(x) else format(x, fmt).replace("-", "−")


def tile(title: str, value: str, phrase: str, note: str = "", color: str = ui.INK) -> str:
    return (f'<div style="border-top:2px solid {ui.INK};padding-top:.45rem">'
            f'<div style="font-size:.82rem;font-weight:600;color:{ui.INK_2}">{ui.esc(title)}</div>'
            f'<div style="font-family:{ui.HEADING_FONT};font-weight:700;font-size:2rem;'
            f'line-height:1.15;color:{color}">{value}</div>'
            f'<div style="font-size:.9rem;color:{ui.INK}">{phrase}</div>'
            f'<div style="font-size:.78rem;color:{ui.MUTED};margin-top:.15rem">{note}</div></div>')


def zcell(x: float) -> ui.Raw:
    """A score as a tinted cell: red leans stagflation, blue away from it."""
    if x is None or pd.isna(x):
        return ui.Raw(f'<span style="color:{ui.MUTED}">–</span>')
    a = min(abs(x) / 2, 1) * 0.55
    rgb = "227,73,72" if x > 0 else "42,120,214"
    return ui.Raw(f'<span style="display:inline-block;min-width:3.2rem;padding:.1rem .35rem;'
                  f'text-align:right;background:rgba({rgb},{a:.2f})">{num(x)}</span>')


def show(x: float, fmt: str, unit: str) -> str:
    """A reading with its unit: 9.9% a year, +0.67 pp, 48.2 index."""
    return num(x, fmt) + ("" if unit.startswith("%") else " ") + unit


def chip(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;padding:.05rem .45rem;border:1px solid {color};'
            f'color:{color};font-size:.78rem;font-weight:600;white-space:nowrap">{ui.esc(text)}</span>')


# ---------- readings used across the page ----------

month, alarm_now = last(alarm)
g_now = {g: last(gauges[g])[1] for g in gauges.columns}
alarm_word = {"warning": "Warning", "building": "Building", "quiet": "Quiet",
              "not scored": "Not scored"}[wt.reading(alarm_now, WARN)]
alarm_color = RED if alarm_now >= WARN else ui.INK

gcfg = cfg["panels"]["greenspan"]
gs = wt.greenspan(wide, gcfg["series"], float(gcfg["tolerance"]))
gs_now = gs.iloc[-1] if len(gs) else None
met_now = int(gs_now["met"]) if gs_now is not None else None

prod_level = wt.monthly(wide, cfg["panels"]["context"]["series"]["productivity"]).dropna()
prod = wt._ann(prod_level, 24).dropna()
prod_month, prod_now = last(prod)


def era(a: str, b: str | None = None) -> float:
    """Growth in output per hour between two months, % a year, from the index itself."""
    lv = prod_level[a:b] if b else prod_level[a:]
    if len(lv) < 13:
        return np.nan
    years = (lv.index[-1] - lv.index[0]).n / 12
    return float(((lv.iloc[-1] / lv.iloc[0]) ** (1 / years) - 1) * 100)


ai = cfg["panels"]["ai"]["series"]
gdp = wt.monthly(wide, ai["gdp"])
use_now, use_6m = wt.monthly(wide, ai["use_now"]), wt.monthly(wide, ai["use_in_6m"])
ramp_adopt = wt.monthly(wide, ai["ramp_adoption"], carry=0).dropna()
ramp_med = wt.monthly(wide, ai["ramp_pepm_median"], carry=0).dropna()
ramp_top = wt.monthly(wide, ai["ramp_pepm_top10"], carry=0).dropna()
ramp_top1 = wt.monthly(wide, ai["ramp_pepm_top1"], carry=0).dropna()
dc_share = wt.share(wt.monthly(wide, ai["data_centers"]), gdp, 1e8)      # $ millions over $
comp_share = wt.share(wt.monthly(wide, ai["computers"]), gdp, 1e11)      # $ billions over $
capex = wt.monthly(wide, ai["capex_expected"]).abs()
capex_yoy = ((capex / capex.shift(12) - 1) * 100).dropna()
dc_raw = wt.monthly(wide, ai["data_centers"])
dc_yoy = ((dc_raw / dc_raw.shift(12) - 1) * 100).dropna()

fiscal_now = g_now.get("fiscal", np.nan)
fiscal_word = ("Elevated" if fiscal_now >= 1 else "Above normal" if fiscal_now >= WARN
               else "Normal" if not pd.isna(fiscal_now) else "Not scored")


def check_phrase() -> str:
    if gs_now is None:
        return "Not available"
    parts = [("pay growth is " + ("holding" if gs_now["ok_wages"] else "slowing")),
             ("margins are " + ("holding" if gs_now["ok_margin"] else "falling")),
             ("inflation is " + ("steady or falling" if gs_now["ok_core"] else "rising"))]
    return ui.join_words(parts).capitalize()


# ---------- where it stands ----------

st.html(ui.section_head(
    "Where it stands", meta=f"Latest month {when(month)}",
    caption="A high-growth, high-inflation spell can end three ways. Each has its own gauge. "
            "Scores are in standard deviations from each series' own history."))
cols = st.columns(3)
cols[0].html(tile(
    "Toward stagflation: the alarm", num(alarm_now, "+.2f"), f"<b>{alarm_word}</b>. Warning line "
    f"{num(WARN, '+.1f')}.",
    "Inflation momentum, policy squeeze and rate-sensitive demand, averaged over three months. "
    "Tested against the six past exits.", alarm_color))
cols[1].html(tile(
    "Toward goldilocks: the productivity check",
    "–" if met_now is None else f"{met_now} of 3", f"{check_phrase()}.",
    f"Output per hour is growing {num(prod_now, '.1f')}% a year over two years. Pay holding, "
    "margins holding and inflation easing together would point to faster productivity.",
    GREEN if met_now == 3 else ui.INK))
cols[2].html(tile(
    "Toward a bust: fiscal pressure", num(fiscal_now, "+.2f"), f"<b>{fiscal_word}</b>.",
    "Interest costs, the deficit and long yields against the Fed. Watched for its rationale: "
    "no past spell ended this way.", ui.INK))

read = []
if alarm_now < 0:
    read.append(f"The pressures that came before each past stagflation exit are absent: the alarm reads "
                f"{num(alarm_now, '+.2f')}, close to its average in quiet months "
                f"({num(record['quiet_mean'], '+.2f')}).")
elif alarm_now < WARN:
    read.append(f"Pressure toward a stagflation exit is building but below the warning line "
                f"({num(alarm_now, '+.2f')} against {num(WARN, '+.1f')}).")
else:
    read.append(f"The alarm is above its warning line ({num(alarm_now, '+.2f')}), as it was before each "
                f"past stagflation exit.")
if met_now is not None:
    read.append("The productivity check holds on all three conditions." if met_now == 3 else
                f"The productivity check holds on {ui.count_word(met_now)} of three conditions, so the "
                f"case for a goldilocks exit is not yet made.")
if not pd.isna(fiscal_now) and fiscal_now >= WARN:
    read.append(f"Fiscal pressure is the reading furthest from normal ({num(fiscal_now, '+.2f')}).")
st.html(f'<div class="m-body" style="margin-top:.9rem"><p>{" ".join(read)}</p></div>')


# ---------- three ways out ----------

st.html(ui.section_head(
    "Three ways out",
    caption="How each exit happens, how often it has happened since 1968, and how far the evidence goes."))


def today_for(exit_id: str) -> ui.Raw:
    if exit_id == "stagflation":
        return ui.Raw(chip(f"Alarm {num(alarm_now, '+.2f')}, {alarm_word.lower()}", alarm_color))
    if exit_id == "goldilocks":
        return ui.Raw(chip("Check not available" if met_now is None else f"Check {met_now} of 3",
                           GREEN if met_now == 3 else ui.INK_2))
    return ui.Raw(chip(f"Fiscal {num(fiscal_now, '+.2f')}, {fiscal_word.lower()}", ui.INK_2))


st.html(ui.table(
    ["Exit", "How it happens", "Seen before", "What we watch", "Evidence", "Today"],
    [[ui.Raw(f'<b style="color:{EXIT_COLOR[k]}">{ui.esc(v["label"])}</b>'), v["how"], v["seen"],
      ui.join_words([g["label"].lower() for g in G.values() if g["exit"] == k])
      + (", the AI build-out" if k == "bust" else ", the productivity check, AI use" if k == "goldilocks" else ""),
      v["evidence"], today_for(k)] for k, v in cfg["exits"].items()]))


# ---------- the stagflation alarm ----------

st.html(ui.section_head(
    "The stagflation alarm",
    caption="The average of three gauges, smoothed over three months. Shaded bands are high-growth, "
            "high-inflation spells; a dot marks the month each ended, colored by what followed."))

hist = alarm[alarm.index >= pd.Period("1968-01", "M")]
line = pd.DataFrame({"month": ui.month_mid(hist.index.to_timestamp()), "alarm": hist.to_numpy()})
bands = sp[sp["label"] == wt.HGHI].copy()
bands["x0"] = [p.to_timestamp() for p in bands["start"]]
bands["x1"] = [(p + 1).to_timestamp() for p in bands["end"]]
dots = pd.DataFrame([{"month": ui.month_mid([r.end.to_timestamp()])[0],
                      "alarm": float(alarm.get(r.end, np.nan)),
                      "next": EXIT_NAME.get(r.next, r.next), "ended": when(r.end)}
                     for r in ex.itertuples()]).dropna()
x = ui.time_x("month", line["month"].min(), line["month"].max())
chart = alt.layer(
    alt.Chart(bands).mark_rect(color=BLUE, opacity=0.10).encode(
        x=alt.X("x0:T", scale=alt.Scale(type="utc")), x2="x1:T"),
    alt.Chart(pd.DataFrame({"v": [0.0]})).mark_rule(color=ui.AXIS).encode(y="v:Q"),
    alt.Chart(pd.DataFrame({"v": [WARN]})).mark_rule(color=RED, strokeDash=[4, 3]).encode(y="v:Q"),
    alt.Chart(pd.DataFrame({"v": [WARN], "t": [f"Warning line {WARN:+.1f}"],
                            "month": [pd.Timestamp("2010-03-15")]})).mark_text(
        align="left", dy=-7, dx=4, color=RED, fontSize=11).encode(x="month:T", y="v:Q", text="t:N"),
    alt.Chart(line).mark_line(color=ui.INK, strokeWidth=1.6).encode(
        x=x, y=alt.Y("alarm:Q", title="Alarm, standard deviations", axis=alt.Axis(format="+.1f")),
        tooltip=[ui.month_tip("month"), alt.Tooltip("alarm:Q", format="+.2f", title="Alarm")]),
    alt.Chart(dots).mark_circle(size=130, stroke="white", strokeWidth=1.5, opacity=1).encode(
        x="month:T", y="alarm:Q",
        color=alt.Color("next:N", scale=alt.Scale(domain=["Stagflation", "Goldilocks"],
                                                  range=[RED, GREEN]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("ended:N", title="Spell ended"), alt.Tooltip("next:N", title="Followed by"),
                 alt.Tooltip("alarm:Q", format="+.2f", title="Alarm")]),
).properties(height=300)
left, right = st.columns([3, 2])
left.altair_chart(ui.style(chart), width="stretch")

pe = record["per_exit"]
stf = pe[pe["next"] == wt.STAGFLATION]
gld = pe[pe["next"] == wt.GOLDILOCKS]
leads = stf["lead"].dropna().astype(int)
rt_leads = record_rt["per_exit"].loc[record_rt["per_exit"]["next"] == wt.STAGFLATION, "lead"].dropna().astype(int)
right.html(
    '<div class="m-body">'
    f'<p><b>What the record shows.</b> The alarm crossed {num(WARN, "+.1f")} before '
    f'{ui.count_word(len(leads))} of {ui.count_word(len(stf))} stagflation exits, between {leads.min()} and '
    f'{leads.max()} months ahead. It stayed below the line before '
    f'{"both" if len(gld) == 2 and gld["lead"].isna().all() else str(int(gld["lead"].isna().sum())) + " of " + str(len(gld))} '
    f'goldilocks exits.</p>'
    f'<p>Of the {record["flagged"]} months it was above the line inside a finished spell, '
    f'{100 * record["then_stagflation"]:.0f}% were followed by a stagflation exit within '
    f'{int(meta["horizon_months"])} months.</p>'
    f'<p><b>Without hindsight.</b> Scoring each month on earlier data only, the lead was {rt_leads.min()} to '
    f'{rt_leads.max()} months and {100 * record_rt["then_stagflation"]:.0f}% of months above the line were '
    f'followed by a stagflation exit.</p>'
    f'<p style="color:{ui.MUTED}">A low reading says no stagflation exit is in sight. It does not time a '
    f'goldilocks exit: the alarm before those read much as it does in any quiet month '
    f'({num(record["quiet_mean"], "+.2f")}).</p></div>')

st.html(ui.section_head(
    "Today against the months before each past exit",
    caption=f"Each gauge averaged over the {ui.count_word(WINDOW)} months before the spell ended. "
            "Red leans toward the stagflation pattern, blue away from it. Productivity and fiscal "
            "pressure are shown for comparison and are outside the alarm."))
be = wt.before_exits(gauges, ex, WINDOW)
order = [g for g in G if g in gauges.columns]
rows = []
for r, a in zip(be.itertuples(), pe.itertuples()):
    rows.append([when(r.end), ui.Raw(f'<b style="color:{EXIT_COLOR.get(r.next, ui.INK)}">'
                                     f'{ui.esc(EXIT_NAME.get(r.next, r.next))}</b>'),
                 *[zcell(getattr(r, g)) for g in order], zcell(a.last6),
                 "–" if pd.isna(a.lead) else f"{int(a.lead)} months"])
rows.append([ui.Raw(f"<b>Today</b>, {when(month)}"), "Open", *[zcell(g_now[g]) for g in order],
             zcell(alarm_now), "–"])
st.html(ui.table(["Spell ended", "Followed by", *[G[g]["short"] for g in order], "Alarm",
                  "First warning"], rows, numeric=set(range(2, 3 + len(order)))))


# ---------- what is in each gauge ----------

st.html(ui.section_head(
    "What is in each gauge",
    caption="Today's reading, the series' own average, and the score. A positive score leans toward "
            "the stagflation pattern, so a component that points the other way is flipped."))
rows = []
for gid, gauge in G.items():
    role = "in the alarm" if gauge["role"] == "alarm" else "watch item, outside the alarm"
    rows.append(ui.Raw(f'<b>{ui.esc(gauge["label"])}</b> &nbsp;{num(g_now.get(gid, np.nan), "+.2f")}'
                       f' &nbsp;<span style="color:{ui.MUTED};font-weight:400">'
                       f'{ui.esc(gauge["question"])} ({role})</span>'))
    for comp in gauge["components"]:
        key = f"{gid}::{comp['id']}"
        if key not in res["values"]:
            rows.append([comp["label"], "–", "–", "–", "Not in the store yet."])
            continue
        v, z = res["values"][key], res["scores"][key].dropna()
        unit = comp.get("units", "")
        fmt = ".1f" if unit != "pp" else "+.2f"
        rows.append([
            ui.Raw(f'{ui.esc(comp["label"])}<br><span style="color:{ui.MUTED};font-size:.8rem">'
                   f'{ui.esc(comp["source"])}, {when(v.index[-1])}</span>'),
            show(float(v.iloc[-1]), fmt, unit), show(res["norms"][key]["center"], fmt, unit),
            zcell(float(z.iloc[-1])), comp["why"]])
st.html(ui.table(["Component", "Today", "Average", "Score", "Why it is here"], rows, numeric={1, 2, 3}))


# ---------- productivity ----------

st.html(ui.section_head(
    "Productivity: the goldilocks exit",
    caption="Growth is hours worked plus output per hour. With unemployment low and the labor force "
            "growing slowly, output per hour is the engine left, and it also has to bring inflation "
            "down."))
left, right = st.columns([3, 2])
pl = prod[prod.index >= pd.Period("1970-01", "M")]
pdf = pd.DataFrame({"month": ui.month_mid(pl.index.to_timestamp()), "pace": pl.to_numpy()})
ERAS = [("1995-12", "2004-12", "1996 to 2004"), ("2004-12", "2019-12", "2005 to 2019"),
        ("2022-12", None, "since 2023")]
eras = pd.DataFrame([
    {"x0": pd.Period(a, "M").to_timestamp(), "x1": (pd.Period(b, "M") if b else pl.index[-1]).to_timestamp(),
     "avg": era(a, b), "t": t} for a, b, t in ERAS])
pchart = alt.layer(
    alt.Chart(pd.DataFrame({"v": [0.0]})).mark_rule(color=ui.AXIS).encode(y="v:Q"),
    alt.Chart(pdf).mark_line(color=ui.INK, strokeWidth=1.6).encode(
        x=ui.time_x("month", pdf["month"].min(), pdf["month"].max()),
        y=alt.Y("pace:Q", title="Output per hour, % a year over two years"),
        tooltip=[ui.month_tip("month"), alt.Tooltip("pace:Q", format=".1f", title="% a year")]),
    alt.Chart(eras).mark_rule(color=GREEN, strokeWidth=2.5).encode(
        x=alt.X("x0:T", scale=alt.Scale(type="utc")), x2="x1:T", y="avg:Q"),
).properties(height=280)
left.altair_chart(ui.style(pchart), width="stretch")
left.html(f'<p class="mr-caption" style="color:{GREEN}">Green lines are the average pace in each period: '
          + ", ".join(f"{t} {a:.1f}%" for t, a in zip(eras["t"], eras["avg"])) + " a year.</p>")

if gs_now is not None:
    def mark(ok: bool) -> ui.Raw:
        return ui.Raw(f'<b style="color:{GREEN if ok else RED}">{"Holds" if ok else "Fails"}</b>')

    check_rows = [
        ["Pay growth", f"{gs_now['wages']:.1f}%", num(gs_now["d_wages"], "+.1f") + " pp",
         "Steady or rising", mark(bool(gs_now["ok_wages"]))],
        ["Profit margin", f"{gs_now['margin']:.1f}%", num(gs_now["d_margin"], "+.1f") + " pp",
         "Steady or rising", mark(bool(gs_now["ok_margin"]))],
        ["Core inflation", f"{gs_now['core']:.1f}%", num(gs_now["d_core"], "+.1f") + " pp",
         "Steady or falling", mark(bool(gs_now["ok_core"]))],
    ]
    all3 = gs["met"] == 3
    boom = all3[all3.index.year.isin(range(1996, 2001))]
    fwd = prod.shift(-24).reindex(gs.index)
    right.html(
        '<div class="m-body"><p><b>The productivity check.</b> In 1996 Greenspan read faster productivity '
        'from three facts together, before the official numbers showed it. Pay was rising, margins '
        'were holding and inflation was steady, which needs output per hour to be growing faster '
        f'than reported.</p></div>'
        + ui.table(["", "Now", "Over a year", "Needs to be", ""], check_rows, numeric={1, 2})
        + f'<div class="m-body"><p style="font-size:.86rem;color:{ui.INK_2}">All three held in '
          f'{100 * boom.mean():.0f}% of months from 1996 to 2000, against {100 * all3.mean():.0f}% of '
          f'all months since {gs.index[0].year}. It is a consistency check with no tested forecasting '
          f'power: output per hour grew {fwd[all3].mean():.1f}% a year over the two years after months '
          f'when all three held, and {fwd[~all3].mean():.1f}% after the rest.</p>'
          f'<p style="font-size:.8rem;color:{ui.MUTED}">Pay is average hourly earnings of production '
          f'workers (BLS). The margin is pre-tax profit per unit of output at nonfinancial corporations, '
          f'as a share of the price (BEA), latest {when(wt.monthly(wide, gcfg["series"]["unit_profit"]).dropna().index[-1])}. '
          f'A move under {gcfg["tolerance"]} pp reads as steady.</p></div>')
else:
    right.html('<div class="mr-custom">The productivity check needs unit profits and prices from '
               '<code>ingest.py watch</code>.</div>')


# ---------- AI ----------

st.html(ui.section_head(
    "AI: is the build-out turning into use?",
    caption="Investment comes first and shows up at once. The lasting gain needs firms across the "
            "economy to use the tools, which for computers took several years after the spending."))
st.html('<p class="mr-caption"><b>Use.</b> Two sources that measure different things: the Census '
        'Bureau asks a sample of all US firms whether they used AI; Ramp sees what firms on its '
        'payment platform actually pay for AI.</p>')
cols = st.columns(4)
if not use_now.empty:
    first_m, first_v = use_now.dropna().index[0], float(use_now.dropna().iloc[0])
    cols[0].html(tile("Firms using AI", f"{use_now.dropna().iloc[-1]:.0f}%",
                      f"Up from {first_v:.0f}% in {when(first_m)}.",
                      "Share of US firms that used AI in the last two weeks. Census Bureau survey; "
                      "the question was reworded in late 2025, so earlier readings are left out."))
    if not use_6m.empty:
        cols[1].html(tile("Expect to use AI in six months", f"{use_6m.dropna().iloc[-1]:.0f}%",
                          f"{use_6m.dropna().iloc[-1] - use_now.dropna().iloc[-1]:+.0f} pp above "
                          f"today's use.".replace("-", "−"),
                          "The same survey. A gap that keeps closing upward means adoption is spreading."))
else:
    cols[0].html(tile("Firms using AI", "–", "Not in the store yet.", "Pulled by ingest.py watch."))
if not ramp_adopt.empty:
    year_ago = ramp_adopt.get(ramp_adopt.index[-1] - 12, np.nan)
    cols[2].html(tile("Firms paying for AI", f"{ramp_adopt.iloc[-1]:.0f}%",
                      f"Up from {year_ago:.0f}% a year earlier.",
                      f"Ramp AI Index, {when(ramp_adopt.index[-1])}: share of the 85,000+ US firms on "
                      "Ramp with any AI transaction that month. Ramp's customers lean to smaller, "
                      "tech-heavy firms, so it runs above the Census figure."))
if not ramp_med.empty:
    jan = ramp_med.get(pd.Period(f"{ramp_med.index[-1].year}-01", "M"), np.nan)
    cols[3].html(tile("AI spend per employee", f"${ramp_med.iloc[-1]:.0f}",
                      f"A month, median firm. ${jan:.0f} in January.",
                      f"Ramp AI Index, {when(ramp_med.index[-1])}, firms paying for AI only. "
                      "Intensity of use, where the share of firms measures breadth."))

if not ramp_med.empty and not ramp_top.empty:
    rl = pd.concat([
        pd.DataFrame({"month": ui.month_mid(ramp_med.index.to_timestamp()), "usd": ramp_med.to_numpy(),
                      "series": "Median firm"}),
        pd.DataFrame({"month": ui.month_mid(ramp_top.index.to_timestamp()), "usd": ramp_top.to_numpy(),
                      "series": "Median of the top 10%"}),
        pd.DataFrame({"month": ui.month_mid(ramp_top1.index.to_timestamp()), "usd": ramp_top1.to_numpy(),
                      "series": "Median of the top 1%"})])
    first = ramp_med.index[0]
    yoy_med = (ramp_med.iloc[-1] / ramp_med.get(ramp_med.index[-1] - 12, np.nan) - 1) * 100
    yoy_top = (ramp_top.iloc[-1] / ramp_top.get(ramp_top.index[-1] - 12, np.nan) - 1) * 100
    yoy_top1 = (ramp_top1.iloc[-1] / ramp_top1.get(ramp_top1.index[-1] - 12, np.nan) - 1) * 100         if not ramp_top1.empty else np.nan
    # Three panels, each on its own linear axis: on one axis the median ($16) would sit flat
    # against the top 1% ($10,000).
    colors = {"Median firm": ui.INK, "Median of the top 10%": BLUE, "Median of the top 1%": "#c98700"}
    for col, (name, color) in zip(st.columns(3), colors.items()):
        part = rl[rl["series"] == name]
        if part.empty:
            continue
        col.altair_chart(ui.style(alt.Chart(part).mark_line(
            strokeWidth=1.8, color=color, point=alt.OverlayMarkDef(size=18, color=color)).encode(
            x=ui.time_x("month", rl["month"].min(), rl["month"].max(), every_years=1),
            y=alt.Y("usd:Q", title=None, scale=alt.Scale(zero=True), axis=alt.Axis(format="$,.0f")),
            tooltip=[ui.month_tip("month"), alt.Tooltip("usd:Q", format="$,.2f", title="Per employee")],
        ).properties(height=200, title=alt.TitleParams(
            f"{name}: ${part['usd'].iloc[-1]:,.0f}", anchor="start", fontSize=13, color=ui.INK,
            subtitle="AI spend per employee per month, US$", subtitleColor=ui.MUTED, subtitleFontSize=11))),
            width="stretch")
    # The same three lines on one log axis, where equal slopes mean equal growth rates.
    left, right = st.columns([3, 2])
    left.altair_chart(ui.style(alt.Chart(rl).mark_line(strokeWidth=1.8, point=alt.OverlayMarkDef(size=18)).encode(
        x=ui.time_x("month", rl["month"].min(), rl["month"].max(), every_years=1),
        y=alt.Y("usd:Q", title="US$ per employee per month (log scale)",
                scale=alt.Scale(type="log", domain=[1, 20000]),
                axis=alt.Axis(format="$,.0f", values=[1, 10, 100, 1000, 10000])),
        color=alt.Color("series:N", scale=alt.Scale(domain=list(colors), range=list(colors.values()))),
        tooltip=[ui.month_tip("month"), "series:N", alt.Tooltip("usd:Q", format="$,.2f", title="Per employee")],
    ).properties(height=260, title=alt.TitleParams(
        "All three on a log scale", anchor="start", fontSize=13, color=ui.INK,
        subtitle="Equal slopes mean equal growth rates", subtitleColor=ui.MUTED, subtitleFontSize=11))),
        width="stretch")
    right.html(
        '<div class="m-body">'
        f'<p><b>Spending per head is accelerating.</b> The median firm paying for AI spends '
        f'${ramp_med.iloc[-1]:.2f} per employee a month, up {yoy_med:.0f}% on a year earlier and from '
        f'${ramp_med.iloc[0]:.2f} in {when(first)}. The top 10% spend ${ramp_top.iloc[-1]:,.0f}, up '
        f'{yoy_top:.0f}%, and the top 1% ${ramp_top1.iloc[-1]:,.0f}, up {yoy_top1:.0f}%.</p>'
        f'<p>Use is broad but still shallow for most firms: about ${12 * ramp_med.iloc[-1]:.0f} a year per '
        f'employee at the median, against about {ramp_top.iloc[-1] / ramp_med.iloc[-1]:.0f} times that in the top 10%. A productivity '
        'gain across the economy needs the median to keep closing that gap.</p>'
        f'<p style="color:{ui.MUTED}">Ramp AI Index, from corporate card and bill pay transactions of '
        'firms on Ramp, matched to their headcount. The three small panels each have their own linear scale; the chart beside this text puts all three on one log scale. '
        'History from July 2023 only, so shown as context and not scored.</p></div>')

st.html('<p class="mr-caption" style="margin-top:1rem"><b>Build-out.</b></p>')
cols = st.columns(4)
if not dc_share.empty:
    cols[0].html(tile("Data-center construction", f"{dc_share.iloc[-1]:.2f}% of GDP",
                      f"{num(float(dc_yoy.iloc[-1]), '+.0f')}% on a year earlier.",
                      f"Census Bureau, {when(dc_share.index[-1])}. It was "
                      f"{dc_share.get(pd.Period('2019-12', 'M'), np.nan):.2f}% at the end of 2019."))
if not capex_yoy.empty:
    cols[1].html(tile("Expected capital spending", f"{num(float(capex_yoy.iloc[-1]), '+.0f')}%",
                      "S&P 500, analysts' forecast three years out, on a year earlier.",
                      f"Bloomberg, {when(capex_yoy.index[-1])}. A turn down here would be the first "
                      "sign of a capex bust."))

if not comp_share.empty:
    cs = comp_share[comp_share.index >= pd.Period("1980-01", "M")]
    long = pd.concat([
        pd.DataFrame({"month": ui.month_mid(cs.index.to_timestamp()), "share": cs.to_numpy(),
                      "series": "Computers and servers"}),
        pd.DataFrame({"month": ui.month_mid(dc_share.index.to_timestamp()), "share": dc_share.to_numpy(),
                      "series": "Data-center construction"})])
    it_peak = cs[cs.index < pd.Period("2005-01", "M")]
    left, right = st.columns([3, 2])
    left.altair_chart(ui.style(alt.Chart(long).mark_line(strokeWidth=1.8).encode(
        x=ui.time_x("month", long["month"].min(), long["month"].max()),
        y=alt.Y("share:Q", title="Investment, % of GDP"),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Computers and servers",
                                                            "Data-center construction"],
                                                    range=[ui.INK, BLUE])),
        tooltip=[ui.month_tip("month"), "series:N", alt.Tooltip("share:Q", format=".2f", title="% of GDP")],
    ).properties(height=260)), width="stretch")
    right.html(
        '<div class="m-body">'
        f'<p><b>Against the IT boom.</b> Investment in computers and servers is {cs.iloc[-1]:.2f}% of '
        f'GDP. Its peak in the IT boom was {it_peak.max():.2f}% in {it_peak.idxmax().year}.</p>'
        '<p>In the 1990s the spending came first and the productivity gain followed: output per hour '
        f'grew {era("1995-12", "2004-12"):.1f}% a year from 1996 to 2004, after '
        f'{era("1985-12", "1995-12"):.1f}% in the ten years before. The gain reached the firms using '
        'computers, such as retail and wholesale, several years after the firms making them.</p>'
        f'<p style="color:{ui.MUTED}">Spending shows that capital per worker is rising. Whether the '
        'same workers and machines produce more is what the productivity check and AI use are there '
        'to pick up.</p></div>')


# ---------- the three forces ----------

st.html(ui.section_head(
    "The three forces",
    caption="AI, the public finances and the world order are read here as what decides between the "
            "exits. Each cell is the path from that force to that exit."))
st.html(ui.table(
    ["Force", "Toward stagflation", "Toward goldilocks", "Toward a bust", "What we watch"],
    [[ui.Raw(f"<b>{ui.esc(t['label'])}</b>"), t["stagflation"], t["goldilocks"], t["bust"], t["watch"]]
     for t in cfg["themes"]]))
ctx = cfg["panels"]["context"]["series"]
notes = []
for key, name in (("geopolitical_risk", "the geopolitical risk index"),
                  ("supply_chains", "supply-chain pressure")):
    s = wt.monthly(wide, ctx[key]).dropna()
    if len(s) > 60:
        z = (s.iloc[-1] - s.mean()) / s.std()
        notes.append(f"{name} is {num(float(z), '+.1f')} standard deviations from its average "
                     f"({when(s.index[-1])})")
if notes:
    st.html(f'<p class="mr-caption">World order, as context: {ui.join_words(notes)}. Neither is scored.</p>')


# ---------- scenarios ----------

st.html(ui.section_head(
    "Scenarios",
    caption="Four ways the current spell could end, what would show up first on this page, and where "
            "each reading is today. The stories are judgement; the readings are computed."))


def scenario_today(sid: str) -> str:
    if sid == "ai_delivers":
        use = "" if use_now.empty else f" AI use is {use_now.dropna().iloc[-1]:.0f}% of firms and rising."
        if not ramp_med.empty:
            use += f" Median AI spend is ${ramp_med.iloc[-1]:.0f} per employee a month."
        return ("The productivity check is not available." if met_now is None else
                f"The check holds on {met_now} of 3. Output per hour is growing "
                f"{num(prod_now, '.1f')}% a year.") + use
    if sid == "inflation_returns":
        return (f"The alarm is {num(alarm_now, '+.2f')}, {alarm_word.lower()}. Inflation momentum "
                f"{num(g_now.get('inflation_momentum', np.nan), '+.2f')}, policy squeeze "
                f"{num(g_now.get('policy_squeeze', np.nan), '+.2f')}.")
    if sid == "capex_bust":
        bits = []
        if not capex_yoy.empty:
            bits.append(f"Expected capital spending is {num(float(capex_yoy.iloc[-1]), '+.0f')}% on a year earlier")
        if not dc_yoy.empty:
            bits.append(f"data-center construction {num(float(dc_yoy.iloc[-1]), '+.0f')}%")
        if not bits:
            return "Not in the store yet."
        return (ui.join_words(bits) + ". No sign of a turn.") if min(
            float(capex_yoy.iloc[-1]) if not capex_yoy.empty else 1,
            float(dc_yoy.iloc[-1]) if not dc_yoy.empty else 1) > 0 else ui.join_words(bits) + "."
    return f"Fiscal pressure is {num(fiscal_now, '+.2f')}, {fiscal_word.lower()}."


st.html(ui.table(
    ["Scenario", "Ends in", "The story", "First signs here", "Precedent", "Today"],
    [[ui.Raw(f"<b>{ui.esc(s['label'])}</b>"),
      ui.Raw(f'<b style="color:{EXIT_COLOR[s["exit"]]}">{ui.esc(cfg["exits"][s["exit"]]["label"])}</b>'),
      s["story"], s["first_signs"], s["precedent"], scenario_today(s["id"])] for s in cfg["scenarios"]]))


# ---------- limits ----------

st.html(ui.section_head("What this can and cannot do"))
shock = stf.loc[[p.year in (1973, 2022) for p in stf["end"]], "lead"].dropna().astype(int)
st.html(
    '<div class="m-body"><ul>'
    f'<li><b>Six past exits.</b> The gauges were chosen after looking at those same six, so the record '
    f'above flatters them. The fair claim is that the alarm is consistent with history.</li>'
    f'<li><b>Shocks give little warning.</b> The exits of 1973 and 2022 followed oil and war, and the '
    f'alarm led by {shock.min()} to {shock.max()} months. Where pressure built slowly it led by more '
    f'than a year.</li>'
    '<li><b>It does not time a goldilocks exit.</b> A low alarm rules out one path. The productivity '
    'check is the only reading that speaks to the other, and it has no tested forecasting power.</li>'
    '<li><b>A bust is untested.</b> No spell in the sample ended in one. Fiscal pressure read lower '
    'before stagflation exits than before goldilocks ones, so it is here for its rationale alone.</li>'
    f'<li><b>The warning line is a judgement.</b> {num(WARN, "+.1f")} was set by eye from the past '
    'exits and was not fitted. Scores use each series\' full history; the record is also shown with '
    'scores that use earlier data only.</li>'
    '<li><b>Nothing here feeds the regime model.</b> The three-year call, its drivers and its '
    'backtest are unchanged.</li></ul></div>')
st.html(f'<p class="mr-foot">{published_note()}</p>')
