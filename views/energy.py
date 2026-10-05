"""The energy page: is energy demand outrunning supply?

Work in progress, in the menu as "Energy (WIP)". Built on the manager's request to track the
energy scenario, with ASR's two markers as the frame: whether demand surges,
and whether supply expands to meet it. Scored on its own; the regime model is
untouched. The proposal is that the one number at the top -- energy tightness
-- joins the supply driver, through the twin first.

The page reads the store like every other page, so it works on the hosted app
once `ingest.py energy` has been run and the store published. Every series on
it is public.
"""

import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import energy as en
from src import ui
from views.common import get_store

SCENARIO_COLOR = {"ai_boom": "#2a78d6", "energy_first": "#e34948", "current_policies": "#8c8c8c"}
TIGHT, LOOSE = "#c0392b", "#2a78d6"

cfg = en.load_config()
store = get_store()
wide = store.as_of(en.required_series(cfg), dt.date.today())

st.html('<h1 class="mr-title">Energy</h1>'
        '<p class="mr-sub">Is energy demand outrunning supply? A first cut for discussion. '
        'Not part of the regime model yet.</p>')
st.html('<hr class="mr-rule">')

if wide.empty:
    st.html('<div class="mr-custom"><b>No energy data in this store yet.</b> Pull it on the '
            'desk machine, then publish:<br><code>python ingest.py --db '
            '"%LOCALAPPDATA%\\macro-regime\\macrobond.duckdb" energy</code></div>')
    st.stop()

for c in wide.columns:
    wide[c] = wide[c].astype(float)
res = en.compute(cfg, wide)
blocks, comp, scen = res["blocks"], res["composite"], res["scenario"].dropna()
mid = float(cfg["meta"].get("mid_band", 0.15))


def read(x: float, tight="tighter than usual", loose="looser than usual") -> str:
    if pd.isna(x):
        return "not scored"
    return "close to normal" if abs(x) <= mid else (tight if x > 0 else loose)


def last(s: pd.Series):
    s = s.dropna()
    return (s.index[-1], float(s.iloc[-1])) if len(s) else (None, np.nan)


def why_html(why) -> str:
    """One or two short paragraphs under an indicator."""
    parts = why if isinstance(why, list) else [why]
    return "".join(f'<p style="margin:.2rem 0 0;font-size:.8rem;color:{ui.MUTED};'
                   f'max-width:34rem">{ui.esc(x)}</p>' for x in parts)


def tile(title: str, value: float, phrase: str, note: str = "") -> str:
    color = ui.INK if pd.isna(value) or abs(value) <= mid else (TIGHT if value > 0 else LOOSE)
    num = "–" if pd.isna(value) else ui.signed(value)
    return (f'<div style="border-top:2px solid {ui.INK};padding-top:.45rem">'
            f'<div style="font-size:.82rem;font-weight:600;'
            f'color:{ui.INK_2}">{ui.esc(title)}</div>'
            f'<div style="font-family:{ui.HEADING_FONT};font-weight:700;font-size:2rem;'
            f'line-height:1.15;color:{color}">{num}</div>'
            f'<div style="font-size:.86rem;color:{ui.INK}">{ui.esc(phrase)}</div>'
            f'<div style="font-size:.78rem;color:{ui.MUTED};margin-top:.15rem">{note}</div></div>')


# ---------- headline ----------

when, energy_now = last(comp.get("energy", pd.Series(dtype=float)))
_, power_now = last(comp.get("power", pd.Series(dtype=float)))
_, fuels_now = last(comp.get("fuels", pd.Series(dtype=float)))
scen_now = scen.iloc[-1] if len(scen) else None
scen_label, scen_text = en.SCENARIOS.get(scen_now, ("–", ""))
asof = f"{when:%B %Y}" if when is not None else "–"

st.html(ui.section_head("Where it stands", meta=f"Latest month {asof}",
                        caption="Scores run from −1 to +1. Positive means tighter energy than each "
                                "series' average to 2019, against how far it has ranged since."))
cols = st.columns(4)
cols[0].html(tile("Energy tightness", energy_now, read(energy_now),
                  "Power and fuels, equal weight. Proposed to replace the energy majors' capex "
                  "revision in the supply driver."))
cols[1].html(tile("Power", power_now, read(power_now), "Demand, less supply expanding, plus prices."))
cols[2].html(tile("Oil and gas", fuels_now, read(fuels_now), "Spare capacity, inventories, gas price."))
cols[3].html(
    f'<div style="border-top:2px solid {ui.INK};padding-top:.45rem">'
    f'<div style="font-size:.82rem;font-weight:600;color:{ui.INK_2}">'
    f'Reads like ASR\'s</div>'
    f'<div style="font-family:{ui.HEADING_FONT};font-weight:700;font-size:2rem;line-height:1.15;'
    f'color:{SCENARIO_COLOR.get(scen_now, ui.INK)}">{ui.esc(scen_label)}</div>'
    f'<div style="font-size:.86rem;color:{ui.INK}">{ui.esc(scen_text)}</div>'
    f'<div style="font-size:.78rem;color:{ui.MUTED};margin-top:.15rem">From the two markers below. '
    f'Net Zero and High Damage are not read from them.</div></div>')


# ---------- the two markers ----------

st.html(ui.section_head(
    "The two markers",
    caption="ASR separates AI Boom from Energy First by whether power demand surges and "
            "whether supply expands to meet it. Each dot is a month; the line is the last "
            "three years, the large dot the latest. The shading is the rule the label uses."))


def regions(mid: float) -> pd.DataFrame:
    """The four labelled regions, as bands between a lower and an upper edge.

    Drawn from the same rule as en.scenario_for, so the picture cannot disagree
    with the label: above the demand line, supply within `mid` of demand is AI
    Boom and further left is Energy First; below it, supply shrinking past
    `mid` is Energy First and the rest Current Policies.
    """
    rows = []
    for name, pts_ in {
        "Energy First": [(-1, mid, 1), (0, mid, 1), (1 - mid, 1, 1)],
        "AI Boom": [(0, mid, mid), (1 - mid, mid, 1), (1, mid, 1)],
        "Energy First (supply-led)": [(-1, -1, mid), (-mid, -1, mid)],
        "Current Policies": [(-mid, -1, mid), (1, -1, mid)],
    }.items():
        rows += [{"region": name, "x": x, "lo": lo, "hi": hi} for x, lo, hi in pts_]
    return pd.DataFrame(rows)


REGION_COLOR = {"AI Boom": SCENARIO_COLOR["ai_boom"], "Energy First": SCENARIO_COLOR["energy_first"],
                "Energy First (supply-led)": SCENARIO_COLOR["energy_first"],
                "Current Policies": SCENARIO_COLOR["current_policies"]}

if {"power_demand", "power_supply"} <= set(blocks.columns):
    pts = blocks[["power_demand", "power_supply"]].dropna().copy()
    pts["month"] = ui.month_mid(pts.index)
    pts["scenario"] = scen.reindex(pts.index).map(lambda k: en.SCENARIOS.get(k, ("",))[0])
    recent = pts[pts.index >= pts.index.max() - pd.DateOffset(months=36)]
    latest = recent.tail(1).assign(label=lambda d: d.index.strftime("%b %Y"))
    dom = alt.Scale(domain=[-1.05, 1.05], nice=False)
    ticks = [-1, -0.5, 0, 0.5, 1]
    labels = pd.DataFrame([
        {"x": -0.5, "y": 0.72, "t": "Energy First"},
        {"x": 0.72, "y": 0.42, "t": "AI Boom"},
        {"x": -0.58, "y": -0.55, "t": "Energy First"},
        {"x": -0.58, "y": -0.68, "t": "(supply-led)"},
        {"x": 0.45, "y": -0.6, "t": "Current Policies"},
    ])
    region_scale = alt.Scale(domain=list(REGION_COLOR), range=list(REGION_COLOR.values()))
    chart = alt.layer(
        alt.Chart(regions(mid)).mark_area(opacity=0.10, clip=True).encode(
            x=alt.X("x:Q", scale=dom, title="Supply expanding →",
                    axis=alt.Axis(values=ticks, format="+.1f", grid=False)),
            y=alt.Y("lo:Q", scale=dom, title="Demand surging →",
                    axis=alt.Axis(values=ticks, format="+.1f", grid=False)),
            y2="hi:Q",
            color=alt.Color("region:N", scale=region_scale, legend=None),
            detail="region:N"),
        alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=ui.AXIS).encode(x="v:Q"),
        alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=ui.AXIS).encode(y="v:Q"),
        alt.Chart(pts).mark_circle(size=22, color=ui.MUTED, opacity=0.45, clip=True).encode(
            x="power_supply:Q", y="power_demand:Q",
            tooltip=[ui.month_tip("month"), alt.Tooltip("power_demand:Q", format="+.2f", title="Demand"),
                     alt.Tooltip("power_supply:Q", format="+.2f", title="Supply"),
                     alt.Tooltip("scenario:N", title="Reads like")]),
        alt.Chart(recent).mark_line(color=ui.INK, strokeWidth=1.4, clip=True).encode(
            x="power_supply:Q", y="power_demand:Q", order="month:T"),
        alt.Chart(latest).mark_circle(size=180, color=SCENARIO_COLOR.get(scen_now, ui.INK),
                                      stroke="white", strokeWidth=1.5, opacity=1).encode(
            x="power_supply:Q", y="power_demand:Q"),
        alt.Chart(latest).mark_text(dy=-16, fontSize=11, color=ui.INK).encode(
            x="power_supply:Q", y="power_demand:Q", text="label:N"),
        alt.Chart(labels).mark_text(color=ui.INK_2, fontSize=12, fontWeight="bold").encode(
            x="x:Q", y="y:Q", text="t:N"),
    ).properties(height=420)
    left, right = st.columns([3, 2])
    left.altair_chart(ui.style(chart), width="stretch")
    first = pts.index.min()
    right.html(
        '<div class="m-body">'
        f'<p><b>Demand</b> is {read(pts["power_demand"].iloc[-1], "surging", "weak")} '
        f'({ui.signed(pts["power_demand"].iloc[-1])}) and <b>supply</b> is '
        f'{read(pts["power_supply"].iloc[-1], "expanding", "contracting")} '
        f'({ui.signed(pts["power_supply"].iloc[-1])}).</p>'
        '<p><b>How the label is read.</b> When demand is surging, supply within 0.15 of it '
        'is keeping pace, which is AI Boom; further behind is Energy First. Without a surge, '
        'shrinking supply is Energy First on its own, and anything else Current Policies.</p>'
        f'<p style="color:{ui.MUTED}">Both markers exist from {first:%B %Y}, when the planned-'
        'capacity history starts. One power-demand boom has happened since, so the picture '
        'cannot be validated by a backtest; it rests on what each indicator measures.</p></div>')


# ---------- history ----------

st.html(ui.section_head("Over time", caption="The three composite scores, monthly."))
hist = comp.dropna(how="all").copy()
if not hist.empty:
    long = hist.rename(columns={"energy": "Energy tightness", "power": "Power", "fuels": "Oil and gas"})
    long.index = ui.month_mid(long.index)
    long = long.reset_index(names="month").melt("month", var_name="series", value_name="score").dropna()
    chart = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=ui.time_x("month", long["month"].min(), long["month"].max()),
        y=alt.Y("score:Q", scale=alt.Scale(domain=[-1, 1]), title=None),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["Energy tightness", "Power", "Oil and gas"], range=[ui.INK, "#2a78d6", "#c98700"])),
        strokeWidth=alt.condition(alt.datum.series == "Energy tightness", alt.value(2.6), alt.value(1.4)),
        tooltip=[ui.month_tip("month"), "series:N", alt.Tooltip("score:Q", format="+.2f")],
    ) + alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=ui.AXIS).encode(y="v:Q")
    st.altair_chart(ui.style(chart.properties(height=260)), width="stretch")


# ---------- the indicators, block by block ----------

def spark(values: pd.Series, units: str) -> alt.Chart:
    d = values.dropna().to_frame("v")
    d["month"] = ui.month_mid(d.index)
    return alt.Chart(d).mark_line(color=ui.INK, strokeWidth=1.3).encode(
        x=ui.time_x("month", d["month"].min(), d["month"].max()),
        y=alt.Y("v:Q", title=units, scale=alt.Scale(zero=False)),
        tooltip=[ui.month_tip("month"), alt.Tooltip("v:Q", format=",.2f", title=units)],
    ).properties(height=150)


for block_id, block in cfg["blocks"].items():
    when_b, val_b = last(blocks.get(block_id, pd.Series(dtype=float)))
    meta = f"Block score {ui.signed(val_b)}" if not pd.isna(val_b) else "Not scored"
    st.html(ui.section_head(f"{block['label']}: {block['question']}", meta=meta))
    rows, charts = [], []
    for ind in block["indicators"]:
        key = f"{block_id}::{ind['id']}"
        v, z = res["values"].get(key), res["scores"].get(key)
        if v is None or v.dropna().empty:
            rows.append([ind["label"], "–", "–", "–", "No data yet",
                         ui.Raw(f'<a href="{ind["link"]}" target="_blank">{ui.esc(ind["source"])}</a>')])
            continue
        t, x = last(v)
        _, zx = last(z)
        rows.append([
            ui.Raw(f'{ui.esc(ind["label"])}{why_html(ind["why"])}'),
            f"{x:,.2f} {ind['units']}", f"{t:%b %Y}", ui.signed(zx),
            f"From {v.dropna().index.min():%Y}; scored against {res['norms'][key]['basis']}",
            ui.Raw(f'<a href="{ind["link"]}" target="_blank">{ui.esc(ind["source"])}</a>'),
        ])
        charts.append((ind["label"], spark(v, ind["units"])))
    st.html(ui.table(["Indicator", "Latest", "As of", "Score", "History", "Source"], rows,
                     numeric={1, 3}))
    if charts:
        grid = st.columns(min(3, len(charts)))
        for i, (label, ch) in enumerate(charts):
            grid[i % len(grid)].caption(label)
            grid[i % len(grid)].altair_chart(ui.style(ch), width="stretch")


# ---------- context, not scored ----------

st.html(ui.section_head(
    "Context, not scored",
    caption="Direct readings on AI load that cannot yet be scored: too short, no history of "
            "earlier estimates, or published by hand twice a year."))

for c in cfg.get("context", []):
    s = wide.get(c["series"])
    if s is None or s.dropna().empty:
        continue
    d = s.dropna().to_frame("gw")
    d["month"] = ui.month_mid(d.index)
    today = pd.Timestamp.today()
    d["part"] = np.where(d.index <= today, "Built or under way", "Planned")
    now_gw = float(s[s.index <= today].dropna().iloc[-1])
    ahead = {y: float(s[s.index <= pd.Timestamp(f"{y}-12-31")].dropna().iloc[-1])
             for y in (today.year + 1, today.year + 2) if (s.index >= pd.Timestamp(f"{y}-12-01")).any()}
    chart = alt.Chart(d).mark_area(opacity=0.85).encode(
        x=ui.time_x("month", d["month"].min(), d["month"].max()),
        y=alt.Y("gw:Q", title="GW"),
        color=alt.Color("part:N", scale=alt.Scale(domain=["Built or under way", "Planned"],
                                                  range=[ui.INK, "#c3ccd6"])),
        tooltip=[ui.month_tip("month"), alt.Tooltip("gw:Q", format=".1f", title="GW")],
    ) + alt.Chart(pd.DataFrame({"t": [ui.month_mid([today])[0]]})).mark_rule(
        color=TIGHT, strokeDash=[4, 3]).encode(x="t:T")
    left, right = st.columns([3, 2])
    left.caption(c["label"])
    left.altair_chart(ui.style(chart.properties(height=220)), width="stretch")
    ahead_txt = "; ".join(f"{v:.1f} GW by the end of {y}" for y, v in ahead.items())
    right.html(
        f'<div class="m-body"><p><b>{now_gw:.1f} GW</b> today across the sites Epoch tracks; '
        f'on current plans {ahead_txt}.</p>{why_html(c["why"])}'
        f'<p><a href="{c["link"]}" target="_blank">{ui.esc(c["source"])}</a></p></div>')

manual = cfg.get("manual", [])
if manual:
    st.html(ui.table(["Reading", "Latest", "As of", "Source"], [
        [m["label"], m["value"], m["as_of"],
         ui.Raw(f'<a href="{m["link"]}" target="_blank">{ui.esc(m["source"])}</a>')]
        for m in manual]))


# ---------- what is missing ----------

st.html(ui.section_head("Not in yet", caption="Proposed, and what stands in the way."))
st.html(ui.table(["Indicator", "Blocker"], [[p["item"], p["blocker"]] for p in cfg.get("pending", [])]))
st.html(f'<p class="mr-source">Sources: EIA Short-Term Energy Outlook (every archived release since '
        f'2009) and Form EIA-860M (quarterly inventories since July 2015); BLS producer prices '
        f'and IMF copper via FRED; Epoch AI Frontier Data Centers (CC BY). Definitions and '
        f'reasons in <code>config/energy.yml</code>.</p>')
