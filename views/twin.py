"""Twin: what the proposed indicator set would have called, against what ships.

A hidden page — reachable at /twin, never in the nav — so the presentation is
untouched while the proposal is being weighed. Both models are scored on the
same store at the same vintage, so the only thing that differs between the two
columns is the indicator set in config/twin.yml.

Local only: the twin reads a store that holds Bloomberg series, which the hosted
app neither has nor may publish.
"""

from __future__ import annotations

import datetime as dt
import os

import altair as alt
import pandas as pd
import streamlit as st

from src import backtest as bt
from src import twin as tw
from src import ui
from src.drivers import compute, load_config
from src.regimes import load_regimes, run
from src.store import Store
from views.common import INDICATORS, REGIMES

st.html('<h1 class="mr-title">Twin</h1>'
        '<p class="mr-sub">The live model and the bench\'s proposed indicator set, '
        'scored side by side on the same data. Not part of the presentation.</p>')

db = tw.default_db()
if not os.path.exists(db):
    st.html('<div class="mr-custom"><b>No twin store yet.</b> Build it on a machine '
            'with the Bloomberg terminal logged in:<br><code>python ingest.py --db '
            '"%LOCALAPPDATA%\\macro-regime\\macrobond.duckdb" twin-build</code></div>')
    st.stop()

overlay = tw.load_overlay()
live_cfg = load_config(INDICATORS())
twin_cfg = tw.build_config(live_cfg, overlay)
reg = load_regimes(REGIMES())
removed, added, changed = tw.swapped_ids(overlay)


@st.cache_data(ttl=900, show_spinner="Scoring both models…")
def _score(db_path: str, stamp: float, vintage: dt.date):
    store = Store(db_path)
    try:
        out = {}
        for name, cfg in (("live", live_cfg), ("twin", twin_cfg)):
            res = compute(store, cfg, vintage)
            res.update(run(res["drivers"], reg))
            out[name] = {k: res[k] for k in ("drivers", "calls", "probabilities", "indicators")}
        # The benchmark needs only three series every store already holds.
        out["actual"] = store.as_of(["UNRATE", "NROU", "PCEPILFE"], vintage)
        return out
    finally:
        store.close()


res = _score(db, os.path.getmtime(db), dt.date.today())
live, twin = res["live"], res["twin"]
rlabel = {k: v["label"] for k, v in reg["regimes"].items()}
rlabel.update({"transitional": "Transitional", "unclassified": "No clear regime"})
dlabel = {k: v.get("label", k) for k, v in live_cfg["drivers"].items()}

calls = pd.DataFrame({"live": live["calls"]["called"],
                      "twin": twin["calls"]["called"]}).dropna(how="all")
# Compare months both models have fully scored. The newest month is usually
# still missing a driver in one model or the other, and reading it would put
# a half-scored month next to a whole one.
start = pd.Timestamp(str((live_cfg.get("meta") or {}).get("history_start", "1970-01-01")))
complete = (live["drivers"].notna().all(axis=1) & twin["drivers"].notna().all(axis=1))
complete = complete[complete].index
calls = calls.loc[calls.index.isin(complete) & (calls.index >= start)]


def fmt(x: float) -> str:
    return "–" if pd.isna(x) else ui.signed(x)
built = pd.Timestamp(dt.datetime.fromtimestamp(os.path.getmtime(db)))
st.caption(f"Twin store built {built:%d %B %Y, %H:%M}. Rebuild with `ingest.py twin-build` "
           f"to pick up newer data.")


# ---------- the headline ----------

def agreement(frame: pd.DataFrame, since: str) -> tuple[float, int]:
    f = frame.loc[since:].dropna()
    return (float((f["live"] == f["twin"]).mean()) if len(f) else float("nan"), len(f))


latest = calls.dropna().index.max()
now_live, now_twin = calls.loc[latest, "live"], calls.loc[latest, "twin"]
windows = [("Since 1990", "1990"), ("Since 2003", "2003"),
           ("Since 2018", "2018"), ("Last 5 years", str(latest.year - 4))]
st.html(ui.section_head(
    "How often the two agree", meta=f"latest confirmed month {latest:%B %Y}",
    caption="Share of months where both models called the same regime. The windows "
            "matter: some proposed series start late — sovereign CDS in 2018, CBO's "
            "labour-force revisions in 2003 — and "
            "before then the twin scores those drivers on fewer indicators."))
cells = []
for label, since in windows:
    share, n = agreement(calls, since)
    cells.append(f'<div style="border-top:2px solid {ui.INK};padding:0.5rem 0.2rem">'
                 f'<div style="font-size:0.78rem;color:{ui.MUTED};text-transform:uppercase;'
                 f'letter-spacing:0.04em">{label}</div>'
                 f'<div style="font-size:1.7rem;font-weight:600">{share:.0%}</div>'
                 f'<div style="font-size:0.74rem;color:{ui.MUTED}">{n} months</div></div>')
st.html('<div style="display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,'
        f'minmax(9rem,1fr))">{"".join(cells)}</div>')

same = now_live == now_twin
st.html(f'<p style="margin-top:0.8rem;font-size:1rem">{latest:%B %Y}: the live model calls '
        f'<b>{ui.esc(rlabel.get(now_live, now_live))}</b>, the twin calls '
        f'<b>{ui.esc(rlabel.get(now_twin, now_twin))}</b>'
        f'{" — the same call." if same else " — they disagree."}</p>')


# ---------- the backtest ----------

st.html(ui.section_head(
    "How each did, after the fact",
    caption="Each month's call against what the economy actually did in the six months "
            "either side. An NBER recession anywhere in that window is a hard landing. "
            "Otherwise growth held if the unemployment gap rose by no more than 0.1 "
            "points, and inflation was high if core PCE averaged above 2.5%. Slow growth "
            "with low inflation and no recession fits no archetype, so declining to call "
            "it counts as right."))

truth = bt.realised(res["actual"])
# `calls` is already limited to months both models fully scored.
scored = {"live": calls["live"], "twin": calls["twin"]}
board = bt.scorecard(scored, truth, since="1990")
since03 = bt.scorecard(scored, truth, since="2003")
pct = lambda x: f"{x:.0%}"
rows_bt = []
for measure in board.index:
    fmt_ = (lambda v: f"{int(v)}") if measure.endswith("months") else pct
    rows_bt.append([measure, fmt_(board.loc[measure, "live"]), fmt_(board.loc[measure, "twin"])])
    if measure == "Months called correctly":
        rows_bt.append(["…since 2003", pct(since03.loc[measure, "live"]),
                        pct(since03.loc[measure, "twin"])])
st.html(ui.table(["Measure", "Live (presented)", "Twin (proposed)"], rows_bt, numeric={1, 2}))
naive_regime, naive_share = board.attrs["naive"]
st.caption(
    f"{board.attrs['months']} months, {board.attrs['first'].strftime('%b %Y')} to "
    f"{board.attrs['last'].strftime('%b %Y')}. "
    f"A costly error is a recession called a good regime, or a hard landing called in "
    f"Goldilocks. For scale: always calling {rlabel.get(naive_regime, naive_regime)} "
    f"would get {naive_share:.0%} of months right, while saying nothing about recessions. "
    f"Both models are scored on today's revised data, so the absolute numbers flatter "
    f"both; the comparison between them is like for like.")


# ---------- the two calls through time ----------

st.html(ui.section_head("The two calls through time",
                        caption="Each band is the regime the model called that month."))
colors = {k: v.get("color", "#999") for k, v in reg["regimes"].items()}
# Dark enough to read against the white page, so an unclassified month is not
# mistaken for a month neither model scored (those are left blank).
colors.update({"transitional": "#b3b3b3", "unclassified": "#d6d6d6"})
long = (calls.reset_index(names="month")
        .melt("month", var_name="model", value_name="regime").dropna())
long["model"] = long["model"].map({"live": "Live (presented)", "twin": "Twin (proposed)"})
long["label"] = long["regime"].map(lambda r: rlabel.get(r, r))
# One band per run of consecutive months with the same call, from the start of
# its first month to the start of the month after its last. Drawn month by
# month, each rect is under two pixels wide and the seams show as stripes; as
# runs they are solid, and hovering names the whole episode.
long["start"] = long["month"].dt.to_period("M").dt.to_timestamp()
long = long.sort_values(["model", "start"])
new_run = ((long["regime"] != long.groupby("model")["regime"].shift())
           | (long["start"] != long.groupby("model")["start"].shift() + pd.offsets.MonthBegin(1)))
long["run"] = new_run.groupby(long["model"]).cumsum()
long = (long.groupby(["model", "run"], as_index=False)
        .agg(regime=("regime", "first"), label=("label", "first"),
             start=("start", "first"), last=("start", "last")))
long["end"] = long["last"] + pd.offsets.MonthBegin(1)
long["months"] = ((long["end"].dt.year - long["start"].dt.year) * 12
                  + long["end"].dt.month - long["start"].dt.month)
domain = [r for r in colors if r in set(long["regime"])]
chart = (alt.Chart(long).mark_rect(strokeWidth=0)
         .encode(x=alt.X("start:T", title=None, axis=alt.Axis(format="%Y", tickCount=12)),
                 x2="end:T",
                 y=alt.Y("model:N", title=None, sort=["Live (presented)", "Twin (proposed)"],
                         scale=alt.Scale(paddingInner=0.3),
                         axis=alt.Axis(labelLimit=200, labelFontSize=12, ticks=False,
                                       domain=False, labelPadding=8)),
                 color=alt.Color("regime:N", title=None,
                                 scale=alt.Scale(domain=domain, range=[colors[r] for r in domain]),
                                 legend=alt.Legend(orient="bottom",
                                                   labelExpr=" + ".join(
                                                       f"(datum.label == '{r}' ? '{rlabel.get(r, r)}' : '')"
                                                       for r in domain))),
                 tooltip=[alt.Tooltip("model:N"), alt.Tooltip("label:N", title="Regime"),
                          alt.Tooltip("start:T", title="From", format="%b %Y"),
                          alt.Tooltip("last:T", title="To", format="%b %Y"),
                          alt.Tooltip("months:Q", title="Months")])
         # A fixed height per row: a total height was being shrunk to fit, which
         # squeezed both rows into a sliver.
         .properties(height=alt.Step(44)))
st.altair_chart(ui.style(chart), use_container_width=True)


# ---------- where they disagree ----------

st.html(ui.section_head("Where they disagree",
                        caption="Runs of consecutive months with different calls, longest first."))
diff = calls.dropna()
diff = diff[diff["live"] != diff["twin"]]
if diff.empty:
    st.caption("They never disagree.")
else:
    key = (diff.index.to_series().diff() > pd.Timedelta(days=40)).cumsum()
    runs = []
    for _, g in diff.groupby(key):
        runs.append([f"{g.index[0]:%b %Y}" + ("" if len(g) == 1 else f" – {g.index[-1]:%b %Y}"),
                     len(g), rlabel.get(g["live"].mode()[0], g["live"].mode()[0]),
                     rlabel.get(g["twin"].mode()[0], g["twin"].mode()[0])])
    runs.sort(key=lambda r: -r[1])
    st.html(ui.table(["Period", "Months", "Live calls", "Twin calls"], runs[:15], numeric={1}))


# ---------- which drivers moved ----------

st.html(ui.section_head(
    "Which drivers the new indicators move",
    caption="Driver scores under each model. The gap is the effect of the proposed "
            "indicators — largest on demand, where the unemployment gap has been cut to "
            "0.15 and the CBO revision dropped, and slightest on inflation expectations, "
            "which only adds a 5% revision alongside the anchors it already reads."))
now = latest
rows = []
for d in live_cfg["drivers"]:
    a, b = live["drivers"][d].loc[start:], twin["drivers"][d].loc[start:]
    gap = (b - a).dropna()
    rows.append([dlabel[d], fmt(a.get(now, float("nan"))), fmt(b.get(now, float("nan"))),
                 fmt(gap.get(now, float("nan"))), f"{gap.abs().mean():.2f}",
                 f"{a.corr(b):.2f}" if len(gap) > 24 else "–"])
st.html(ui.table(["Driver", f"Live, {now:%b %Y}", f"Twin, {now:%b %Y}", "Difference",
                  "Mean absolute gap", "Correlation"], rows, numeric={1, 2, 3, 4, 5}))

pick = st.selectbox("Show a driver through time", list(live_cfg["drivers"]),
                    format_func=lambda d: dlabel[d], index=0)
series = pd.DataFrame({"Live (presented)": live["drivers"][pick],
                       "Twin (proposed)": twin["drivers"][pick]}).loc[start:].dropna(how="all")
line = (alt.Chart(series.reset_index(names="month").melt("month", var_name="model",
                                                         value_name="score").dropna())
        .mark_line(strokeWidth=2)
        .encode(x=alt.X("month:T", title=None),
                y=alt.Y("score:Q", title=f"{dlabel[pick]} score", scale=alt.Scale(domain=[-1, 1])),
                color=alt.Color("model:N", title=None,
                                scale=alt.Scale(range=[ui.INK, "#2a78d6"]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("month:T", format="%b %Y"), "model:N",
                         alt.Tooltip("score:Q", format="+.2f")])
        .properties(height=240))
st.altair_chart(ui.style(line), use_container_width=True)


# ---------- what changed ----------

st.html(ui.section_head("What the twin changes",
                        caption="Read from config/twin.yml. A swapped-in indicator takes the "
                                "weight of the one it replaces, so the only weights that move "
                                "are the two listed here as deliberate cuts."))
names = {i["id"]: i.get("label", i["id"]) for d in live_cfg["drivers"].values()
         for i in d["indicators"]}
live_w = {i["id"]: i.get("weight", 0) for d in live_cfg["drivers"].values()
          for i in d["indicators"]}
tnames = {i["id"]: (i.get("label", i["id"]), i.get("weight", 0), d_key)
          for d_key, d in twin_cfg["drivers"].items() for i in d["indicators"]}
reweighted = overlay.get("reweight") or {}
table = []
for driver, swaps in (overlay.get("swaps") or {}).items():
    for s in swaps:
        nid = s["in"]["id"]
        table.append([dlabel[driver], names.get(s["out"], s["out"]), tnames[nid][0],
                      f"{tnames[nid][1]:.0%}"])
for driver, adds in (overlay.get("adds") or {}).items():
    for spec in adds:
        table.append([dlabel[driver], "— added, nothing dropped —",
                      tnames[spec["id"]][0], f"{tnames[spec['id']][1]:.0%}"])
for ind_id in (overlay.get("changes") or {}):
    table.append([dlabel[tnames[ind_id][2]], names.get(ind_id, ind_id),
                  f"{tnames[ind_id][0]} (read differently)", f"{tnames[ind_id][1]:.0%}"])
for ind_id, weight in reweighted.items():
    was = live_w.get(ind_id)
    # A swapped-in indicator has no live weight of its own; it inherited the
    # weight of what it replaced, so that is what the cut is measured from.
    if was is None:
        was = next(live_w[s["out"]] for sw in (overlay.get("swaps") or {}).values()
                   for s in sw if s["in"]["id"] == ind_id)
    verb = "weight cut" if weight < was else "weight raised"
    table.append([dlabel[tnames[ind_id][2]], names.get(ind_id, tnames[ind_id][0]),
                  f"{tnames[ind_id][0]} ({verb} from {was:.0%})", f"{weight:.0%}"])
st.html(ui.table(["Driver", "Out", "In", "Weight"], table, numeric={3}))
