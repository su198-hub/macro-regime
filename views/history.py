"""The /history page: which regime the US was actually in, month by month since 1980.

Built from the TR_ series `ingest.py truth` stores; src/truth.py holds the
method. Laid out for checking rather than for reading top to bottom: every
period shows its growth and inflation inputs in their own units, so a reader
can test each call against their own memory of the time, then open the months
behind it.
"""

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from src import truth, ui
from src.regimes import load_regimes
from views.common import REGIMES, data_version, get_store, published_note, refresh_button

RECESSION = "NBER recession"
RECESSION_COLOR = "#9a9a9a"
MON = "%b %Y"


@st.cache_data(ttl=900, show_spinner="Labeling every month since 1980…")
def labels(version: str) -> pd.DataFrame | None:
    wide = get_store().latest(list(truth.STORE_IDS.values()))
    if wide.empty or not set(truth.STORE_IDS.values()) <= set(wide.columns):
        return None
    return truth.build(truth.from_store(wide))


@st.cache_data(ttl=3600, show_spinner=False)
def notes() -> dict:
    with open("config/regime_history.yml", encoding="utf-8") as f:
        return {pd.Period(k, "M"): v for k, v in (yaml.safe_load(f).get("notes") or {}).items()}


reg = load_regimes(REGIMES())["regimes"]
NAME = {k: v["label"] for k, v in reg.items() if k in truth.QUADRANT.values()}
COLOR = {k: reg[k]["color"] for k in NAME}

st.html('<h1 class="mr-title">Regime history</h1>'
        '<p class="mr-sub">Which growth and inflation regime the US was actually in, month by month '
        'since 1980, judged with hindsight. A draft benchmark for scoring the model. Pick any period '
        'to see the numbers behind it and check the call against your own reading.</p>')
st.html('<hr class="mr-rule">')
refresh_button()

lab = labels(data_version())
if lab is None:
    st.html('<div class="mr-custom"><b>No regime history in this store yet.</b> Build it on the '
            'desk machine, then publish:<br><code>python ingest.py --db '
            '"%LOCALAPPDATA%\\macro-regime\\macrobond.duckdb" truth</code></div>')
    st.stop()

sp = truth.spells(lab)
NOTES = notes()
sp["note"] = [" ".join(v for k, v in NOTES.items() if s <= k <= e) for s, e in zip(sp["start"], sp["end"])]


def when(p: pd.Period) -> str:
    return p.to_timestamp().strftime(MON)


def span(r) -> str:
    return f"{when(r.start)} – {when(r.end)}"


def growth_word(g: float) -> str:
    return ("well below trend" if g <= -1 else "below trend" if g < -0.25 else "near trend"
            if g < 0.25 else "above trend" if g < 1 else "well above trend")


def inflation_word(x: float) -> str:
    return "below expectations" if x < -0.5 else "close to expectations" if x < 0.5 else "above expectations"


def confidence(a: float) -> str:
    return "solid" if a >= 0.7 else "mixed" if a >= 0.4 else "contested"


def chip(label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:.4rem;font-weight:600;'
            f'white-space:nowrap"><span style="width:.7rem;height:.7rem;border-radius:2px;'
            f'background:{COLOR[label]};display:inline-block"></span>{ui.esc(NAME[label])}</span>')


# ---------- summary ----------

now = sp.iloc[-1]
mix = lab["label"].value_counts(normalize=True)
rec = lab["recession"]
down = lab["label"].isin(["stagflation", "hard_landing"])
contested = int((sp["agree"] < 0.4).sum())
gold = sp[sp["label"] == "goldilocks"].sort_values("months").tail(1)
points = [
    f"<b>Now:</b> {NAME[now.label].lower()} since {when(now.start)}"
    + (" (provisional: the latest months' inflation window is not complete)." if now.provisional else "."),
    f"<b>{len(sp)} regime periods</b> from {when(lab.index[0])} to {when(lab.index[-1])}. "
    f"The median period lasted {int(sp['months'].median())} months.",
    "Share of months: " + ", ".join(f"{NAME[k].lower()} {mix.get(k, 0):.0%}" for k in NAME) + "."
    + (f" The longest goldilocks run was {span(gold.iloc[0])}." if len(gold) else ""),
    f"{down[rec].mean():.0%} of NBER recession months fall in a growth-down regime. "
    "A recession with inflation above expectations counts as stagflation, not hard landing.",
    f"The cross-check recipe agrees on {lab['agree'].mean():.0%} of months; {contested} of the "
    f"{len(sp)} periods are contested (under 40% agreement). Start your review with those.",
]
st.html(ui.section_head("Summary") + '<div class="m-body"><ul>'
        + "".join(f"<li>{p}</li>" for p in points) + "</ul></div>")

# ---------- check a period ----------

ALL = f"All periods, {lab.index[0].year}–{lab.index[-1].year}"
options = [ALL] + [f"{span(r)} · {NAME[r.label]}" for r in sp.itertuples()]
st.html(ui.section_head("The record", caption=
        "Top strip: the call, AQR's cross-check and NBER recessions. Below, the inputs in their own "
        "units, shaded by the call. Growth is up when GDP runs above potential and the activity index "
        "above its trend; inflation is high when core PCE runs above the expected line."))
pick = st.selectbox("Check a period", options, index=0, key="hist_period")
sel = None if pick == ALL else sp.iloc[options.index(pick) - 1]

if sel is not None:
    w = lab.loc[sel.start:sel.end]
    st.html(
        '<div class="mr-custom" style="background:#f6f7f8">'
        f'<p style="margin:0 0 .4rem">{chip(sel.label)} &nbsp; {span(sel)} · {sel.months} months'
        + (f" · {sel.recession_months} in recession" if sel.recession_months else "") + "</p>"
        f'<p style="margin:0 0 .3rem"><b>Growth {growth_word(sel.growth)}.</b> Real GDP grew '
        f'{sel.gdp_growth:.1f}% a year against CBO potential of {sel.potential_growth:.1f}%. The Chicago '
        f'Fed activity index averaged {sel.cfnai_ma3:+.2f} against its 10-year trend of {sel.cfnai_trend:+.2f}.</p>'
        f'<p style="margin:0 0 .3rem"><b>Inflation {inflation_word(sel.inflation)}.</b> Core PCE ran '
        f'{sel.core_pce:.1f}% against {sel.expected:.1f}% expected: the SPF 10-year CPI forecast of '
        f'{w["spf_cpi10"].mean():.1f}% less the usual CPI–PCE gap of {w["wedge"].mean():.1f} points.</p>'
        f'<p style="margin:0">AQR\'s recipe gives the same label in {sel.agree:.0%} of these months '
        f'({confidence(sel.agree)}). {sel.weak:.0%} of months sat inside a neutral band on at least one '
        'axis, where the call carries over from before.'
        + (f" <i>{ui.esc(sel.note)}</i>" if sel.note else "") + "</p></div>")
    pad = 18
    lo, hi = sel.start - pad, sel.end + pad
else:
    lo, hi = lab.index[0], lab.index[-1]
lo, hi = max(lo, lab.index[0]), min(hi, lab.index[-1])

# ---------- charts ----------

view = lab.loc[lo:hi]
x0, x1 = lo.to_timestamp(), (hi + 1).to_timestamp()
X = ui.time_x("start", x0, x1, fmt="%Y" if (hi - lo).n > 30 else "%b %Y",
              months=(1,) if (hi - lo).n > 30 else (1, 7))
bands = sp[(sp["end"] >= lo) & (sp["start"] <= hi)].assign(
    start=lambda d: pd.to_datetime([max(p, lo).to_timestamp() for p in d["start"]]),
    end=lambda d: pd.to_datetime([(min(p, hi) + 1).to_timestamp() for p in d["end"]]),
    regime=lambda d: d["label"].map(NAME))
domain, rng = list(NAME.values()) + [RECESSION], list(COLOR.values()) + [RECESSION_COLOR]
regime_color = alt.Color("regime:N", scale=alt.Scale(domain=domain, range=rng), title=None)


def runs(s: pd.Series, row: str) -> pd.DataFrame:
    """Unbroken runs of one value as rectangles; blank months break a run."""
    s = s.fillna("")
    r = (s != s.shift()).cumsum()
    out = [{"row": row, "start": g.index[0].to_timestamp(), "end": (g.index[-1] + 1).to_timestamp(),
            "regime": g.iloc[0]} for _, g in s.groupby(r) if g.iloc[0]]
    return pd.DataFrame(out, columns=["row", "start", "end", "regime"]).astype(
        {"start": "datetime64[ns]", "end": "datetime64[ns]"})


strip = pd.concat([runs(view["label"].map(NAME), "Call"),
                   runs(view["aqr_label"].map(NAME), "AQR recipe"),
                   runs(view["recession"].map({True: RECESSION}), "Recession")], ignore_index=True)
strip_chart = alt.Chart(strip).mark_rect().encode(
    x=X, x2="end:T", y=alt.Y("row:N", sort=["Call", "AQR recipe", "Recession"], title=None,
                              axis=alt.Axis(labelColor=ui.INK, labelFontSize=12, ticks=False, domain=False)),
    color=regime_color,
    tooltip=[alt.Tooltip("regime:N", title="Regime"),
             alt.Tooltip("start:T", title="From", timeUnit="utcyearmonth", format="%B %Y")],
).properties(height=78)


def panel(cols: dict, title: str, unit: str, clip: tuple, dash: dict, zero: bool = False):
    d = view[list(cols)].rename(columns=cols)
    d["start"] = ui.month_mid(d.index.to_timestamp())
    long = d.melt("start", var_name="series", value_name="value").dropna()
    lo_v = max(clip[0], float(long["value"].min()) - 0.5)
    hi_v = min(clip[1], float(long["value"].max()) + 0.5)
    shade = alt.Chart(bands).mark_rect(opacity=0.13).encode(
        x=X, x2="end:T", color=alt.Color("regime:N", scale=alt.Scale(domain=domain, range=rng), legend=None))
    names = list(cols.values())
    lines = alt.Chart(long).mark_line(strokeWidth=1.6, clip=True).encode(
        x=X, y=alt.Y("value:Q", title=f"{title} ({unit})", scale=alt.Scale(domain=[lo_v, hi_v], nice=False)),
        stroke=alt.Stroke("series:N", scale=alt.Scale(domain=names, range=[ui.INK, "#6b6b6b"][:len(names)]),
                          legend=alt.Legend(orient="top", title=None, labelLimit=400)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=names, range=[dash.get(n, [1, 0]) for n in names]),
                                  legend=None),
        tooltip=[ui.month_tip("start"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("value:Q", title=unit, format=".2f")])
    layers = [shade, lines]
    if sel is not None:
        # Edges of the period picked, so the zoomed context reads at a glance.
        edges = pd.DataFrame({"start": [sel.start.to_timestamp(), (sel.end + 1).to_timestamp()]})
        layers.append(alt.Chart(edges).mark_rule(color=ui.INK, strokeDash=[2, 2], strokeWidth=1.2).encode(x=X))
    if zero:
        layers.insert(1, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=ui.AXIS).encode(y="y:Q"))
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=170)


g1 = panel({"gdp_growth": "Real GDP growth", "potential_growth": "CBO potential growth"},
           "GDP", "% a year", (-10, 12), {"CBO potential growth": [5, 3]}, zero=True)
g2 = panel({"cfnai_ma3": "Chicago Fed activity index (3-month average)", "cfnai_trend": "Its 10-year median"},
           "Activity", "index", (-4, 3), {"Its 10-year median": [5, 3]}, zero=True)
i1 = panel({"core_pce": "Core PCE inflation", "expected": "Expected inflation (PCE terms)"},
           "Inflation", "% a year", (-1, 12), {"Expected inflation (PCE terms)": [5, 3]})
chart = alt.vconcat(strip_chart, g1, g2, i1, spacing=14).resolve_scale(
    x="shared", color="independent", stroke="independent", strokeDash="independent")
st.altair_chart(ui.style(chart), width="stretch")
st.html('<p class="mr-caption">GDP and CBO potential are growth over the half-year around each quarter, '
        'annualized. Core PCE is the 12 months centered on each month. Expected inflation is the SPF '
        '10-year CPI forecast less the usual CPI–PCE gap. Covid quarters are clipped at the panel edge.'
        + (' Dotted lines mark the period picked.' if sel is not None else '') + '</p>')

# ---------- table of periods ----------

st.html(ui.section_head("Regime by period", meta="oldest first", caption=
        "Each input averaged over the period. Agreement is how often AQR's recipe gives the same "
        "label: solid is 70% or more, contested under 40%."))
rows = []
for r in sp.itertuples():
    rows.append([
        span(r) + (" *" if r.provisional else ""), r.months, ui.Raw(chip(r.label)),
        f"{r.gdp_growth:.1f}% vs {r.potential_growth:.1f}%",
        f"{r.cfnai_ma3:+.2f} vs {r.cfnai_trend:+.2f}",
        f"{r.core_pce:.1f}% vs {r.expected:.1f}%",
        f"{r.agree:.0%} {confidence(r.agree)}",
        (f"{r.recession_months} months of recession. " if r.recession_months else "") + r.note,
    ])
st.html(ui.table(["Period", "Months", "Regime", "GDP vs potential", "Activity vs trend",
                  "Core PCE vs expected", "Agreement", "What was going on"], rows, numeric={1}))
st.html('<p class="mr-caption">* Provisional: core PCE for the latest months is not yet centered on '
        'a full 12-month window. Notes are written by hand in config/regime_history.yml.</p>')

# ---------- the months behind it ----------

COLS = {
    "label": "Call", "strength": "Strength", "aqr_label": "AQR recipe",
    "gdp_growth": "Real GDP growth %", "potential_growth": "CBO potential %",
    "cfnai_ma3": "Activity index (3-mo)", "cfnai_trend": "Activity trend",
    "growth": "Growth score", "core_pce": "Core PCE %", "spf_cpi10": "SPF 10-yr CPI %",
    "wedge": "CPI–PCE gap", "expected": "Expected %", "inflation": "Inflation gap",
    "recession": "Recession", "provisional": "Provisional",
}
monthly = lab[list(COLS)].rename(columns=COLS)
for c in ("Call", "AQR recipe"):
    monthly[c] = monthly[c].map(NAME)
monthly.index = monthly.index.strftime("%Y-%m")
monthly.index.name = "Month"
shown = monthly.loc[str(lo):str(hi)] if sel is not None else monthly
st.html(ui.section_head("The months behind it", meta=f"{len(shown)} months", caption=
        "Every input, month by month, for the period picked above (all months otherwise). Growth score is "
        "in typical moves; inflation gap is in points. The call flips only when a score clears its "
        f"band (±{truth.GROWTH_BAND} for growth, ±{truth.INFLATION_BAND} points for inflation)."))
st.dataframe(shown.round(2), width="stretch", height=360)
a, b, _ = st.columns([1, 1, 3])
a.download_button("Download all months (CSV)", monthly.to_csv().encode("utf-8"),
                  file_name="regime_history_monthly.csv", mime="text/csv", key="dl_months")
periods = sp.assign(start=sp["start"].astype(str), end=sp["end"].astype(str), label=sp["label"].map(NAME))
b.download_button("Download periods (CSV)", periods.round(3).to_csv(index=False).encode("utf-8"),
                  file_name="regime_history_periods.csv", mime="text/csv", key="dl_periods")

# ---------- methodology ----------

st.html(ui.section_head("How each month is labeled"))
src_rows = [
    ["Chicago Fed National Activity Index", "Chicago Fed", "Monthly, 1967–", truth.CODES["cfnai"]],
    ["Real GDP", "BEA", "Quarterly", truth.CODES["gdp"]],
    ["Real potential GDP", "CBO", "Quarterly", truth.CODES["potential"]],
    ["PCE price index, and excluding food and energy", "BEA", "Monthly",
     f'{truth.CODES["pce"]}, {truth.CODES["core_pce"]}'],
    ["Consumer price index, all items", "BLS", "Monthly", truth.CODES["cpi"]],
    ["Expected CPI inflation over 10 years, median", "Philadelphia Fed SPF; Blue Chip and Livingston before 1991",
     "Quarterly, 1979–", truth.CODES["spf_cpi10"]],
    ["Industrial production", "Federal Reserve", "Monthly", truth.CODES["ip"]],
    ["SPF forecasts of industrial production and CPI", "Philadelphia Fed SPF", "Quarterly",
     f'{truth.CODES["spf_ip_q0"]}, {truth.CODES["spf_ip_q4"]}, {truth.CODES["spf_cpi1y"]}'],
    ["Recession dates", "NBER", "Monthly", "fixed list in src/backtest.py"],
]
st.html(
    '<div class="m-body">'
    "<p>Each month gets one of the model's four regimes by where growth and inflation stood. "
    "It uses today's revised data, so it shows what happened, not what anyone could see at the time.</p>"
    "<ol>"
    "<li><b>Growth.</b> Two measures, each divided by its typical move over 1980–2019 (median absolute "
    "deviation, so recessions and Covid don't set the scale), then averaged:"
    "<ul><li>The Chicago Fed activity index, averaged over the month and its neighbours, minus its median "
    "over the 10 years around it. The index's zero is the average pace since 1967, which sits above the "
    "trend since 2000.</li>"
    "<li>Real GDP growth minus CBO potential growth, over the half-year around the month. This carries "
    "the services side, which the activity index underweights.</li></ul></li>"
    "<li><b>Inflation.</b> Core PCE over the 12 months centered on the month, annualized, minus expected "
    "inflation. Expected inflation is the SPF 10-year CPI median less the CPI–PCE gap over the 10 years "
    "around the month. For the latest 6 months the window runs to the last print, and those months are "
    "marked provisional.</li>"
    f"<li><b>Neutral bands.</b> Growth flips only when its score clears ±{truth.GROWTH_BAND}; inflation "
    f"only when its gap clears ±{truth.INFLATION_BAND} points. Inside a band the axis keeps its last "
    "call and the month is marked weak.</li>"
    "<li><b>Quadrant.</b> Growth up and inflation low is goldilocks. Up and high is high growth, high "
    "inflation. Down and high is stagflation. Down and low is hard landing.</li>"
    f"<li><b>Persistence.</b> A quadrant must hold {truth.MIN_MONTHS} months. A shorter spell takes the "
    "label before it.</li>"
    "<li><b>Cross-check.</b> AQR's published recipe (Ilmanen, Maloney and Ross, <i>Journal of Portfolio "
    "Management</i>, 2014), extended to today. Growth averages z-scores of the activity index and of "
    "industrial production growth less the SPF forecast. Inflation averages z-scores of CPI inflation "
    "and of CPI less the SPF forecast. Each axis is up when above its full-sample median. It ranks "
    "rather than anchors, so it calls half of all months high inflation by construction, which is "
    "where most disagreement comes from.</li>"
    "</ol>"
    "<p><b>Not the same as the backtest's current truth.</b> That rule uses a fixed 2.5% core PCE "
    "line, the change in the unemployment gap, and forces every recession to hard landing. The two "
    "match in about half of months; see the review notes for where and why.</p>"
    "<p><b>Still open:</b> whether inflation should be judged against expectations or a fixed line, "
    "whether a slowdown without a recession should count as hard landing (1995, 2016, 2019), and "
    "whether to score the model on every month or only where both recipes agree.</p>"
    "</div>")
st.html(ui.table(["Input", "Publisher", "Frequency", "Macrobond code"], src_rows))

ep = truth.episode_check(lab)
st.html(ui.section_head("Sanity check against well-known episodes", caption=
        "The share of months in each stretch that match the textbook reading. The textbook readings "
        "are a judgment too, so challenge them."))
st.html(ui.table(
    ["Episode", "Months", "Textbook reading", "Call match", "AQR match"],
    [[e["episode"], f"{pd.Period(e['from'], 'M').to_timestamp():%b %Y} – "
      f"{pd.Period(e['to'], 'M').to_timestamp():%b %Y}",
      ui.Raw(chip(e["expected"])), f"{e['primary share']:.0%}", f"{e['aqr share']:.0%}"]
     for e in ep.to_dict("records")],
    numeric={3, 4}))

st.html(f'<div class="mr-foot"><b>Sources:</b> Chicago Fed, BEA, BLS, CBO, Federal Reserve, '
        f'Philadelphia Fed Survey of Professional Forecasters and NBER, via Macrobond. Rebuild with '
        f'<code>python ingest.py truth</code>, then publish. {published_note()}</div>')
