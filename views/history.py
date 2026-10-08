"""The /history page: which regime the US was actually in, month by month since 1980.

Built from the TR_ series `ingest.py truth` stores; src/truth.py holds the
method. The page opens on the record itself, with a picker to zoom into any
period and check its label against the two inputs in their own units: GDP
against potential, core PCE against expectations. How a month is labeled and
what the record shows follow underneath.

Two controls: which label (the nowcast, the regime at the time; or a forward
label, the regime of the next three or five years with perfect foresight) and
which inflation benchmark (the survey, or a stepped target). Forward labels
answer a different question, so they are checked against what followed rather
than against episodes.
Notes are written for the adopted method and only appear with it.
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
GB, IB = truth.GROWTH_BAND, truth.INFLATION_BAND


@st.cache_data(ttl=900, show_spinner=False)
def inputs(version: str) -> pd.DataFrame | None:
    wide = get_store().latest(list(truth.STORE_IDS.values()))
    if wide.empty or not set(truth.STORE_IDS.values()) <= set(wide.columns):
        return None
    return truth.from_store(wide)


@st.cache_data(ttl=900, show_spinner="Labeling every month since 1980…")
def labels(version: str, years: int, align: str, anchor: str) -> pd.DataFrame | None:
    m = inputs(version)
    return None if m is None else truth.build(m, years=years, align=align, anchor=anchor)


def mode(s: pd.Series):
    """Most common label, or None when the window has no labels (forward
    labels stop years before the latest data)."""
    return s.value_counts().idxmax() if len(s) else None


@st.cache_data(ttl=900, show_spinner="Comparing labels…")
def comparison(version: str) -> pd.DataFrame:
    base = labels(version, **truth.DEFAULT)
    rows = []
    for view, cfg in truth.VIEWS.items():
        years, align = cfg["years"], cfg["align"]
        for anchor in truth.ANCHORS:
            lab_ = labels(version, years, align, anchor)
            sp_ = truth.spells(lab_)
            ep_ = truth.episode_check(lab_)
            down_ = lab_["label"].isin(["stagflation", "hard_landing"])
            rows.append({
                "view": view, "years": years, "align": align, "anchor": anchor, "periods": len(sp_),
                "median": sp_["months"].median(), "recessions": down_[lab_["recession"]].mean(),
                "episodes": int((ep_["share"] >= 0.5).sum()), "of": len(ep_),
                "same": (lab_["label"].reindex(base.index) == base["label"]).mean(),
                "mid80s": mode(lab_.loc["1983-07":"1986-12", "label"]),
                "soft": mode(lab_.loc["2023-07":"2024-12", "label"]),
                "provisional": int(lab_["provisional"].sum()), "end": lab_.index[-1],
                "followed": (truth.what_followed(lab_, base["label"], years)["agree"].mean()
                             if align == "forward" else None)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def notes() -> dict:
    with open("config/regime_history.yml", encoding="utf-8") as f:
        return {pd.Period(k, "M"): v for k, v in (yaml.safe_load(f).get("notes") or {}).items()}


reg = load_regimes(REGIMES())["regimes"]
NAME = {k: v["label"] for k, v in reg.items() if k in truth.QUADRANT.values()}
COLOR = {k: reg[k]["color"] for k in NAME}

st.html('<h1 class="mr-title">Regime history</h1>'
        '<p class="mr-sub">Which growth and inflation regime the US was in, month by month since 1980, '
        "judged with hindsight. This is the yardstick the model's calls are scored against, not the "
        "model's call.</p>")
st.html('<hr class="mr-rule">')
refresh_button()

WINDOW_NAME = {1: "1 year", 3: "3 years", 5: "5 years"}
VIEW_NAME = {"now": "Nowcast: the regime at the time", "next3": "Forward: the next 3 years",
             "next5": "Forward: the next 5 years"}
ANCHOR_NAME = {"survey": "Expected inflation (survey)", "stepped": "Stepped target: 4%, 3%, 2%"}
ADOPTED = " (adopted)"

with st.container(border=True):
    st.html('<p class="mr-caption" style="margin:0"><b>Label shown.</b> The nowcast answers "what regime '
            'were we in?"; a forward label answers "what did the next few years turn out to be?", with '
            'perfect foresight. Every combination is compared in "Compare labels" below.</p>')
    c1, c2 = st.columns(2)
    view = c1.radio("Label", list(truth.VIEWS), key="hist_view",
                    format_func=lambda v: VIEW_NAME[v] + (ADOPTED if v == "now" else ""))
    anchor = c2.radio("Inflation benchmark", truth.ANCHORS, key="hist_anchor",
                      format_func=lambda a: ANCHOR_NAME[a] + (ADOPTED if a == truth.DEFAULT["anchor"] else ""))
years, align = truth.VIEWS[view]["years"], truth.VIEWS[view]["align"]
METHOD = {"years": years, "align": align, "anchor": anchor}
FORWARD = align == "forward"
GB, IB = truth.band(years, truth.GROWTH_BAND), truth.band(years, truth.INFLATION_BAND)
IS_DEFAULT = METHOD == truth.DEFAULT
if not IS_DEFAULT:
    st.html('<div class="mr-custom"><b>Showing an alternative, not the adopted label:</b> '
            f'{VIEW_NAME[view].lower()}, with {ANCHOR_NAME[anchor].lower()}.'
            + (f" Each month is labeled with the regime of the {WINDOW_NAME[years]} that followed, with "
               "perfect foresight: the yardstick for judging what the signposts said about the years ahead."
               if FORWARD else "") + '</div>')

lab = labels(data_version(), years, align, anchor)
if lab is None:
    st.html('<div class="mr-custom"><b>No regime history in this store yet.</b> Build it on the '
            'desk machine, then publish:<br><code>python ingest.py --db '
            '"%LOCALAPPDATA%\\macro-regime\\macrobond.duckdb" truth</code></div>')
    st.stop()

sp = truth.spells(lab)
NOTES = notes()
sp["note"] = ([" ".join(v for k, v in NOTES.items() if s <= k <= e) for s, e in zip(sp["start"], sp["end"])]
              if IS_DEFAULT else "")

# Words for the method shown, used in every sentence that describes it.
G_WIN = {1: "four quarters", 3: "three years", 5: "five years"}[years]
I_WIN = {1: "12 months", 3: "three years", 5: "five years"}[years]
WHERE = {"centered": "centered on the month", "forward": "starting in the month"}[align]
BENCH = ("expected inflation (PCE terms)" if anchor == "survey"
         else "a stepped target: 4% before 1990, 3% for 1990–95, 2% from 1996")
REF_NAME = "Expected inflation (PCE terms)" if anchor == "survey" else "Stepped target (4% / 3% / 2%)"


def when(p: pd.Period) -> str:
    return p.to_timestamp().strftime("%b %Y")


def span(r) -> str:
    return f"{when(r.start)} – {when(r.end)}"


def versus(gap: float, band: float, below: str, near: str, above: str) -> str:
    return below if gap <= -band else above if gap >= band else near


def chip(label: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;gap:.4rem;font-weight:600;'
            f'white-space:nowrap"><span style="width:.7rem;height:.7rem;border-radius:2px;'
            f'background:{COLOR[label]};display:inline-block"></span>{ui.esc(NAME[label])}</span>')


# ---------- the record ----------

ALL = f"All periods, {lab.index[0].year}–{lab.index[-1].year}"
options = [ALL] + [f"{span(r)} · {NAME[r.label]}" for r in sp.itertuples()]
st.html(ui.section_head("The record", caption=
        "The regime each month, then the two inputs behind it, shaded by regime. Growth is up when "
        "GDP runs above the potential line; inflation is high when core PCE runs above the inflation "
        "benchmark. Pick a period to zoom in and see its numbers."))
pick = st.selectbox("Check a period", options, index=0, key=f"hist_period_{years}_{align}_{anchor}")
sel = None if pick == ALL else sp.iloc[options.index(pick) - 1]

if sel is not None:
    w = lab.loc[sel.start:sel.end]
    st.html(
        '<div class="mr-custom" style="background:#f6f7f8">'
        f'<p style="margin:0 0 .4rem">{chip(sel.label)} &nbsp; {span(sel)} · {sel.months} months'
        + (f" · {sel.recession_months} in recession" if sel.recession_months else "")
        + (" · provisional" if sel.provisional else "") + "</p>"
        f'<p style="margin:0 0 .3rem"><b>Growth {versus(sel.growth, GB, "below", "close to", "above")} '
        f'potential.</b> Real GDP grew {sel.gdp_growth:.2f}% a year against potential of '
        f'{sel.potential_growth:.2f}%, a gap of {sel.growth:+.2f} points.</p>'
        f'<p style="margin:0 0 .3rem"><b>Inflation {versus(sel.inflation, IB, "below", "close to", "above")} '
        + (f'expectations.</b> Core PCE ran {sel.core_pce:.2f}% against {sel.expected:.2f}% expected '
           f'(the survey\'s {w["spf_cpi10"].mean():.2f}% for CPI, less the usual CPI–PCE gap of '
           f'{w["wedge"].mean():.2f}), a gap of {sel.inflation:+.2f} points.</p>' if anchor == "survey" else
           f'the line.</b> Core PCE ran {sel.core_pce:.2f}% against a line of {sel.expected:.2f}%, '
           f'a gap of {sel.inflation:+.2f} points.</p>')
        + f'<p style="margin:0">In {sel.clear:.0%} of these months both gaps were outside ±{GB}; '
        'in the rest, at least one axis carried its earlier reading.'
        + (f" <i>{ui.esc(sel.note)}</i>" if sel.note else "") + "</p></div>")
    lo, hi = max(sel.start - 18, lab.index[0]), min(sel.end + 18, lab.index[-1])
else:
    lo, hi = lab.index[0], lab.index[-1]

view = lab.loc[lo:hi]
x0, x1 = lo.to_timestamp(), (hi + 1).to_timestamp()
long_span = (hi - lo).n > 30
X = ui.time_x("start", x0, x1, fmt="%Y" if long_span else "%b %Y", months=(1,) if long_span else (1, 7))
bands = sp[(sp["end"] >= lo) & (sp["start"] <= hi)].assign(
    start=lambda d: pd.to_datetime([max(p, lo).to_timestamp() for p in d["start"]]),
    end=lambda d: pd.to_datetime([(min(p, hi) + 1).to_timestamp() for p in d["end"]]),
    regime=lambda d: d["label"].map(NAME))
domain, rng = list(NAME.values()) + [RECESSION], list(COLOR.values()) + [RECESSION_COLOR]


def runs(s: pd.Series, row: str) -> pd.DataFrame:
    """Unbroken runs of one value as rectangles; blank months break a run."""
    s = s.fillna("")
    r = (s != s.shift()).cumsum()
    out = [{"row": row, "start": g.index[0].to_timestamp(), "end": (g.index[-1] + 1).to_timestamp(),
            "regime": g.iloc[0]} for _, g in s.groupby(r) if g.iloc[0]]
    return pd.DataFrame(out, columns=["row", "start", "end", "regime"]).astype(
        {"start": "datetime64[ns]", "end": "datetime64[ns]"})


# Provisional months get their own runs so they can be drawn faded.
regime_runs = view["label"].map(NAME).where(~view["provisional"], view["label"].map(NAME) + " ")
strip = pd.concat([runs(regime_runs, "Regime"),
                   runs(view["recession"].map({True: RECESSION}), "Recession")], ignore_index=True)
strip["provisional"] = strip["regime"].str.endswith(" ")
strip["regime"] = strip["regime"].str.strip()
strip_chart = alt.Chart(strip).mark_rect().encode(
    x=X, x2="end:T",
    y=alt.Y("row:N", sort=["Regime", "Recession"], title=None,
            axis=alt.Axis(labelColor=ui.INK, labelFontSize=12, ticks=False, domain=False)),
    color=alt.Color("regime:N", scale=alt.Scale(domain=domain, range=rng), title=None),
    opacity=alt.Opacity("provisional:N", scale=alt.Scale(domain=[False, True], range=[1, 0.4]), legend=None),
    tooltip=[alt.Tooltip("regime:N", title="Regime"), alt.Tooltip("provisional:N", title="Provisional"),
             alt.Tooltip("start:T", title="From", timeUnit="utcyearmonth", format="%B %Y")],
).properties(height=52)


def panel(cols: dict, title: str, clip: tuple, band: float, zero: bool = False):
    """The measure against its reference line, with the ±band around the
    reference inside which the axis keeps its earlier reading."""
    ref = list(cols)[1]
    zone = pd.DataFrame({"start": ui.month_mid(view.index.to_timestamp()),
                         "lo": (view[ref] - band).values, "hi": (view[ref] + band).values})
    d = view[list(cols)].rename(columns=cols)
    d["start"] = ui.month_mid(d.index.to_timestamp())
    long = d.melt("start", var_name="series", value_name="value").dropna()
    lo_v = max(clip[0], float(long["value"].min()) - 0.5)
    hi_v = min(clip[1], float(long["value"].max()) + 0.5)
    names = list(cols.values())
    shade = alt.Chart(bands).mark_rect(opacity=0.13).encode(
        x=X, x2="end:T", color=alt.Color("regime:N", scale=alt.Scale(domain=domain, range=rng), legend=None),
        tooltip=[alt.Tooltip("regime:N", title="Regime"),
                 alt.Tooltip("start:T", title="From", timeUnit="utcyearmonth", format="%B %Y")])
    lines = alt.Chart(long).mark_line(strokeWidth=1.6, clip=True).encode(
        x=X, y=alt.Y("value:Q", title=f"{title} (% a year)", scale=alt.Scale(domain=[lo_v, hi_v], nice=False)),
        stroke=alt.Stroke("series:N", scale=alt.Scale(domain=names, range=[ui.INK, "#6b6b6b"]),
                          legend=alt.Legend(orient="top", title=None, labelLimit=400)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=names, range=[[1, 0], [5, 3]]),
                                  legend=None),
        tooltip=[ui.month_tip("start"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("value:Q", title="% a year", format=".2f")])
    neutral = alt.Chart(zone).mark_area(color="#000000", opacity=0.08, clip=True).encode(
        x=X, y=alt.Y("lo:Q", scale=alt.Scale(domain=[lo_v, hi_v], nice=False)), y2="hi:Q",
        tooltip=[ui.month_tip("start"),
                 alt.Tooltip("lo:Q", title="Band, lower edge (% a year)", format=".2f"),
                 alt.Tooltip("hi:Q", title="Band, upper edge (% a year)", format=".2f")])
    layers = [shade, neutral, lines]
    if zero:
        layers.insert(1, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=ui.AXIS, tooltip=False).encode(y="y:Q"))
    if sel is not None:
        edges = pd.DataFrame({"start": [sel.start.to_timestamp(), (sel.end + 1).to_timestamp()]})
        layers.append(alt.Chart(edges).mark_rule(color=ui.INK, strokeDash=[2, 2], strokeWidth=1.2, tooltip=False).encode(x=X))
    return alt.layer(*layers).resolve_scale(color="independent").properties(height=190)


growth_chart = panel({"gdp_growth": "Real GDP growth", "potential_growth": "CBO potential growth"},
                     "Growth", (-6, 10), GB, zero=True)
inflation_chart = panel({"core_pce": "Core PCE inflation", "expected": REF_NAME},
                        "Inflation", (-1, 12), IB)
chart = alt.vconcat(strip_chart, growth_chart, inflation_chart, spacing=14).resolve_scale(
    x="shared", color="independent", stroke="independent", strokeDash="independent")
st.altair_chart(ui.style(chart), width="stretch")
st.html(f'<p class="mr-caption">Each line is measured over the window {WHERE}: {G_WIN} for GDP, '
        f'{I_WIN} for core PCE. The gray band is ±{GB} points around potential and the inflation '
        'benchmark: inside it, an axis keeps its earlier reading. Faded months are provisional. '
        'Covid quarters are clipped at the panel edge.'
        + (" Dotted lines mark the period picked." if sel is not None else "") + "</p>")

# ---------- reading the record ----------

now = sp.iloc[-1]
mix = lab["label"].value_counts(normalize=True)
rec = lab["recession"]
down = lab["label"].isin(["stagflation", "hard_landing"])
first_prov = lab.index[lab["provisional"]].min() if lab["provisional"].any() else None
how = [
    f"<b>Growth</b> is real GDP growth minus CBO's estimate of potential growth, over the {G_WIN} "
    f"{WHERE}. Above +{GB} points is growth up; below −{GB} is growth down.",
    f"<b>Inflation</b> is core PCE inflation over the {I_WIN} {WHERE}, minus "
    + ("expected inflation: the Philadelphia Fed survey's 10-year forecast, converted from CPI to PCE terms"
       + (", averaged over the same window" if years > 1 else "") if anchor == "survey" else BENCH)
    + f". Above +{IB} points is high; below −{IB} is low.",
    f"<b>Regime</b> is the combination of the two. Inside the ±{GB} bands an axis keeps its previous "
    f"reading, and a regime must last {truth.MIN_MONTHS} months to count.",
    "<b>Updates:</b> every month is relabeled from the latest revised data whenever the dashboard "
    "data is refreshed. GDP and the survey are quarterly; core PCE is monthly."
    + (f" Months from {when(first_prov)} on are provisional until their windows fill, and keep the "
       "last confirmed regime until then." if first_prov else
       f" Forward labels need the whole window, so they stop {WINDOW_NAME[years]} before the latest data."),
]
# What the provisional months would say on their own, if it differs.
leans = next((x for x in lab.loc[lab["provisional"], "provisional_reading"].unique() if x != now.label), None)
shows = [
    f"<b>Latest:</b> {NAME[now.label].lower()} since {when(now.start)}, with labels running to "
    f"{when(lab.index[-1])}, "
    + (f"the last month with a full {WINDOW_NAME[years]} of data after it." if FORWARD else
       "the last month GDP covers.")
    + (f" The provisional months' own readings point to {NAME[leans].lower()}; a new regime is "
       "not called until the data is complete." if leans else ""),
    f"<b>{len(sp)} regime periods</b> since {when(lab.index[0])}, lasting {int(sp['months'].median())} "
    "months at the median.",
    "<b>Share of months:</b> " + ", ".join(f"{NAME[k].lower()} {mix.get(k, 0):.0%}" for k in NAME) + ".",
    ("<b>Recessions</b> show up in the labels of the months before them, since a forward label "
     "describes the years that follow." if FORWARD else
     f"<b>Recessions:</b> {down[rec].mean():.0%} of NBER recession months fall in a growth-down regime. "
     "By recession: " + "; ".join(
        f"{pk[:4]}{'–' + tr[2:4] if tr[:4] != pk[:4] else ''} {NAME[w.value_counts().idxmax()].lower()}"
        for pk, tr in truth.NBER if pk >= "1980"
        for w in [lab.loc[pd.Period(pk, "M"):pd.Period(tr, "M"), "label"]] if len(w)) + "."),
]
st.html(ui.section_head("Reading the record"))
left, right = st.columns(2, gap="large")
left.html('<div class="m-body"><p><b>How a month is labeled</b></p><ul>'
          + "".join(f"<li>{x}</li>" for x in how) + "</ul></div>")
right.html('<div class="m-body"><p><b>What the record shows</b></p><ul>'
           + "".join(f"<li>{x}</li>" for x in shows) + "</ul></div>")

# ---------- does it match history ----------

ep = truth.episode_check(lab)
if FORWARD:
    now_lab = labels(data_version(), **truth.DEFAULT)["label"]
    wf = truth.what_followed(lab, now_lab, years)
    st.html(ui.section_head("Does it match what followed?", caption=
            f"A forward label sums up the {WINDOW_NAME[years]} after each month. Here it is set against the "
            "regimes those years actually went through, by the adopted month-by-month labels. The "
            f"forward label matches the most common of them in {wf['agree'].mean():.0%} of months."))
    jan = wf[wf.index.month == 1]
    with st.expander(f"Show each January, {jan.index[0].year}–{jan.index[-1].year}", expanded=False):
        st.html(ui.table(
            ["Start", "Forward label", "GDP vs potential", "Core PCE vs benchmark", "What those years went through"],
            [[f"{t.year}", ui.Raw(chip(r["label"])),
              f"{lab.loc[t, 'gdp_growth']:.2f}% vs {lab.loc[t, 'potential_growth']:.2f}%",
              f"{lab.loc[t, 'core_pce']:.2f}% vs {lab.loc[t, 'expected']:.2f}%",
              ", ".join(f"{NAME[k].lower()} {v:.0%}" for k, v in r["mix"].items() if v >= 0.1)]
             for t, r in jan.iterrows()]))
else:
    st.html(ui.section_head("Does it match history?", caption=
            "Well-known stretches and how most people would label them, with the share of months the "
            "method agrees. The readings in the middle column are a judgment too, so challenge them."))
    st.html(ui.table(
        ["Episode", "Months", "Usual reading", "Labeled here", "Agrees"],
        [[e["episode"], f"{pd.Period(e['from'], 'M').to_timestamp():%b %Y} – "
          f"{pd.Period(e['to'], 'M').to_timestamp():%b %Y}",
          ui.Raw(chip(e["expected"])), ui.Raw(chip(e["labeled"])), f"{e['share']:.0%}"]
         for e in ep.to_dict("records")],
        numeric={4}))
miss = ep[ep["share"] < 0.5] if not FORWARD else ep.iloc[0:0]
for e in miss.to_dict("records"):
    w = lab.loc[pd.Period(e["from"], "M"):pd.Period(e["to"], "M")]
    st.html(f'<p class="mr-caption">{ui.esc(e["episode"])} reads as {NAME[e["labeled"]].lower()}, not '
            f'{NAME[e["expected"]].lower()}: GDP ran {w["growth"].mean():+.2f} points against potential '
            f'and core PCE {w["inflation"].mean():+.2f} points against expectations.</p>')

# ---------- compare methods ----------

cmp = comparison(data_version())
st.html(ui.section_head("Compare labels", caption=
        "Each label with each inflation benchmark, scored the same way. Episodes use the readings in the "
        "table above, and the 1983–86 reading is itself in dispute, so read that column alongside the score."))


def tag(r) -> str:
    this = {"years": r.years, "align": r.align, "anchor": r.anchor}
    tags = [t for t, on in (("adopted", this == truth.DEFAULT), ("shown", this == METHOD)) if on]
    return ANCHOR_NAME[r.anchor] + (f" ({', '.join(tags)})" if tags else "")


now_rows = cmp[cmp["view"] == "now"]
st.html('<p class="mr-caption"><b>Nowcast: the regime at the time</b></p>')
st.html(ui.table(["Inflation benchmark", "Periods", "Median months", "Recession months growth down",
                  "Episodes matched", "Same as adopted", "1983–86", "2023–24", "Provisional months"],
                 [[tag(r), r.periods, f"{r.median:.0f}", f"{r.recessions:.0%}", f"{r.episodes} of {r.of}",
                   f"{r.same:.0%}", ui.Raw(chip(r.mid80s)), ui.Raw(chip(r.soft)), r.provisional]
                  for r in now_rows.itertuples()], numeric={1, 2, 3, 4, 5, 8}))
fwd_rows = cmp[cmp["view"] != "now"]
st.html('<p class="mr-caption" style="margin-top:1rem"><b>Forward: the regime of the years ahead.</b> '
        'Scored against what followed, not against episodes. The band shrinks with the window, '
        f'to ±{truth.band(3):.2f} points for 3 years and ±{truth.band(5):.2f} for 5, since an average over '
        'more years moves less.</p>')
st.html(ui.table(["Window", "Inflation benchmark", "Periods", "Median months", "Labels end",
                  "Matches what followed", "1983–86"],
                 [[WINDOW_NAME[r.years], tag(r), r.periods, f"{r.median:.0f}", when(r.end),
                   f"{r.followed:.0%}", ui.Raw(chip(r.mid80s))] for r in fwd_rows.itertuples()],
                 numeric={2, 3, 5}))

# ---------- periods and months ----------

st.html(ui.section_head("Regime by period", meta=f"{len(sp)} periods, oldest first"))
with st.expander("Show every period with its numbers", expanded=False):
    rows = [[span(r) + (" *" if r.provisional else ""), r.months, ui.Raw(chip(r.label)),
             f"{r.gdp_growth:.2f}% vs {r.potential_growth:.2f}%",
             f"{r.core_pce:.2f}% vs {r.expected:.2f}%",
             f"{r.clear:.0%}",
             (f"{r.recession_months} months of recession. " if r.recession_months else "") + r.note]
            for r in sp.itertuples()]
    st.html(ui.table(["Period", "Months", "Regime", "GDP vs potential", "Core PCE vs expected",
                      "Clear months", "What was going on"], rows, numeric={1, 5}))
    st.html(f'<p class="mr-caption">Inputs are averaged over the period. Clear months: both gaps outside '
            f'±{GB}; the rest carried an earlier reading on at least one axis. * Includes provisional '
            'months. ' + ('Notes are written by hand in config/regime_history.yml.' if IS_DEFAULT else
                          'Notes are written for the adopted method, so they are hidden here.') + '</p>')

COLS = {
    "label": "Regime", "provisional_reading": "Provisional reading", "gdp_growth": "Real GDP growth %", "potential_growth": "CBO potential %",
    "growth": "Growth gap", "core_pce": "Core PCE %", "spf_cpi10": "Survey 10-yr CPI %",
    "wedge": "CPI–PCE gap", "expected": "Expected %", "inflation": "Inflation gap",
    "strength": "Clear or weak", "recession": "Recession", "provisional": "Provisional",
}
monthly = lab[list(COLS)].rename(columns=COLS)
monthly["Regime"] = monthly["Regime"].map(NAME)
monthly["Provisional reading"] = monthly["Provisional reading"].map(NAME).where(lab["provisional"].values, "")
monthly.index = monthly.index.strftime("%Y-%m")
monthly.index.name = "Month"
shown = monthly.loc[str(lo):str(hi)] if sel is not None else monthly
SUFFIX = "" if IS_DEFAULT else f"_{years}y_{align}_{anchor}"
st.html(ui.section_head("Monthly data", meta=f"{len(shown)} months shown"))
with st.expander("Show the monthly inputs" + (" for the period picked" if sel is not None else ""),
                 expanded=sel is not None):
    st.dataframe(shown.round(2), width="stretch", height=360)
    a, b, _ = st.columns([1, 1, 3])
    a.download_button("Download all months (CSV)", monthly.to_csv().encode("utf-8"),
                      file_name=f"regime_history_monthly{SUFFIX}.csv", mime="text/csv", key="dl_months")
    periods = sp.assign(start=sp["start"].astype(str), end=sp["end"].astype(str), label=sp["label"].map(NAME))
    b.download_button("Download periods (CSV)", periods.round(3).to_csv(index=False).encode("utf-8"),
                      file_name=f"regime_history_periods{SUFFIX}.csv", mime="text/csv", key="dl_periods")

# ---------- methodology ----------

st.html(ui.section_head("Methodology in detail"))
st.html(
    '<div class="m-body">'
    "<p>The labels use today's revised data, so they show what happened, not what anyone could see "
    "at the time.</p>"
    "<ol>"
    "<li><b>Growth: real GDP against potential.</b> Real GDP growth over the four quarters centered on "
    "each quarter, minus CBO's potential growth over the same window. Potential is how fast the economy "
    "can grow without overheating, given its workforce, capital and productivity. The gap is placed on "
    "the quarter's middle month and interpolated between quarters. A full-year window smooths out "
    "single-quarter swings in inventories and trade, at the cost of blurring the edges of very short "
    "recessions by a month or so.</li>"
    "<li><b>Inflation: core PCE against expectations.</b> Core PCE inflation over the 12 months centered "
    "on the month, annualized. Core leaves out food and energy, so oil swings alone do not flip the "
    "label. Expected inflation is the median 10-year CPI forecast from the Philadelphia Fed's Survey "
    "of Professional Forecasters, filled from the Blue Chip and Livingston surveys before 1991. CPI "
    "runs above PCE, so the average CPI–PCE gap over the 10 years around each month is subtracted. "
    "Judging inflation against expectations, not a fixed line, means 4% in 1985 counts as low: it was "
    "below what people expected and falling.</li>"
    f"<li><b>Bands.</b> An axis flips only when its gap clears ±{GB} points on the other side of zero. "
    "Inside the band it keeps its previous reading, so a wobble around zero does not flip the label. "
    "Months where either gap sits inside its band are marked weak.</li>"
    "<li><b>Regime.</b> Growth up and inflation low is goldilocks; up and high is high growth, high "
    "inflation; down and high is stagflation; down and low is hard landing.</li>"
    f"<li><b>Persistence.</b> A regime must hold {truth.MIN_MONTHS} months. A shorter spell takes the "
    "label of the period before it.</li>"
    "<li><b>Recessions</b> are not forced into any regime. They appear as growth down because GDP fell "
    "below potential; whether that is stagflation or hard landing depends on inflation.</li>"
    "<li><b>Updates and provisional months.</b> Each data refresh pulls the latest GDP, potential, "
    "core PCE and survey readings and relabels every month from 1980, so revisions flow through. At "
    "the end of the sample a centered window is not complete: the window runs to the latest print "
    "instead, and those months are marked provisional. A provisional month cannot start a new regime; "
    "it keeps the last confirmed one until its window fills. Labels run to the last month GDP "
    "covers.</li>"
    "</ol>"
    "<p><b>Other labels on this page.</b> A forward label marks each month with the regime of the "
    "next three or five years: growth and inflation over the window starting in the month, with "
    "perfect foresight. It is the yardstick for asking what the signposts said about the years ahead, "
    "and it stops a full window before the latest data. Its band is the half-point band divided by "
    "the square root of the years, since a multi-year average moves less. The stepped target replaces "
    "the survey with 4% before 1990, 3% for 1990–95 and 2% from 1996. It reads the mid-1980s as high "
    "inflation, at the cost of step dates fixed by judgment.</p>"
    "<p><b>Not the same as the backtest's current rule.</b> That rule uses a fixed 2.5% core PCE line "
    "and the change in the unemployment gap, and forces every recession to hard landing.</p>"
    "<p><b>Still open:</b> whether inflation should be judged against expectations or a fixed line, "
    "and whether a slowdown without a recession (1995, 2016) should count as hard landing.</p>"
    "</div>")
st.html(ui.table(["Input", "Publisher", "Frequency", "Macrobond code"], [
    ["Real GDP", "BEA", "Quarterly", truth.CODES["gdp"]],
    ["Real potential GDP", "CBO", "Quarterly", truth.CODES["potential"]],
    ["PCE price index excluding food and energy", "BEA", "Monthly", truth.CODES["core_pce"]],
    ["PCE price index, and CPI (for the CPI–PCE gap)", "BEA, BLS", "Monthly",
     f'{truth.CODES["pce"]}, {truth.CODES["cpi"]}'],
    ["Expected CPI inflation over 10 years, median", "Philadelphia Fed; Blue Chip and Livingston before 1991",
     "Quarterly", truth.CODES["spf_cpi10"]],
    ["Recession dates", "NBER", "Monthly", "fixed list in src/backtest.py"],
]))

st.html(f'<div class="mr-foot"><b>Sources:</b> BEA, BLS, CBO, Philadelphia Fed Survey of Professional '
        f'Forecasters and NBER, via Macrobond. Rebuild with <code>python ingest.py truth</code>, then '
        f'publish. {published_note()}</div>')
