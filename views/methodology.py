"""Methodology page.

Every table and parameter here is read from the live config and store, so the
page cannot drift from the model the dashboard is running. Prose explains the
fixed mechanics; anything a person might argue about comes from config.
"""

from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd
import streamlit as st

from src import ui
from src.drivers import DRIVER_SCALE, config_hash, indicator_inputs, required_series
from src.regimes import contributions
from src.transform import CLIP
from views.common import (INDICATORS, REGIMES, REPO_URL, country, country_picker,
                          get_results, get_store, is_demo, settings_banner,
                          source_sentence)

store = get_store()
here = country()
ind_path, reg_path = INDICATORS(), REGIMES()
results = get_results(dt.date.today())
demo = is_demo(store)

if results is None:
    st.html('<h1 class="mr-title">Methodology</h1><p class="mr-sub">Load data first; '
            'this page is generated from the live model and its data.</p>')
    st.stop()

cfg, reg = results["config"], results["regimes"]
drivers, probs = results["drivers"], results["probabilities"].dropna(how="all")
settings, salience = reg["settings"], reg["driver_salience"]
driver_names = list(cfg["drivers"])
regime_names = list(reg["regimes"])
dlabel = {n: cfg["drivers"][n].get("label", n) for n in driver_names}
rlabel = {n: reg["regimes"][n]["label"] for n in regime_names}
latest = probs.index[-1]
chash = config_hash(cfg)
n_drivers = ui.count_word(len(driver_names))
n_regimes = ui.count_word(len(regime_names))

TRANSFORM_TEXT = ui.TRANSFORM_TEXT


def section(anchor: str, number: int, title: str) -> None:
    number = num(anchor)  # numbered from SECTIONS, so inserting one renumbers the rest
    st.html(f'<h2 class="mr-h2 m-section" id="{anchor}">'
            f'<span><span style="color:{ui.MUTED};font-weight:400">{number}.</span> '
            f'{ui.esc(title)}</span></h2>')


def num(anchor: str) -> int:
    return [a for a, _ in SECTIONS].index(anchor) + 1


def prose(markup: str) -> None:
    st.html(f'<div class="m-body">{markup}</div>')


@st.cache_data(ttl=900, show_spinner=False)
def series_vendors() -> dict:
    """Which vendor each series actually comes from, read from sources.yml."""
    import yaml
    try:
        with open("config/sources.yml", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError:
        return {}
    default = cfg.get("default", "fred")
    return {sid: (entry or {}).get("use", default)
            for sid, entry in (cfg.get("series") or {}).items()}


def series_link(sid: str) -> str:
    """A link only where one exists.

    Series names in indicators.yml are FRED-style whatever the vendor, so that
    an expression reads the same everywhere. That is not a reason to send a
    reader to FRED for a Bloomberg ticker: the page used to link every series
    to fred.stlouisfed.org, which for the Bloomberg and Macrobond ones led to a
    page about something else or to nothing at all. Only FRED series get a
    link; the rest say where they come from.
    """
    vendor = series_vendors().get(sid, "fred")
    if vendor == "fred":
        return (f'<a href="https://fred.stlouisfed.org/series/{ui.esc(sid)}" target="_blank" '
                f'rel="noopener">{ui.esc(sid)}</a>')
    label = {"macrobond": "Macrobond", "bloomberg": "Bloomberg terminal",
             "ciq": "Capital IQ", "derived": "built here"}.get(vendor, vendor)
    return (f'{ui.esc(sid)}<span style="color:{ui.INK_2};font-size:0.8rem"> '
            f'&middot; {ui.esc(label)}</span>')


def signed(x: float) -> str:
    return ui.signed(x)


# ---------- header ----------

SECTIONS = [
    ("m-summary", "Summary"),
    ("m-framework", "Framework: drivers and regimes"),
    ("m-signposts", "Reading the signpost chart"),
    ("m-data", "Data and point-in-time vintages"),
    ("m-indicators", "From series to indicator scores"),
    ("m-drivers", "From indicators to driver scores"),
    ("m-regimes", "From drivers to a regime call"),
    ("m-provisional", "Confirmed call and provisional reading"),
    ("m-example", "Worked example"),
    ("m-limits", "Limitations"),
    ("m-versions", "Versions and reproducibility"),
    ("m-glossary", "Glossary"),
]

st.html('<hr class="mr-mast-rule">')
head_left, head_country = st.columns([3, 1], vertical_alignment="bottom")
head_left.html('<div class="mr-mast bare"><p class="mr-eyebrow">Macro Regime Monitor</p>'
               '<h1 class="mr-title">Methodology</h1></div>')
country_picker(head_country)
st.html(
    f'<p class="mr-sub">How the monitor turns point-in-time {ui.esc(here["label"])} data into '
    'a regime call. '
    'Tables and parameters on this page are read from the live configuration, so they '
    'always describe the model the dashboard is running, including any assumption you '
    'have changed in the control room.</p>'
    f'<p class="mr-lede-muted" style="margin-top:0.6rem">Config {chash} · '
    f'{len(required_series(cfg))} source series · latest scored month {latest:%B %Y}</p>'
    '<hr class="mr-rule"><p class="mr-probs-head">On this page</p><ol class="m-toc">'
    + "".join(f'<li><span>{i}.</span><a href="#{a}">{ui.esc(t)}</a></li>'
              for i, (a, t) in enumerate(SECTIONS, 1))
    + "</ol>")
st.session_state["_mr_page"] = "methodology"
settings_banner()

# ---------- 1. summary ----------

section("m-summary", 1, "Summary")
prose(
    '<p>The monitor answers one question: <b>which macro regime do current conditions '
    'most resemble?</b> It does so in six steps.</p>'
    '<ol class="m-steps">'
    '<li><b>Read the data as it was known.</b> Every series is taken as published on the '
    'chosen date, before later revisions.</li>'
    '<li><b>Score each indicator.</b> Each series is transformed, then measured against a '
    'reference: an economic anchor such as the 2% target, its own value a year ago, or its '
    'own recent average, giving a signed score.</li>'
    f'<li><b>Build {n_drivers} driver scores.</b> Indicators are averaged by weight into '
    f'{ui.esc(ui.join_words([dlabel[n].lower() for n in driver_names]))}, '
    f'each between −1 and +1.</li>'
    f'<li><b>Compare with {n_regimes} regimes.</b> Each regime is a point in driver space. The '
    'closer today\'s driver scores sit to a regime, the higher its probability.</li>'
    f'<li><b>Hold the call steady.</b> A new regime is called only after it has led for '
    f'{settings["persistence_months"]} consecutive months.</li>'
    f'<li><b>Read the newest month early.</b> The call is confirmed only once a month\'s core '
    f'data, including consumer spending, is out. Until then the month gets a provisional '
    f'reading from faster indicators, labeled as such (section {num("m-provisional")}).</li></ol>'
    '<div class="m-callout"><p><b>What it is not.</b> The monitor describes current '
    'conditions; it is not a forecast, and it makes no subjective adjustments. The regime '
    'archetypes are judgments that have not yet been checked against a labeled history '
    f'(section {num("m-limits")}).</p></div>')

# ---------- 2. framework ----------

section("m-framework", 2, "Framework: drivers and regimes")
prose(f'<p>{n_drivers.capitalize()} drivers summarize the macro environment. Each runs from −1 '
      f'to +1; positive means hot, tight or restrictive. Policy is split into monetary and '
      f'fiscal, because the two often pull in opposite directions and regimes differ on each.</p>')
rows = []
for n in driver_names:
    d = cfg["drivers"][n]
    sp = d.get("signpost") or {}
    rows.append([dlabel[n], d.get("description", ""),
                 (sp.get("low") or {}).get("label", "−1"),
                 (sp.get("high") or {}).get("label", "+1"),
                 len(d["indicators"]),
                 ui.Raw(ui.horizon_bar(cfg, n)
                        + f'<br><span style="color:{ui.INK_2};font-size:0.78rem;'
                          f'white-space:nowrap">{ui.esc(ui.horizon_text(cfg, n))}</span>')])
st.html(ui.table(["Driver", "What it measures", "At −1", "At +1", "Indicators", "Horizon mix"],
                 rows, numeric={4}))
_mix = {h: sum(ui.horizon_mix(cfg, n)[h] for n in driver_names) / len(driver_names)
        for h in ui.HORIZON_ORDER}
prose(f'<p>The horizon mix is the share of each driver\'s weight looking out over weeks to a '
      f'quarter ({ui.horizon_chip("short")}), a business cycle of one to three years '
      f'({ui.horizon_chip("medium")}), or longer ({ui.horizon_chip("long")}). A monthly '
      f'monitor has to carry something slower than the news, or it reports only what just '
      f'happened. Across the six drivers <b>{_mix["medium"] + _mix["long"]:.0%}</b> of weight '
      f'looks beyond a quarter, and <b>{_mix["long"]:.0%}</b> beyond the business cycle: '
      f'market prices of multi-year risk such as the equity risk premium, the yield curve and '
      f'five-year forward breakevens, and slow measures such as productivity and r minus g. '
      f'Every one of them enters as a change, a spread or a deviation from its own trend '
      f'rather than as a level, so a long horizon never becomes a standing bias.</p>')

prose(f'<p>{n_regimes.capitalize()} regimes are defined by where each expects the drivers to '
      'sit. A month is called a regime only when it is genuinely close to one; otherwise the call '
      f'is <b>no clear regime</b> (section {num("m-regimes")}). Middling months are common, and '
      'the monitor says so rather than forcing them into the nearest regime.</p>')
chip = lambda n: ui.Raw(
    f'<span class="sp-chip" style="background:{reg["regimes"][n]["color"]};'
    f'color:{ui.text_on(reg["regimes"][n]["color"])}">{ui.esc(reg["regimes"][n].get("short", n))}</span>')
st.html(ui.table(["Code", "Regime", "Description", "What to watch for"],
                 [[chip(n), rlabel[n], reg["regimes"][n].get("description", ""),
                   reg["regimes"][n].get("tell", "")] for n in regime_names]))

# ---------- 3. signposts ----------

section("m-signposts", 3, "Reading the signpost chart")
reversed_ = [dlabel[n] for n in driver_names
             if (cfg["drivers"][n].get("signpost") or {}).get("reverse")]
prose(
    '<ul>'
    '<li><b>Each row is one driver</b>, drawn from one extreme to the other. Red end boxes '
    'mark outcomes that are a risk; green marks outcomes that are good for growth.</li>'
    '<li><b>Regime codes</b> sit where that regime expects the driver to be, taken from '
    f'the archetype table in section {num("m-regimes")}. Codes close together share a slot.</li>'
    '<li><b>The solid star</b> is the latest confirmed month\'s score; <b>the outlined star</b>, '
    'when shown, is the provisional month. <b>The hollow circle</b> is the score twelve months '
    'earlier, as the data is known today.</li>'
    '<li><b>The thin center line</b> is zero, the neutral reading.</li>'
    + (f'<li><b>{ui.esc(ui.join_words([reversed_[0]] + [n.lower() for n in reversed_[1:]]))}</b> '
       f'{"is" if len(reversed_) == 1 else "are"} drawn with the tighter end on the left, as on '
       f'the scenario slide. Scores still count restrictive as positive.</li>' if reversed_ else '')
    + '<li>A star sitting near a regime\'s code on most rows is what a high probability '
    'for that regime looks like.</li></ul>')

# ---------- 4. data ----------

section("m-data", 4, "Data and point-in-time vintages")
meta_cfg = cfg.get("meta") or {}
carry = int(meta_cfg.get("carry_forward_periods", 0))
reported = float(meta_cfg.get("min_reported_share", 0))
vendors = store.sources()
prose(
    '<p><b>Source.</b> ' + source_sentence(vendors) + '</p>'
    '<p><b>Point in time.</b> Every observation is stored with its <i>vintage</i>: the date '
    'that value was published. To view the monitor as of a date, each observation takes '
    'the latest vintage published on or before that date. Later revisions are invisible, '
    'so rewinding the dashboard shows the call that was actually available at the time.</p>'
    + ('<p><b>Before revisions were recorded.</b> Each series has a date from which its '
       'vendor recorded every revision. The history as it stood before that date is treated '
       'as published on that date, not earlier. A rewind to before it therefore sees none of '
       'that series, rather than a revised version presented as if it had been known. '
       'Rewinding far enough back leaves drivers without inputs, and then no regime is '
       'called.</p>' if vendors - {"demo"} else '')
    + '<p><b>Projections.</b> Values dated after the vintage that published them, such as '
    'CBO and OMB projections, are forecasts rather than observations and are not used.</p>'
    f'<p><b>Frequency.</b> Everything is converted to month-end. Daily and weekly series '
    f'take the last value in the month. Quarterly and annual series hold their last value '
    f'until the next release, for at most {carry} of their own periods, which is how an '
    f'analyst would read them; they still move in steps. Beyond that limit a series counts '
    f'as missing, so a discontinued input drops out rather than freezing a driver.</p>'
    f'<p><b>The latest month.</b> Series are released on different days, so the newest '
    f'month is always incomplete. A driver is scored in a month only when at least '
    f'{reported:.0%} of the weight of its indicators that already existed has reported. '
    f'Indicators that had not started yet do not count against a month, so early history '
    f'is kept.</p>'
    '<p><b>Revision history.</b> Market series such as breakevens are rarely revised and '
    'legitimately have few vintages. For revised series, a single vintage means the history '
    'is as revised, not as first published, and a backtest on it overstates what was '
    'knowable.</p>')
cov = store.coverage()
meta = store.con.execute(
    "SELECT series_id, source, source_code, title, frequency FROM series_meta").df()
cov = cov.merge(meta, on="series_id", how="left").fillna(
    {"title": "", "frequency": "", "source": "", "source_code": ""})
# Only what the current configuration asks for. The store keeps series from
# indicators that have since been dropped, and listing them here implies they
# are behind the call.
cov = cov[cov["series_id"].isin(required_series(cfg))]


def code_cell(r) -> ui.Raw:
    if r.source == "fred":
        return ui.Raw(series_link(r.source_code or r.series_id))
    return ui.Raw(f'{ui.esc(r.source_code or r.series_id)}<br><span style="color:{ui.INK_2};'
                  f'font-size:0.8rem">{ui.esc(r.source.capitalize())}</span>')


st.html(ui.table(
    ["Series", "Vendor code", "Title", "Frequency", "Vintages", "First", "Last"],
    [[r.series_id, code_cell(r), r.title, r.frequency, f"{r.vintages:,}",
      f"{r.first_obs:%b %Y}", f"{r.last_obs:%b %Y}"] for r in cov.itertuples()],
    numeric={4}))

# ---------- 5. indicators ----------

section("m-indicators", 5, "From series to indicator scores")
prose('<p>Each indicator goes through three steps: a transform, a normalization that '
      'puts it on a common scale, and a direction that makes positive mean the driver is '
      'being pushed up.</p><h3 class="m-h3">Normalization</h3>'
      '<p>Every score is a distance from a reference point, divided by a scale:</p>')
st.latex(r"g_t \;=\; \dfrac{x_t - c}{s}")
prose(
    '<p>What differs between indicators is what that reference is, and the choice is not a '
    'matter of taste. There are three:</p><ul>'
    '<li><b>A fixed anchor</b>, where a level has economic meaning of its own: the 2% '
    'inflation target, the neutral real rate, the noncyclical unemployment rate, long-run '
    'average capacity utilization. This is the strongest form, because the score then says '
    '<i>tight</i> or <i>loose</i> rather than merely <i>unusual</i>.</li>'
    '<li><b>The series\' own past</b>, for anything that grows: a year-on-year or '
    'twelve-month change is a gap whose reference is the value a year ago. Consumption, '
    'permits, the Fed\'s portfolio and the debt ratio all enter this way.</li>'
    '<li><b>The series\' own recent average</b>, where the level grows with the economy and '
    'no absolute number means anything. Crude inventories are measured against their own '
    'five-year average, which is the convention the EIA publishes.</li></ul>'
    '<p>The second and third forms carry their own defense against a stale reference: it '
    'moves with the data. A fixed anchor does not, so an anchor that drifts away from the '
    'economy quietly biases its driver, and the ones in use are worth re-examining rather '
    'than trusting (section ' + str(num("m-limits")) + ').</p>'
    f'<p>Scores are clipped to ±{CLIP:g} so a single extreme print cannot dominate a driver, '
    f'then multiplied by the direction, +1 or −1. A z-score against a rolling window is '
    f'available for indicators with no meaningful reference at all; none currently needs '
    f'it.</p>')

taylor = next((i for spec in cfg["drivers"].values() for i in spec["indicators"]
               if i["id"] == "taylor_gap"), None)
if taylor:
    d = taylor["source"].get("derived") or {}
    rule = d.get("taylor_rate", {})
    prose(
        '<h3 class="m-h3">Monetary policy: neutral rate and Taylor rule</h3>'
        '<p>"Too hawkish" or "too dovish" is a judgment against what conditions call for, not '
        'just whether rates are high. The core monetary indicator is therefore a <b>Taylor-rule '
        'gap</b>: the fed funds rate less the rate a Taylor rule recommends given inflation and '
        'slack. Below zero, policy is looser than the rule; above zero, tighter.</p>'
        f'<p>As configured, the recommended rate is <code>{ui.esc(rule.get("expr", ""))}</code>'
        + (f', floored at {float(rule["floor"]):g}% so that years at the zero lower bound do not '
           f'read as too hawkish' if "floor" in rule else '')
        + '. The output gap is proxied by twice the gap between CBO\'s noncyclical unemployment '
        'rate and the actual rate, which updates monthly.</p>'
        '<p><b>The neutral rate (r*)</b> cannot be observed and has to be estimated. The monitor '
        'uses the FOMC\'s longer-run fed funds median less the 2% target where it has been '
        'published (2012 on): what policymakers believed at the time, and never revised. Before '
        'that, it uses the New York Fed\'s one-sided Holston-Laubach-Williams estimate, which only '
        'uses data available at each date. Estimates of r* are uncertain by a point or more, '
        'enough to flip the sign of the gap, so a plain real-rate-versus-neutral measure is kept '
        'alongside at a lower weight.</p>')

prose('<h3 class="m-h3">Indicator set</h3>'
      '<p>Ordered within each driver by weight, heaviest first. Weight is shown as a share of '
      'its driver, which is all that matters: a driver divides by the weight of whatever has '
      'reported that month, so only the ratios do any work.</p>')

rows = []
horizon_of = ui.horizons(cfg)
for n in driver_names:
    inds = cfg["drivers"][n]["indicators"]
    total = sum(float(i["weight"]) for i in inds)
    # The driver's horizon mix sits on its heading row, so the shape of each
    # block is visible above the indicators that make it up.
    rows.append(ui.Raw(
        f'{ui.esc(dlabel[n])} <span style="font-weight:400;margin-left:0.5rem">'
        f'{ui.horizon_bar(cfg, n, width=90)}</span>'
        f'<span style="font-weight:400;color:{ui.INK_2};font-size:0.82rem;margin-left:0.4rem">'
        f'{ui.esc(ui.horizon_text(cfg, n))}</span>'))
    # Heaviest first, so the table reads as what each driver leans on rather
    # than the order the indicators happen to sit in the config file.
    for i in sorted(inds, key=lambda x: float(x["weight"]), reverse=True):
        src = i["source"]
        if src.get("expr"):
            # The expression, with its series linked. Intermediate aliases used
            # to be spelled out here and swamped the row; the Taylor rule, the
            # only one complicated enough to need it, is written out in prose
            # above, and the rest are in the config file this page links to.
            # One pass over whole words. Replacing series one at a time breaks
            # when one name contains another: linking DGS10 first, then DGS1,
            # rewrote the markup of the link just inserted.
            links = {s: series_link(s) for s in src.get("series", [])}
            source = re.sub(r"[A-Za-z_][A-Za-z0-9_]*",
                            lambda m: links.get(m.group(0), m.group(0)),
                            ui.esc(src["expr"]))
        else:
            source = series_link(src["series"])
        norm = i["normalize"]
        if norm.get("method") == "gap":
            center = f'{float(norm["center"]):g}'.replace("-", "−")
            norm_text = f'Gap from {center}, scale {float(norm["scale"]):g}'
        else:
            norm_text = f'Z-score, {int(norm.get("window", 240))}-month window'
        rows.append([
            ui.Raw(ui.esc(i.get("label") or i["id"].replace("_", " ").capitalize())
                   + (f'<br><span style="color:{ui.INK_2};font-size:0.8rem">Anchor: needed to '
                      f'confirm a month</span>' if i.get("anchor") else "")),
            ui.Raw(source),
            TRANSFORM_TEXT.get(i.get("transform", "level"), i.get("transform", "")),
            norm_text, "+1" if int(i["direction"]) > 0 else "−1",
            ui.Raw(ui.horizon_chip(horizon_of.get(i["id"], "medium"))),
            f'{float(i["weight"]) / total:.0%}',
            i.get("why", "")])
st.html(ui.table(["Indicator", "Source", "Transform", "Normalization", "Direction",
                  "Horizon", "Weight", "Why it is included"], rows, numeric={4, 5, 6}))

# ---------- 6. drivers ----------

section("m-drivers", 6, "From indicators to driver scores")
prose('<p>A driver score is the weighted mean of its indicator scores, taken over the '
      'indicators that have a value that month:</p>')
st.latex(r"D_t \;=\; \operatorname{clip}\!\left(\frac{1}{%g}\cdot"
         r"\frac{\sum_{i \in A_t} w_i\, g_{i,t}}{\sum_{i \in A_t} w_i},\; -1,\; 1\right)"
         % DRIVER_SCALE)
prose(f'<p>where <i>A<sub>t</sub></i> is the set of indicators available in month <i>t</i>. '
      f'Dividing by {DRIVER_SCALE:g} maps a typical range of indicator scores onto −1 to +1.</p>'
      '<p><b>Renormalizing over available indicators.</b> In early history some inputs '
      'do not yet exist. Rather than let a driver drift toward zero, weights are rescaled '
      'over what is present. The cost is that early scores rest on fewer inputs, so '
      'coverage is tracked: the share of the driver\'s intended weight that was actually '
      'present. The dashboard warns when a driver runs below 75% coverage.</p>')
cov_now = {n: drivers.at[latest, f"{n}__coverage"] for n in driver_names
           if f"{n}__coverage" in drivers.columns}
st.html(ui.table(["Driver", f"Score, {latest:%B %Y}", "Coverage"],
                 [[dlabel[n], signed(drivers.at[latest, n]), f"{cov_now.get(n, float('nan')):.0%}"]
                  for n in driver_names if n in drivers.columns], numeric={1, 2}))

# ---------- 7. regimes ----------

section("m-regimes", 7, "From drivers to a regime call")
prose('<h3 class="m-h3">Archetypes</h3>'
      f'<p>Each regime is a point in the {len(driver_names)}-driver space: where that regime expects each '
      'driver to be. Salience sets how much a driver counts when measuring distance.</p>')
st.html(ui.table(
    ["Regime"] + [dlabel[n] for n in driver_names],
    [[rlabel[r]] + [signed(reg["regimes"][r]["archetype"][n]) for n in driver_names]
     for r in regime_names]
    + [[ui.Raw("<b>Salience</b>")] + [f'{float(salience.get(n, 1.0)):.1f}' for n in driver_names]],
    numeric=set(range(1, len(driver_names) + 1))))
prose('<h3 class="m-h3">Distance and probability</h3>'
      '<p>The distance from this month\'s scores to regime <i>k</i> weights each driver '
      'by its salience <i>λ</i>:</p>')
st.latex(r"d_k \;=\; \sqrt{\sum_{j} \lambda_j \,\bigl(D_j - a_{k,j}\bigr)^2}")
prose(f'<p>Distances become probabilities with a softmax. The temperature <i>T</i> = '
      f'{settings["temperature"]} sets how sharp the call is: lower values push more '
      f'probability onto the nearest regime.</p>')
st.latex(r"p_k \;=\; \dfrac{\exp(-d_k / T)}{\sum_m \exp(-d_m / T)}")
prose(f'<p>A month with any driver missing gets no probabilities rather than a guess.</p>'
      f'<h3 class="m-h3">Making the call</h3><ul>'
      f'<li><b>Persistence.</b> The leading regime becomes the call only after leading for '
      f'{settings["persistence_months"]} consecutive months. This stops the call flipping '
      f'on noise.</li>'
      f'<li><b>Confidence floor.</b> If no regime reaches {settings["min_confidence"]:.0%}, '
      f'the call is reported as transitional.</li></ul>')
if str(settings.get("fit_gate", "none")) == "closer_than_neutral":
    tolerance = float(settings.get("fit_tolerance", 1.0))
    # Measured, not remembered: this number moves whenever the indicator set or
    # the tolerance changes, and a stale one here would be the page's own lie.
    _called = results["calls"]["called"].loc["1990":].dropna()
    unclassified_share = (_called == "unclassified").mean() if len(_called) else float("nan")
    thresholds = ", ".join(
        f'{rlabel[r]} {np.sqrt(sum(float(salience.get(n, 1.0)) * float(reg["regimes"][r]["archetype"][n]) ** 2 for n in driver_names)):.2f}'
        for r in regime_names)
    prose(
        '<h3 class="m-h3">Fit gate: no clear regime</h3>'
        '<p>Probabilities are relative. They always add up to 100%, whether or not any regime '
        'describes the month, so on their own they turn every middling month into whichever '
        'regime sits nearest the center. Without a check, the slow, disinflationary recovery of '
        '2011 to 2014 came out as goldilocks.</p>'
        '<p>So each month is also measured against its nearest regime on an absolute yardstick: '
        'how far <b>a neutral economy</b>, with every driver at zero, would sit from that '
        f'regime\'s archetype. Those distances are {ui.esc(thresholds)}. A regime far from '
        'neutral, like a hard landing, accepts months far from neutral; one near it, like '
        'goldilocks, demands a closer match. The month then gets one of three grades:</p><ul>'
        '<li><b>Clear fit.</b> Closer to the archetype than neutral would be.</li>'
        f'<li><b>Weak fit.</b> Further than neutral, but within {tolerance:.2f} times that '
        'distance. The regime is still called, flagged as a weak fit and drawn in a paler shade '
        'on the history chart.</li>'
        f'<li><b>No fit.</b> Beyond {tolerance:.2f} times the distance. The month is <b>no clear '
        'regime</b>.</li></ul>'
        '<p>The tolerance is a judgment. With no tolerance at all, 53% of months since 1990 '
        f'fitted no regime, most of them only just, and a monitor that says "no clear regime" '
        f'half the time says little. At {tolerance:.2f}, {unclassified_share:.0%} of months '
        f'since 1990 are unclassified: the shocks are all still called, while the slower '
        f'stretches in between, 2010 to 2014 above all, report no clear regime rather than a '
        f'quiet goldilocks.</p>'
        '<p>No clear regime goes through the same persistence rule as any regime, so the call '
        'moves to and from it deliberately. It is different from transitional: transitional '
        'means regimes are close to one another; no clear regime means none of them fits.</p>')

# ---------- 8. worked example ----------

section("m-provisional", 0, "Confirmed call and provisional reading")
prov_cfg = meta_cfg.get("provisional") or {}
anchor_names = [i.get("label") or i["id"] for spec in cfg["drivers"].values()
                for i in spec["indicators"] if i.get("anchor")]
# Read the fast inputs off the data rather than a list kept by hand, which went
# stale the moment an indicator was swapped out.
_freq = results.get("frequency", {})
_inputs = indicator_inputs(cfg)
timely_names = [i.get("label") or i["id"] for driver, spec in cfg["drivers"].items()
                for i in spec["indicators"]
                if max((_freq.get(s, 12) for s in _inputs.get(f"{driver}::{i['id']}", [])),
                       default=12) > 12]
prose(
    '<p>Releases arrive weeks apart. Consumer spending and core PCE inflation come about four '
    'weeks after a month ends; jobless claims, surveys and market prices come within days. The '
    'monitor separates what is settled from what is early, and never mixes the two.</p>'
    '<h3 class="m-h3">The confirmed call</h3>'
    f'<p>A month is confirmed only when, for every driver, at least {reported:.0%} of the '
    f'weight of its already-started indicators has reported <b>and</b> every anchor has '
    f'reported. Anchors are the hard data the call should never go without:</p><ul>'
    + "".join(f"<li>{ui.esc(n)}</li>" for n in anchor_names)
    + '</ul><p>The persistence rule and confidence floor apply only to confirmed months, and '
    'the history chart shows only confirmed months.</p>'
    '<h3 class="m-h3">The provisional reading</h3>'
    f'<p>For months after the latest confirmed one, the same model runs on whatever has been '
    f'released. An indicator not yet released carries its latest score for up to '
    f'{int(prov_cfg.get("fill_months", 2))} months, so a driver keeps its usual mix of inputs '
    f'rather than resting on whichever arrived first. A month is shown only once at least '
    f'{float(prov_cfg.get("min_reported_share", 0.5)):.0%} of its data, averaged across drivers, '
    f'has been released. It gets probabilities but no call: no persistence rule, no floor.</p>'
    '<p>Faster indicators were added at small weights so the provisional reading has something '
    'real to read. They also enter confirmed months, where they are a minority of each '
    'driver\'s weight:</p><ul>'
    + "".join(f"<li>{ui.esc(n)}</li>" for n in timely_names)
    + '</ul>'
    '<h3 class="m-h3">How users are kept informed</h3><ul>'
    '<li>The dashboard headline is always the confirmed call. The provisional reading sits '
    'beneath it, marked provisional, with the share of data behind it and when the month '
    'should be confirmed.</li>'
    '<li>Probabilities show the confirmed month as bars and the provisional month as a thin '
    'tick and a separate column. On the signpost chart the confirmed reading is a solid star, '
    'the provisional one an outlined star.</li>'
    '<li>The Data status tab lists every input for the provisional month: released, carried '
    'forward, or not yet released, with an expected release date.</li></ul>'
    '<p><b>Expected release dates</b> are estimates: the median time between the end of a '
    'period and first publication over each series\' last 24 releases, measured from its own '
    'vintages. Holidays and schedule changes can move them by days.</p>')

section("m-example", 8, "Worked example")
contrib = contributions(drivers, reg, latest)
dist = np.sqrt(contrib.sum(axis=1))
call = results["calls"].loc[latest]
leader = call["leading"]
pull = contrib.loc[leader].idxmax()
prose(f'<p>The latest month, {latest:%B %Y}, using data as known today. Driver scores are '
      f'in section {num("m-drivers")}. Each cell below is that driver\'s weighted squared gap to the '
      f'regime; the distance is the square root of the row total.</p>')
st.html(ui.table(
    ["Regime"] + [dlabel[n] for n in driver_names] + ["Distance", "Probability"],
    [[rlabel[r]] + [f"{contrib.at[r, n]:.2f}" for n in driver_names]
     + [f"{dist[r]:.2f}", f"{probs.at[latest, r]:.0%}"] for r in regime_names],
    numeric=set(range(1, len(driver_names) + 3))))
called_text = ui.call_label(call["called"], reg)
fit_text = ""
if pd.notna(call.get("threshold")):
    fit_text = (f' A neutral economy would sit {call["threshold"]:.2f} from it, and the tolerance '
                f'allows up to {call["limit"]:.2f}, so the month is '
                + {"clear": "a clear fit.", "weak": "a weak fit."}.get(call.get("fit"), "no fit."))
prose(f'<p>{ui.esc(rlabel[leader])} is closest, at a distance of {dist[leader]:.2f}, '
      f'giving it {probs.at[latest, leader]:.0%}.{fit_text} The driver pulling hardest against it is '
      f'<b>{ui.esc(dlabel[pull].lower())}</b>. After the fit gate, persistence rule and confidence '
      f'floor, the call is <b>{ui.esc(called_text)}</b>.</p>'
      f'<p>On the dashboard, click any number in "What is pulling the call" to trace it '
      f'further: the driver\'s recent path against that regime, and each indicator\'s '
      f'latest reading, score and contribution to the driver.</p>')

# ---------- 9. limitations ----------

section("m-limits", 9, "Limitations")
prose(
    '<ul>'
    '<li><b>Unvalidated archetypes.</b> Where each regime sits in driver space is a '
    'judgment, informed but not estimated, and not yet checked against a labeled '
    'history.</li>'
    '<li><b>Stepwise inputs.</b> Quarterly and annual series are carried forward, so their '
    'drivers move in steps. The annual federal deficit adds almost no timely signal.</li>'
    '<li><b>Lagging inputs.</b> Senior Loan Officer lending standards (quarterly) and '
    'CBO\'s estimate of the noncyclical unemployment rate both lag.</li>'
    '<li><b>Fiscal is federal and not cyclically adjusted.</b> State and local budgets are '
    'left out, and recessions widen the deficit automatically, which reads as looser '
    'fiscal even when policy has not changed.</li>'
    '<li><b>Fixed centers drift.</b> Where a score is measured against an anchor rather than '
    'against the series\' own past, the anchor can stop describing the economy. Capacity '
    'utilization has averaged about a point below its 1990 to 2019 center since 2016, and the '
    'primary balance about a point below its own: either the economy has changed or those '
    'centers have, and only judgment separates the two.</li>'
    '<li><b>Four regimes do not cover the space.</b> There is no box for a capex recession '
    'with a resilient consumer, which is why mid-2016 reports as a hard landing, or for a '
    'slow recovery under emergency policy, which is why 2010 to 2014 reports as no clear '
    'regime. A fifth regime would close both; it has not been added because it should be '
    'argued from a labeled history rather than fitted to these two episodes.</li>'
    '<li><b>Persistence can hold a call through a turn.</b> The rule that stops the call '
    'flipping on noise also delays it at a genuine turning point. In mid-2008 the monthly '
    'readings had already moved to stagflation and then to a hard landing while the called '
    'regime was still goldilocks, because the challengers alternated and neither held three '
    'months in a row.</li>'
    '<li><b>Calibration risk.</b> Tuning thresholds on 2021 to 2023 would overfit an '
    'unusual episode.</li>'
    + ('<li><b>Demo data.</b> This deployment runs on synthetic series.</li>' if demo else '')
    + '</ul>')

# ---------- 11. versions ----------

section("m-versions", 11, "Versions and reproducibility")
# What changed, dated. This belongs here rather than as "new" flags on the
# indicator table: a badge decays within weeks, and a reader looking at the
# table wants to know what the model is, not what it was.
prose(
    '<h3 class="m-h3">What changed</h3>'
    '<p><b>September 2026 — longer horizons.</b> The set had been built to read the latest '
    'month and was mostly reporting it. Six inputs were added, each looking past the current '
    'quarter, and four short-horizon ones dropped to make room:</p><ul>'
    '<li><b>Supply:</b> crude oil inventories against their five-year average, and labor '
    'productivity. Energy is the supply shock that has actually caused stagflation; '
    'productivity is the other half of whether capacity can expand.</li>'
    '<li><b>Monetary:</b> the yield curve, and the Fed\'s securities portfolio as a share of '
    'GDP. The rate against a rule says what policy is; the curve says what the market thinks '
    'it will do, and the portfolio is the part of the stance a rate cannot show.</li>'
    '<li><b>Investment:</b> the equity risk premium, the excess bond premium and building '
    'permits — the cost of equity, the price of credit risk, and the longest-leading sector '
    'in the economy, none of which the set had.</li>'
    '<li><b>Fiscal:</b> r minus g, the term that decides whether debt compounds.</li>'
    '<li><b>Demand:</b> wage growth, from the employment cost index.</li>'
    '<li><b>Dropped:</b> the weekly economic index, NFIB capex plans, the Philadelphia and '
    'Empire State capex surveys, manufacturing construction and prime-age employment.</li>'
    '</ul>'
    '<p>Judgment calls behind these are in each indicator\'s reason above; the effect on the '
    'history is that 92% of monthly calls were unchanged.</p>')
prose(f'<ul><li><b>Config hash.</b> Every call is stamped with a short fingerprint of the '
      f'indicator configuration. The current one is <code>{chash}</code>. If the hash '
      f'changes, weights or definitions changed.</li>'
      f'<li><b>Rewind.</b> Set "Data as known on" on the dashboard to reproduce what the '
      f'monitor showed on any past date, with the current configuration.</li>'
      f'<li><b>Change history.</b> Code, configuration and this page are versioned in '
      f'<a href="{REPO_URL}/commits/main" target="_blank" rel="noopener">the project '
      f'repository</a>. The weights and archetypes live in '
      f'<a href="{REPO_URL}/blob/main/{ind_path}" target="_blank" '
      f'rel="noopener"><code>{ui.esc(ind_path)}</code></a> and '
      f'<a href="{REPO_URL}/blob/main/{reg_path}" target="_blank" '
      f'rel="noopener"><code>{ui.esc(reg_path)}</code></a>, one set per country, '
      f'listed in <a href="{REPO_URL}/blob/main/config/countries.yml" target="_blank" '
      f'rel="noopener"><code>config/countries.yml</code></a>.</li></ul>')

# ---------- 12. glossary ----------

section("m-glossary", 12, "Glossary")
terms = [
    ("Archetype", "The driver scores a regime would have in its purest form."),
    ("Coverage", "The share of a driver's intended indicator weight that had data in a month."),
    ("Driver", f"One of {n_drivers} summary dimensions, scored from −1 to +1."),
    ("Gap", "Distance of an indicator from a meaningful center, in units of a set scale."),
    ("Persistence", "Consecutive months a new leader must hold before the call changes."),
    ("Salience", "How much a driver counts when measuring distance to an archetype."),
    ("Temperature", "Sets how sharply distances turn into probabilities."),
    ("Transitional", "Reported when no regime clears the confidence floor."),
    ("No clear regime", "Reported when conditions are too far from every regime to fit any, "
                        "beyond the tolerance around a neutral economy's distance."),
    ("Weak fit", "A called regime that conditions fit only loosely: further from its archetype "
                 "than a neutral economy, but within the tolerance."),
    ("Vintage", "The date a value was published. Revisions create new vintages."),
    ("Anchor", "An indicator a month cannot be confirmed without, such as consumer spending."),
    ("Neutral rate (r*)", "The real interest rate that neither stimulates nor restrains the "
                          "economy. Estimated, not observed."),
    ("Taylor rule", "A benchmark policy rate set from the neutral rate, inflation's distance "
                    "from target and the output gap. The Taylor gap is the actual rate less it."),
    ("Confirmed", "A month whose core data is in. Only confirmed months get a regime call."),
    ("Provisional reading", "An early read of a month not yet confirmed, from the data released "
                            "so far. It has probabilities but no call."),
    ("Carried forward", "An input not yet released for a month, held at its latest value in a "
                        "provisional reading until it arrives."),
] + [(reg["regimes"][n].get("short", n), rlabel[n]) for n in regime_names]
st.html('<dl class="m-dl">' + "".join(
    f"<dt>{ui.esc(t)}</dt><dd>{ui.esc(d)}</dd>" for t, d in terms) + "</dl>")

st.html('<div class="mr-foot"><b>Further reading:</b> OECD and European Commission Joint '
        'Research Centre, <i>Handbook on Constructing Composite Indicators: Methodology and '
        'User Guide</i> (2008), for normalization, weighting and aggregation. ALFRED, '
        'Federal Reserve Bank of St. Louis, for vintage data.</div>')
st.page_link("views/dashboard.py", label="Back to the dashboard",
             icon=":material/arrow_back:")
