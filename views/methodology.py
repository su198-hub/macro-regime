"""Methodology page.

Every table and parameter here is read from the live config and store, so the
page cannot drift from the model the dashboard is running. Prose explains the
fixed mechanics; anything a person might argue about comes from config.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import streamlit as st

from src import ui
from src.drivers import DRIVER_SCALE, config_hash, required_series
from src.regimes import contributions
from src.transform import CLIP
from views.common import REPO_URL, get_results, get_store, is_demo, source_sentence

store = get_store()
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

TRANSFORM_TEXT = ui.TRANSFORM_TEXT


def section(anchor: str, number: int, title: str) -> None:
    st.html(f'<hr class="mr-rule"><h2 class="mr-h2 m-section" id="{anchor}">'
            f'<span style="color:{ui.MUTED};font-weight:400">{number}.</span> {ui.esc(title)}</h2>')


def prose(markup: str) -> None:
    st.html(f'<div class="m-body">{markup}</div>')


def fred_link(sid: str) -> str:
    return (f'<a href="https://fred.stlouisfed.org/series/{ui.esc(sid)}" target="_blank" '
            f'rel="noopener">{ui.esc(sid)}</a>')


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
    ("m-example", "Worked example"),
    ("m-validation", "Validation and track record"),
    ("m-limits", "Limitations"),
    ("m-versions", "Versions and reproducibility"),
    ("m-glossary", "Glossary"),
]

st.html(
    '<p class="mr-eyebrow">Macro regime monitor</p>'
    '<h1 class="mr-title">Methodology</h1>'
    '<p class="mr-sub">How the monitor turns point-in-time US data into a regime call. '
    'Tables and parameters on this page are read from the live configuration, so they '
    'always describe the model the dashboard is running.</p>'
    f'<p class="mr-lede-muted" style="margin-top:0.6rem">Config {chash} · '
    f'{len(required_series(cfg))} source series · latest scored month {latest:%B %Y}</p>'
    '<hr class="mr-rule"><p class="mr-probs-head">On this page</p><ol class="m-toc">'
    + "".join(f'<li><span>{i}.</span><a href="#{a}">{ui.esc(t)}</a></li>'
              for i, (a, t) in enumerate(SECTIONS, 1))
    + "</ol>")

# ---------- 1. summary ----------

section("m-summary", 1, "Summary")
prose(
    '<p>The monitor answers one question: <b>which macro regime do current conditions '
    'most resemble?</b> It does so in five steps.</p>'
    '<ol class="m-steps">'
    '<li><b>Read the data as it was known.</b> Every series is taken as published on the '
    'chosen date, before later revisions.</li>'
    '<li><b>Score each indicator.</b> Each series is transformed, then measured against an '
    'economically meaningful centre, such as the 2% target, giving a signed score.</li>'
    f'<li><b>Build five driver scores.</b> Indicators are averaged by weight into demand, '
    f'inflation expectations, supply constraint, policy stance and investment spending, '
    f'each between −1 and +1.</li>'
    '<li><b>Compare with four regimes.</b> Each regime is a point in driver space. The '
    'closer today\'s five scores sit to a regime, the higher its probability.</li>'
    f'<li><b>Hold the call steady.</b> A new regime is called only after it has led for '
    f'{settings["persistence_months"]} consecutive months.</li></ol>'
    '<div class="m-callout"><p><b>What it is not.</b> The monitor describes current '
    'conditions; it is not a forecast. There are no subjective adjustments: analyst '
    'observations logged on the dashboard are stored alongside the data but never change '
    'a score or the call. The regime archetypes are judgements that have not yet been '
    'validated against history (section 9).</p></div>')

# ---------- 2. framework ----------

section("m-framework", 2, "Framework: drivers and regimes")
prose('<p>Five drivers summarise the macro environment. Each runs from −1 to +1; positive '
      'means hot, tight or restrictive.</p>')
rows = []
for n in driver_names:
    d = cfg["drivers"][n]
    sp = d.get("signpost") or {}
    rows.append([dlabel[n], d.get("description", ""),
                 (sp.get("low") or {}).get("label", "−1"),
                 (sp.get("high") or {}).get("label", "+1"),
                 len(d["indicators"])])
st.html(ui.table(["Driver", "What it measures", "At −1", "At +1", "Indicators"],
                 rows, numeric={4}))

prose('<p>Four regimes are defined by where each expects the drivers to sit. A regime is '
      'called only when one clears the confidence floor; otherwise the call is '
      '<b>transitional</b>.</p>')
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
    'the archetype table in section 7. Codes close together share a slot.</li>'
    '<li><b>The star</b> is the latest month\'s score. <b>The hollow circle</b> is the '
    'score twelve months earlier, as the data is known today.</li>'
    '<li><b>The thin centre line</b> is zero, the neutral reading.</li>'
    + (f'<li><b>{ui.esc(", ".join(reversed_))}</b> is drawn with the tighter end on the '
       f'left. Scores still count restrictive as positive.</li>' if reversed_ else '')
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


def code_cell(r) -> ui.Raw:
    if r.source == "fred":
        return ui.Raw(fred_link(r.source_code or r.series_id))
    return ui.Raw(f'{ui.esc(r.source_code or r.series_id)}<br><span style="color:{ui.INK_2};'
                  f'font-size:0.8rem">{ui.esc(r.source.capitalize())}</span>')


st.html(ui.table(
    ["Series", "Vendor code", "Title", "Frequency", "Vintages", "First", "Last"],
    [[r.series_id, code_cell(r), r.title, r.frequency, f"{r.vintages:,}",
      f"{r.first_obs:%b %Y}", f"{r.last_obs:%b %Y}"] for r in cov.itertuples()],
    numeric={4}))

# ---------- 5. indicators ----------

section("m-indicators", 5, "From series to indicator scores")
prose('<p>Each indicator goes through three steps: a transform, a normalisation that '
      'puts it on a common scale, and a direction that makes positive mean the driver is '
      'being pushed up.</p><h3 class="m-h3">Normalisation</h3>'
      '<p><b>Gap</b> is used wherever a level has economic meaning, such as the 2% target, '
      'long-run average utilisation or a neutral real rate. It says how far the indicator '
      'is from that centre in units of a chosen scale:</p>')
st.latex(r"g_t \;=\; \dfrac{x_t - c}{s}")
prose(f'<p><b>Rolling z-score</b> is used only where there is no meaningful centre. It '
      f'compares the indicator with its own recent history, over a window of <i>w</i> '
      f'months and needing at least max(24, w/4) months of data:</p>')
st.latex(r"z_t \;=\; \dfrac{x_t - \operatorname{mean}_w(x)}{\operatorname{sd}_w(x)}")
prose(f'<p>A gap says <i>tight</i>; a z-score only says <i>unusual</i>. That is why gaps '
      f'are preferred. Both are clipped to ±{CLIP:g} so a single extreme print cannot '
      f'dominate a driver, then multiplied by the direction (+1 or −1).</p>'
      '<h3 class="m-h3">Indicator set</h3>'
      '<p>Weights are shown both as configured and as a share of their driver.</p>')

rows = []
for n in driver_names:
    inds = cfg["drivers"][n]["indicators"]
    total = sum(float(i["weight"]) for i in inds)
    rows.append(dlabel[n])
    for i in inds:
        src = i["source"]
        if src.get("expr"):
            source = ui.esc(src["expr"])
            for sid in sorted(src.get("fred", []), key=len, reverse=True):
                source = source.replace(sid, fred_link(sid))
            for alias, spec in (src.get("derived") or {}).items():
                source += (f'<br><span style="color:{ui.INK_2}">{ui.esc(alias)} = '
                           f'{ui.esc(TRANSFORM_TEXT.get(spec.get("transform", "level"), ""))} '
                           f'of {fred_link(spec["fred"])}</span>')
        else:
            source = fred_link(src["fred"])
        norm = i["normalize"]
        if norm.get("method") == "gap":
            centre = f'{float(norm["center"]):g}'.replace("-", "−")
            norm_text = f'Gap from {centre}, scale {float(norm["scale"]):g}'
        else:
            norm_text = f'Z-score, {int(norm.get("window", 240))}-month window'
        rows.append([
            i.get("label") or i["id"].replace("_", " ").capitalize(), ui.Raw(source),
            TRANSFORM_TEXT.get(i.get("transform", "level"), i.get("transform", "")),
            norm_text, "+1" if int(i["direction"]) > 0 else "−1",
            f'{float(i["weight"]):.2f}', f'{float(i["weight"]) / total:.0%}',
            i.get("why", "")])
st.html(ui.table(["Indicator", "Source", "Transform", "Normalisation", "Direction",
                  "Weight", "Share", "Why it is included"], rows, numeric={4, 5, 6}))

# ---------- 6. drivers ----------

section("m-drivers", 6, "From indicators to driver scores")
prose('<p>A driver score is the weighted mean of its indicator scores, taken over the '
      'indicators that have a value that month:</p>')
st.latex(r"D_t \;=\; \operatorname{clip}\!\left(\frac{1}{%g}\cdot"
         r"\frac{\sum_{i \in A_t} w_i\, g_{i,t}}{\sum_{i \in A_t} w_i},\; -1,\; 1\right)"
         % DRIVER_SCALE)
prose(f'<p>where <i>A<sub>t</sub></i> is the set of indicators available in month <i>t</i>. '
      f'Dividing by {DRIVER_SCALE:g} maps a typical range of indicator scores onto −1 to +1.</p>'
      '<p><b>Renormalising over available indicators.</b> In early history some inputs '
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
      '<p>Each regime is a point in the five-driver space: where that regime expects each '
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

# ---------- 8. worked example ----------

section("m-example", 8, "Worked example")
contrib = contributions(drivers, reg, latest)
dist = np.sqrt(contrib.sum(axis=1))
call = results["calls"].loc[latest]
leader = call["leading"]
pull = contrib.loc[leader].idxmax()
prose(f'<p>The latest month, {latest:%B %Y}, using data as known today. Driver scores are '
      f'in section 6. Each cell below is that driver\'s weighted squared gap to the '
      f'regime; the distance is the square root of the row total.</p>')
st.html(ui.table(
    ["Regime"] + [dlabel[n] for n in driver_names] + ["Distance", "Probability"],
    [[rlabel[r]] + [f"{contrib.at[r, n]:.2f}" for n in driver_names]
     + [f"{dist[r]:.2f}", f"{probs.at[latest, r]:.0%}"] for r in regime_names],
    numeric=set(range(1, len(driver_names) + 3))))
called_text = rlabel.get(call["called"], "transitional")
prose(f'<p>{ui.esc(rlabel[leader])} is closest, at a distance of {dist[leader]:.2f}, '
      f'giving it {probs.at[latest, leader]:.0%}. The driver pulling hardest against it is '
      f'<b>{ui.esc(dlabel[pull].lower())}</b>. After the persistence rule and confidence '
      f'floor, the call is <b>{ui.esc(called_text)}</b>.</p>'
      f'<p>On the dashboard, click any number in "What is pulling the call" to trace it '
      f'further: the driver\'s recent path against that regime, and each indicator\'s '
      f'latest reading, score and contribution to the driver.</p>')

# ---------- 9. validation ----------

section("m-validation", 9, "Validation and track record")
prose('<p>There is no track record yet. The archetypes, weights, temperature and '
      'confidence floor are informed judgements, not estimates. Until they are checked '
      'against history, read the probabilities as a structured way to read the drivers, '
      'not as a measured likelihood.</p>')
st.html(ui.table(["Check", "Status", "Notes"], [
    ["Unit tests on transforms, normalisation and the persistence rule", "In place",
     "Run on every change to the code."],
    ["Point-in-time data", "In place" if not demo else "Built, not yet loaded",
     "Full vintage histories; how far back each goes is in section 4."],
    ["Hand-labelled regime history, 1970 to present", "Not yet",
     "Labelled from what was knowable at the time, not with hindsight. The priority."],
    ["Compare the calls with the labelled history", "Not yet",
     "Depends on the labelled history."],
    ["Sensitivity of the call to weights, temperature and persistence", "Not yet",
     "Should precede any recalibration."],
    ["Statistical alternative, such as a Markov-switching model", "Not yet",
     "Only once there is a labelled history to validate against."],
]))

# ---------- 10. limitations ----------

section("m-limits", 10, "Limitations")
prose(
    '<ul>'
    '<li><b>Unvalidated archetypes.</b> Where each regime sits in driver space is a '
    'judgement (section 9).</li>'
    '<li><b>Stepwise inputs.</b> Quarterly and annual series are carried forward, so their '
    'drivers move in steps. The annual federal deficit adds almost no timely signal.</li>'
    '<li><b>Lagging inputs.</b> Senior Loan Officer lending standards (quarterly) and '
    'CBO\'s estimate of the noncyclical unemployment rate both lag.</li>'
    '<li><b>One policy driver.</b> Monetary and fiscal stance are combined, so offsetting '
    'moves can cancel out.</li>'
    '<li><b>Calibration risk.</b> Tuning thresholds on 2021 to 2023 would overfit an '
    'unusual episode.</li>'
    '<li><b>US only.</b> The indicator set and centres are US-specific.</li>'
    + ('<li><b>Demo data.</b> This deployment runs on synthetic series.</li>' if demo else '')
    + '</ul>')

# ---------- 11. versions ----------

section("m-versions", 11, "Versions and reproducibility")
prose(f'<ul><li><b>Config hash.</b> Every call is stamped with a short fingerprint of the '
      f'indicator configuration. The current one is <code>{chash}</code>. If the hash '
      f'changes, weights or definitions changed.</li>'
      f'<li><b>Rewind.</b> Set "Data as known on" on the dashboard to reproduce what the '
      f'monitor showed on any past date, with the current configuration.</li>'
      f'<li><b>Change history.</b> Code, configuration and this page are versioned in '
      f'<a href="{REPO_URL}/commits/main" target="_blank" rel="noopener">the project '
      f'repository</a>. The weights and archetypes live in '
      f'<a href="{REPO_URL}/blob/main/config/indicators.yml" target="_blank" '
      f'rel="noopener"><code>config/indicators.yml</code></a> and '
      f'<a href="{REPO_URL}/blob/main/config/regimes.yml" target="_blank" '
      f'rel="noopener"><code>config/regimes.yml</code></a>.</li></ul>')

# ---------- 12. glossary ----------

section("m-glossary", 12, "Glossary")
terms = [
    ("Archetype", "The driver scores a regime would have in its purest form."),
    ("Coverage", "The share of a driver's intended indicator weight that had data in a month."),
    ("Driver", "One of five summary dimensions, scored from −1 to +1."),
    ("Gap", "Distance of an indicator from a meaningful centre, in units of a set scale."),
    ("Persistence", "Consecutive months a new leader must hold before the call changes."),
    ("Salience", "How much a driver counts when measuring distance to an archetype."),
    ("Temperature", "Sets how sharply distances turn into probabilities."),
    ("Transitional", "Reported when no regime clears the confidence floor."),
    ("Vintage", "The date a value was published. Revisions create new vintages."),
] + [(reg["regimes"][n].get("short", n), rlabel[n]) for n in regime_names]
st.html('<dl class="m-dl">' + "".join(
    f"<dt>{ui.esc(t)}</dt><dd>{ui.esc(d)}</dd>" for t, d in terms) + "</dl>")

st.html('<div class="mr-foot"><b>Further reading:</b> OECD and European Commission Joint '
        'Research Centre, <i>Handbook on Constructing Composite Indicators: Methodology and '
        'User Guide</i> (2008), for normalisation, weighting and aggregation. ALFRED, '
        'Federal Reserve Bank of St. Louis, for vintage data.</div>')
st.page_link("views/dashboard.py", label="Back to the dashboard",
             icon=":material/arrow_back:")
