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
from src.drivers import (DRIVER_SCALE, config_hash, driver_groups, group_scores, indicator_group,
                         indicator_inputs, required_series)
from src.regimes import contributions
from src.transform import CLIP
from views.common import (INDICATORS, REGIMES, REPO_URL, country, country_picker,
                          data_version, get_results, get_store, is_demo, settings_banner,
                          model_vendors, source_sentence)

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
# Words that belong to the framework in view rather than to the page: each
# indicator file says what its signs mean and what it cannot do.
META = cfg.get("meta") or {}
drivers, probs = results["drivers"], results["probabilities"].dropna(how="all")
settings, salience = reg["settings"], reg["driver_salience"]
GATE_ON = str(settings.get("fit_gate", "none")) == "closer_than_neutral"
# A framework whose call reads a period ahead names it (v2: the next three years).
HORIZON = META.get("call_horizon")
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
def series_sources() -> dict:
    """Vendor and whether FRED also publishes it, per series, from sources.yml."""
    import yaml
    try:
        with open("config/sources.yml", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError:
        return {}
    default = cfg.get("default", "fred")
    return {sid: ((entry or {}).get("use", default), bool((entry or {}).get("fred_page")))
            for sid, entry in (cfg.get("series") or {}).items()}


def series_link(sid: str) -> str:
    """A link where a page exists, and the vendor where it does not.

    The question is not who we buy the data from, it is whether a reader has
    somewhere to go and read about it. Macrobond redistributes much of FRED, so
    most of these names are real FRED series and their FRED page documents the
    same underlying data — 18 of the 38 behind the call. The other 20 have no
    FRED entry: invented names for things FRED has no equivalent of, vendor
    series, or series built here. Those get their vendor named instead.

    The page previously linked all 38 to fred.stlouisfed.org, so 20 of the
    links went to a page about a different series, or to nothing.
    """
    vendor, has_page = series_sources().get(sid, ("fred", True))
    if has_page or vendor == "fred":
        return (f'<a href="https://fred.stlouisfed.org/series/{ui.esc(sid)}" target="_blank" '
                f'rel="noopener">{ui.esc(sid)}</a>')
    label = {"macrobond": "Macrobond", "bloomberg": "Bloomberg terminal",
             "ciq": "Capital IQ", "quantgov": "QuantGov", "derived": "built here"}.get(vendor, vendor)
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
head_left.html('<div class="mr-mast bare"><p class="mr-eyebrow">Long-Term Regime Monitor</p>'
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
    (f'<p>The monitor answers one question: <b>which macro regime will {ui.esc(HORIZON)} most '
       'resemble, judging from where the drivers stand now?</b> It does so in six steps.</p>'
       if HORIZON else
       '<p>The monitor answers one question: <b>which macro regime do current conditions '
       'most resemble?</b> It does so in six steps.</p>')
    + '<ol class="m-steps">'
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
    '<div class="m-callout"><p><b>What it is not.</b> '
    + ui.esc(META.get("call_note") or 'The monitor describes current conditions; it is not a '
             'forecast, and it makes no subjective adjustments.') + ' '
    + ui.esc(META.get("summary_note") or
             'The regime archetypes are judgments, and the labeled history shows several placed '
             'further out than the data, with monetary and fiscal reversed for stagflation and '
             'hard landing') .rstrip(".")
    + f' (section {num("m-limits")}).'
    # The call's record is graded on the Regime explorer.
    + (' How well the call has done, against simple benchmarks, is on the '
       '<a href="/outlook" target="_self">Regime explorer</a>.' if HORIZON else '')
    + '</p></div>')

# ---------- 2. framework ----------

section("m-framework", 2, "Framework: drivers and regimes")
prose(f'<p>{n_drivers.capitalize()} drivers summarize the macro environment. Each runs from −1 '
      f'to +1; {ui.esc(META.get("sign") or "positive means hot, tight or restrictive")}. '
      f'{ui.esc(META.get("framework_note") or "")}</p>')
rows = []
for n in driver_names:
    d = cfg["drivers"][n]
    sp = d.get("signpost") or {}
    rows.append([dlabel[n], ui.Raw(ui.rich(d.get("description", ""))),
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
      f'happened. Across the {n_drivers} drivers <b>{_mix["medium"] + _mix["long"]:.0%}</b> of weight '
      f'looks beyond a quarter, and <b>{_mix["long"]:.0%}</b> beyond the business cycle: '
      f'market prices of multi-year risk such as the equity risk premium, the yield curve and '
      f'five-year forward breakevens, and slow measures such as productivity and r minus g. '
      f'Each enters as a change, a spread or a deviation from its own trend, or is read as a '
      f'level on purpose with the reason given, so a long horizon never becomes a standing '
      f'bias by accident.</p>')

prose((f'<p>{n_regimes.capitalize()} regimes are each placed where the drivers sat in the run-up '
       f'to that regime. Every month is called the nearest one, once it has led for '
       f'{settings["persistence_months"]} months in a row (section {num("m-regimes")}).</p>')
      if not GATE_ON and HORIZON else
      (f'<p>{n_regimes.capitalize()} regimes are defined by where each expects the drivers to '
       'sit. A month is called a regime only when it is genuinely close to one; otherwise the call '
       f'is <b>no clear regime</b> (section {num("m-regimes")}). Middling months are common, and '
       'the monitor says so rather than forcing them into the nearest regime.</p>'))
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
    + (('<li><b>Regime codes</b> sit in order along each row, at where the driver stood in the '
        'run-up to that regime (the table in section ' + str(num("m-regimes")) + '). The slots are '
        'evenly spaced either side of neutral, so the gaps show order, not distance; regimes '
        'within 0.03 of each other share a slot.</li>')
       if str(settings.get("signpost_layout", "linear")) == "ordinal" else
       ('<li><b>Regime codes</b> sit where that regime expects the driver to be, taken from the '
        'archetype table in section ' + str(num("m-regimes")) + '. Codes close together share a '
        'slot.</li>'))
    + '<li><b>The solid star</b> is the latest confirmed month\'s score; <b>the outlined star</b>, '
    'when shown, is the provisional month. <b>The hollow circle</b> is the score twelve months '
    'earlier, as the data is known today.</li>'
    '<li><b>The thin center line</b> is zero, the neutral reading.</li>'
    + (f'<li><b>{ui.esc(ui.join_words([reversed_[0]] + [n.lower() for n in reversed_[1:]]))}</b> '
       f'{"is" if len(reversed_) == 1 else "are"} drawn with the tighter end on the left. '
       f'Scores still count restrictive as positive.</li>' if reversed_ else '')
    + ('<li>A star sitting between the same two codes on most rows is what a close call '
       'between those regimes looks like; the probabilities use the actual distances.</li></ul>'
       if str(settings.get("signpost_layout", "linear")) == "ordinal" else
       '<li>A star sitting near a regime\'s code on most rows is what a high probability '
       'for that regime looks like.</li></ul>'))

# ---------- 4. data ----------

section("m-data", 4, "Data and point-in-time vintages")
meta_cfg = cfg.get("meta") or {}
carry = int(meta_cfg.get("carry_forward_periods", 0))
reported = float(meta_cfg.get("min_reported_share", 0))
vendors = model_vendors(store, required_series(cfg))
prose('<p>Every series is stored with the date each value was published, so the monitor can be '
      'rewound to what was known on any date. Sources, the rules for frequency and the latest '
      'month, and every series with its revision history are in the drop-down.</p>')
_data_details = st.expander("Data sources, point-in-time vintages and every series")
with _data_details:
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
    'inflation target, the neutral real rate, the noncyclical unemployment rate. This is the '
    'strongest form, because the score then says '
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
        'just whether rates are high. A core monetary indicator is therefore a <b>Taylor-rule '
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
        'enough to flip the sign of the gap.</p><p>The real rate against neutral and the '
        'Taylor gap read different things. The real rate against neutral is part of '
        '<b>monetary stance</b>, how tight policy is. The Taylor gap is part of <b>monetary '
        'credibility</b>: a Fed holding the rate well below what inflation and slack call for '
        'is behind the curve, and its commitment to 2% is in doubt.</p>')

prose('<h3 class="m-h3">Indicator set</h3>'
      '<p>One table per driver, its indicators grouped by block, heaviest first. Weight is a '
      'share of the driver.</p><ul>'
      '<li><b>Correlation with growth, next 3 years:</b> of each month\'s score with real GDP '
      'growth less potential growth over the following three years.</li>'
      '<li><b>Correlation with inflation, next 3 years:</b> of each month\'s score with core PCE '
      'inflation less expected inflation over the following three years.</li>'
      '<li>Both run from 1971, or the indicator\'s first month, to the last month with three full '
      'years after it. Scores are signed so positive points to the driver\'s +1 end, as named on the chart.</li>'
      '<li><b>Score:</b> the latest confirmed month, in standard deviations from the indicator\'s '
      'norm (capped at ±3); on the driver row, the driver\'s −1 to +1 reading.</li></ul>'
      '<p>Each indicator\'s reason and source are in the drop-down under its table.</p>')


@st.cache_data(ttl=3600, show_spinner="Lining each indicator up against what followed…")
def forward_links(version: str, framework_hash: str, scores: pd.DataFrame,
                  driver_frame: pd.DataFrame, blocks: dict) -> dict:
    """Correlation of every indicator, block and driver score with the next
    three years' growth and inflation gaps, from 1971."""
    from src import truth
    wide = get_store().latest(list(truth.STORE_IDS.values()))
    if wide.empty or not set(truth.STORE_IDS.values()) <= set(wide.columns):
        return {}
    fwd = truth.build(truth.from_store(wide), **truth.METHODS["next3"])[["growth", "inflation"]]

    def link(series: pd.Series) -> tuple[float, float]:
        x = series.copy()
        x.index = pd.DatetimeIndex(x.index).to_period("M")
        e = pd.concat([x.rename("x"), fwd], axis=1).dropna()
        e = e[e.index >= pd.Period("1971-01", "M")]
        if len(e) < 60:
            return (float("nan"), float("nan"))
        return (float(e["x"].corr(e["growth"])), float(e["x"].corr(e["inflation"])))

    out = {col: link(scores[col]) for col in scores.columns}
    out.update({key: link(s_) for key, s_ in blocks.items()})
    out.update({f"driver::{col}": link(driver_frame[col]) for col in driver_frame.columns})
    return out


_scores = results["indicators"]
_blocks = {}
for n in driver_names:
    if len(driver_groups(cfg, n)) > 1:
        _gs = group_scores(cfg, n, _scores)
        for g in _gs.columns:
            _blocks[f"block::{n}::{g}"] = _gs[g]
LINKS = forward_links(data_version(), chash, _scores,
                      drivers[[c for c in driver_names if c in drivers.columns]], _blocks)


def corr_cell(pair, k: int, bold: bool = False):
    v = pair[k] if pair else float("nan")
    text = "–" if pd.isna(v) else signed(v)
    return ui.Raw(f"<b>{text}</b>") if bold else text


def measured_as(i: dict) -> str:
    norm = i["normalize"]
    text = TRANSFORM_TEXT.get(i.get("transform", "level"), i.get("transform", ""))
    if norm.get("method") == "gap":
        center = f'{float(norm["center"]):g}'.replace("-", "−")
        text += f"; center {center}, scale {float(norm['scale']):g}"
    else:
        text += f"; z-score, {int(norm.get('window', 240))}-month window"
    if int(i["direction"]) < 0:
        text += "; sign flipped"
    return text


def block_total(n: str, g: str, blocks: dict, total: float) -> list:
    bl = LINKS.get(f"block::{n}::{g}")
    return [ui.Raw(f"<b>{ui.esc(blocks[g].get('label') or g)} block</b>"), "",
            ui.Raw(f"<b>{blocks[g]['weight'] / total:.0%}</b>"),
            corr_cell(bl, 0, True), corr_cell(bl, 1, True), ""]


horizon_of = ui.horizons(cfg)
_idx = pd.DatetimeIndex(_scores.index)
_today = _scores.index[(_idx.to_period("M") <= pd.Period(latest, "M"))][-1]
for n in driver_names:
    inds = cfg["drivers"][n]["indicators"]
    total = sum(float(i["weight"]) for i in inds)
    dl = LINKS.get(f"driver::{n}")
    dl_text = ("" if not dl or pd.isna(dl[0]) else
               f" Against the next three years: {signed(dl[0])} with the growth gap, "
               f"{signed(dl[1])} with the inflation gap.")
    st.html(f'<h3 class="m-h3">{ui.esc(dlabel[n])}</h3>'
            f'<p style="margin-top:0">{ui.esc(cfg["drivers"][n].get("description", ""))}'
            f'{ui.esc(dl_text)} <span style="color:{ui.INK_2};font-size:0.85rem">'
            f'{ui.esc(ui.horizon_text(cfg, n))}</span></p>')
    blocks = driver_groups(cfg, n)
    block_order = {g: k for k, g in enumerate(blocks)}
    rows, reasons, current = [], [], None
    for i in sorted(inds, key=lambda x: (block_order[indicator_group(x)], -float(x["weight"]))):
        g = indicator_group(i)
        if len(blocks) > 1 and g != current:
            if current is not None:
                rows.append(block_total(n, current, blocks, total))
            current = g
            rows.append(ui.Raw(
                f'<span style="font-weight:600;color:{ui.INK_2}">{ui.esc(blocks[g].get("label") or g)}</span>'
                + (f'<span style="font-weight:400;color:{ui.INK_2};font-size:0.82rem;margin-left:0.4rem">'
                   f'{ui.esc(blocks[g]["description"])}</span>' if blocks[g].get("description") else "")))
        key = f"{n}::{i['id']}"
        now_score = _scores.at[_today, key] if key in _scores.columns else float("nan")
        rows.append([ui.Raw(ui.esc(i.get("label") or i["id"].replace("_", " ").capitalize())
                            + (f'<br><span style="color:{ui.INK_2};font-size:0.8rem">Anchor: needed to '
                               f'confirm a month</span>' if i.get("anchor") else "")),
                     measured_as(i), f'{float(i["weight"]) / total:.0%}',
                     corr_cell(LINKS.get(key), 0), corr_cell(LINKS.get(key), 1),
                     "–" if pd.isna(now_score) else signed(now_score)])
        src = i["source"]
        if src.get("expr"):
            # One pass over whole words, so a series whose name contains another's
            # (DGS10 and DGS1) is linked once.
            links = {s_: series_link(s_) for s_ in src.get("series", [])}
            source = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", lambda m: links.get(m.group(0), m.group(0)),
                            ui.esc(src["expr"]))
        else:
            source = series_link(src["series"])
        reasons.append(f'<li><b>{ui.esc(i.get("label") or i["id"])}</b> '
                       f'{ui.horizon_chip(horizon_of.get(i["id"], "medium"))}<br>'
                       f'{ui.rich(i.get("why", ""))}<br>'
                       f'<span style="color:{ui.INK_2};font-size:0.82rem">Source: {source}</span></li>')
    if len(blocks) > 1 and current is not None:
        rows.append(block_total(n, current, blocks, total))
    today_driver = drivers.at[_today, n] if _today in drivers.index and n in drivers.columns else float("nan")
    rows.append([ui.Raw(f"<b>{ui.esc(dlabel[n])}</b>"), "", "",
                 corr_cell(dl, 0, True), corr_cell(dl, 1, True),
                 ui.Raw("<b>–</b>" if pd.isna(today_driver) else f"<b>{signed(today_driver)}</b>")])
    st.html(ui.table(["Indicator", "Measured as", "Weight",
                      ui.Raw("Correlation with<br>growth, next 3 years"),
                      ui.Raw("Correlation with<br>inflation, next 3 years"),
                      ui.Raw(f"Score,<br>{pd.Period(latest, 'M').to_timestamp():%b %Y}")], rows, numeric={2, 3, 4, 5}))
    with st.expander(f"Why each {dlabel[n].lower()} indicator is included, and its source"):
        st.html('<div class="m-body"><ul>' + "".join(reasons) + "</ul></div>")

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
      'present. The dashboard warns when a driver runs below 75% coverage.</p>'
      '<p><b>Blocks.</b> A driver can split its indicators into blocks, each scored on its own '
      'and then combined at the block\'s share of the weight. Growth and inflation do this: '
      'a block for the next three years (growth\'s cycle, inflation\'s pressure) and a '
      'long-run block (potential growth; anchoring of expectations). Monetary policy is two '
      'drivers rather than two blocks, stance and credibility, because the two move '
      'independently, and so is fiscal policy: spending (impulse and composition blocks) and '
      'debt sustainability; reforms into institutions, regulation and policy uncertainty, a fifth each, and '
      'industrial policy (the tariff rate and public research), two fifths. Within a '
      'block, weights are rescaled over the indicators present, as above. Across blocks, each '
      'block with any reading keeps its full share, so a block whose indicators start late does '
      'not dilute the other. Before any potential-growth indicator exists, growth is the cycle '
      'block alone.</p>')
cov_now = {n: drivers.at[latest, f"{n}__coverage"] for n in driver_names
           if f"{n}__coverage" in drivers.columns}
st.html(ui.table(["Driver", f"Score, {latest:%B %Y}", "Coverage"],
                 [[dlabel[n], signed(drivers.at[latest, n]), f"{cov_now.get(n, float('nan')):.0%}"]
                  for n in driver_names if n in drivers.columns], numeric={1, 2}))

# ---------- 7. regimes ----------

section("m-regimes", 7, "From drivers to a regime call")
prose('<h3 class="m-h3">Archetypes</h3>'
      f'<p>Each regime is a point in the {len(driver_names)}-driver space: '
      + ui.esc(META.get("positions_method") or 'where that regime expects each driver to be.')
      + ' Salience sets how much a driver counts when measuring distance.</p>')
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
      + (f'<li><b>Confidence floor.</b> If no regime reaches {settings["min_confidence"]:.0%}, '
         f'the call is reported as transitional.</li></ul>'
         if float(settings["min_confidence"]) > 1 / len(regime_names) else
         f'<li><b>Confidence floor.</b> {settings["min_confidence"]:.0%}, which with '
         f'{len(regime_names)} regimes never binds: the leader always has at least '
         f'{1 / len(regime_names):.0%}.</li></ul>'))
if str(settings.get("fit_gate", "none")) == "closer_than_neutral":
    tolerance = float(settings.get("fit_tolerance", 1.0))
    # Measured, not remembered: this number moves whenever the indicator set or
    # the tolerance changes, and a stale one here would be the page's own lie.
    _called = results["calls"]["called"].loc["1990":].dropna()
    unclassified_share = (_called == "unclassified").mean() if len(_called) else float("nan")
    _from_neutral = {r: np.sqrt(sum(float(salience.get(n, 1.0)) * float(reg["regimes"][r]["archetype"][n]) ** 2
                                    for n in driver_names)) for r in regime_names}
    _far, _near = max(_from_neutral, key=_from_neutral.get), min(_from_neutral, key=_from_neutral.get)
    thresholds = ", ".join(f'{rlabel[r]} {d:.2f}' for r, d in _from_neutral.items())
    prose(
        '<h3 class="m-h3">Fit gate: no clear regime</h3>'
        '<p>Probabilities are relative. They always add up to 100%, whether or not any regime '
        'describes the month, so on their own they turn every middling month into whichever '
        'regime sits nearest the center.</p>'
        '<p>So each month is also measured against its nearest regime on an absolute yardstick: '
        'how far <b>a neutral economy</b>, with every driver at zero, would sit from that '
        f'regime\'s archetype. Those distances are {ui.esc(thresholds)}. A regime far from '
        f'neutral, like {rlabel[_far].lower()}, accepts months far from neutral; one near it, '
        f'like {rlabel[_near].lower()}, demands a closer match. The month then gets one of three '
        'grades:</p><ul>'
        '<li><b>Clear fit.</b> Closer to the archetype than neutral would be.</li>'
        f'<li><b>Weak fit.</b> Further than neutral, but within {tolerance:.2f} times that '
        'distance. The regime is still called, flagged as a weak fit and drawn in a paler shade '
        'on the history chart.</li>'
        f'<li><b>No fit.</b> Beyond {tolerance:.2f} times the distance. The month is <b>no clear '
        'regime</b>.</li></ul>'
        + f'<p>The tolerance is a judgment, not yet tuned on the three-year call. At '
        f'{tolerance:.2f}, {unclassified_share:.0%} of months since 1990 are unclassified.</p>'
        + '<p>No clear regime goes through the same persistence rule as any regime, so the call '
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
      f'<b>{ui.esc(dlabel[pull].lower())}</b>. After the {"fit gate, " if GATE_ON else ""}persistence rule and confidence '
      f'floor, the call is <b>{ui.esc(called_text)}</b>.</p>'
      f'<p>On the dashboard, click any number in "What is pulling the call" to trace it '
      f'further: the driver\'s recent path against that regime, and each indicator\'s '
      f'latest reading, score and contribution to the driver.</p>')

# ---------- 10. limitations ----------

section("m-limits", 10, "Limitations")
# Listed under meta.limitations in the indicator file, beside what they describe.
prose(
    '<ul>' + "".join(f'<li><b>{ui.esc(x["title"])}.</b> {ui.rich(x["text"])}</li>'
                     for x in META.get("limitations") or [])
    + ('<li><b>Demo data.</b> This deployment runs on synthetic series.</li>' if demo else '')
    + '</ul>')

# ---------- 11. versions ----------

section("m-versions", 11, "Versions and reproducibility")
# What changed, dated. This belongs here rather than as "new" flags on the
# indicator table: a badge decays within weeks, and a reader looking at the
# table wants to know what the model is, not what it was.
st.html('<h3 class="m-h3">What changed</h3>')
# Grades are held out, on the three years that followed: 1976 to 2023 from the
# history back to 1968, 1993 to 2023 before it.
CHANGES = [
    ("Nov 2026", "Debt sustainability adds the primary balance against the level that holds the "
     "debt ratio steady, and net interest as a share of revenue.", "71%, unchanged"),
    ("Nov 2026", "Fiscal policy split into two drivers, a quarter of the weight each: fiscal "
     "spending (impulse and composition) and debt sustainability (forward r − g and repricing). "
     "The two move apart, as stance and credibility do.", "71%, unchanged"),
    ("Nov 2026", "The three-year call is the only framework. The earlier one, v1, which read "
     "current conditions, is no longer offered.", ""),
    ("Nov 2026", "Fiscal impulse adds the one-year change in the primary balance; reforms adds "
     "an institutions block (World Bank governance, from 2001) and regulatory enforcement "
     "(World Justice Project, from 2016). Eight other candidates tested and left out.",
     "71%, unchanged"),
    ("Oct 2026", "Monetary credibility split from stance into its own driver, half the weight "
     "each. It is what tells a stagflation run-up from a hard-landing one.", "71%, from 66%"),
    ("Oct 2026", "A new leader must lead for six months, not three, before the call changes.",
     "66%, from 67%"),
    ("Oct 2026", "The federal investment share read against its own ten-year average, centered "
     "on that gap's 1985–2019 norm.", "67%, from 71%"),
    ("Oct 2026", "The forecasters' ten-year CPI revision leaves the anchoring block.", "68%"),
    ("Oct 2026", "Reforms in three blocks: regulation, policy uncertainty, and industrial policy "
     "(tariffs and non-defense R&D). Trade policy uncertainty dropped.", "68%, from 69%"),
    ("Oct 2026", "Wage growth joins inflation pressure, weighted as forecasters weight it.",
     "71%, from 68%"),
    ("Oct 2026", "Code review: the FRB/US expectations offset corrected and its release dates "
     "fixed. The July 2026 call becomes high growth, high inflation.", "68%"),
    ("Oct 2026", "Two overlapping indicators out: the change in core inflation's distance from "
     "2%, and the Treasury primary balance.", "68%, from 65%"),
    ("Oct 2026", "Core PCE replaces the trimmed mean as underlying inflation, so a shock that "
     "spreads stays in.", "65%, from 63%"),
    ("Oct 2026", "The Baseline scenario dropped: it could not be graded.", "63%, from 59%"),
    ("Oct 2026", "History back to 1968, with the Fed's FRB/US anchor before 1979: fifteen "
     "run-ups instead of nine, so stagflation can be graded.", "59%"),
    ("Oct 2026", "The call reads the next three years: each position is where the drivers sat "
     "in the run-ups to that regime, shrunk toward a judgment.",
     "57% (1993–2023); the judgment positions alone 8%"),
    ("Oct 2026", "The drivers rebuilt one at a time for the three-year call: growth (cycle, "
     "potential growth), inflation (pressure, anchoring), monetary policy (stance, "
     "credibility), fiscal policy (impulse, debt dynamics, composition) and reforms.",
     "48%, from 23% (1993–2023)"),
    ("Oct 2026", "The three-year call (v2) becomes the default; v1 stays under a menu.", ""),
    ("Sep 2026", "v1: six longer-horizon inputs in, four short-horizon ones out.",
     "92% of calls unchanged"),
]
st.html(ui.table(["", "What changed", "Regimes right, average"], [list(c) for c in CHANGES]))
prose('<p>The reasoning behind each choice is in the indicator\'s reason above and in the '
      'project\'s commit history.</p>')
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
