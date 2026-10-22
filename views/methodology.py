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
# A few passages quote figures measured on v1 itself, and appear only with it.
IS_V1 = country()["framework"] == "v1"
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
    + f' (section {num("m-limits")}).</p></div>')

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
       f'to that regime. Every month is called the nearest one, unless none reaches the '
       f'{settings["min_confidence"]:.0%} confidence floor, when the call is transitional '
       f'(section {num("m-regimes")}).</p>') if not GATE_ON and HORIZON else
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
    + ('<li><b>Regime codes</b> sit where the driver stood in the run-up to that regime, taken from '
       if HORIZON else '<li><b>Regime codes</b> sit where that regime expects the driver to be, taken from ')
    +
    f'the archetype table in section {num("m-regimes")}. Codes close together share a slot.</li>'
    '<li><b>The solid star</b> is the latest confirmed month\'s score; <b>the outlined star</b>, '
    'when shown, is the provisional month. <b>The hollow circle</b> is the score twelve months '
    'earlier, as the data is known today.</li>'
    '<li><b>The thin center line</b> is zero, the neutral reading.</li>'
    + (f'<li><b>{ui.esc(ui.join_words([reversed_[0]] + [n.lower() for n in reversed_[1:]]))}</b> '
       f'{"is" if len(reversed_) == 1 else "are"} drawn with the tighter end on the left. '
       f'Scores still count restrictive as positive.</li>' if reversed_ else '')
    + '<li>A star sitting near a regime\'s code on most rows is what a high probability '
    'for that regime looks like.</li></ul>')

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
      '<p>One table per driver, its indicators grouped by block, heaviest first. Weight is a '
      'share of the driver. The two correlation columns show how each signed score lined up '
      'with what followed: the average growth gap and inflation gap over the next three years, '
      'from 1971 or the indicator\'s first month to the last month with three full years after '
      'it. Today is the latest confirmed score. Each indicator\'s reason and source are in the '
      'drop-down under its table.</p>')


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
    st.html(ui.table(["Indicator", "Measured as", "Weight", "Next 3y growth", "Next 3y inflation",
                      f"Today ({pd.Period(latest, 'M').to_timestamp():%b %Y})"], rows, numeric={2, 3, 4, 5}))
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
      'and then combined at the block\'s share of the weight. Growth and inflation do this in '
      'v2: a block for the next three years (growth\'s cycle, inflation\'s pressure) and a '
      'long-run block (potential growth; anchoring of expectations). Monetary policy splits '
      'into stance and credibility at equal weight; fiscal policy into impulse, debt dynamics '
      'and composition; reforms into regulation, policy uncertainty, trade policy and public '
      'research, a quarter each. Within a '
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
      f'<li><b>Confidence floor.</b> If no regime reaches {settings["min_confidence"]:.0%}, '
      f'the call is reported as transitional.</li></ul>')
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
        'regime sits nearest the center.'
        + (' Without a check, the slow, disinflationary recovery of 2011 to 2014 came out as '
           'goldilocks.' if IS_V1 else '') + '</p>'
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
        + ('<p>The tolerance is a judgment. With no tolerance at all, 53% of months since 1990 '
           f'fitted no regime, most of them only just, and a monitor that says "no clear regime" '
           f'half the time says little. At {tolerance:.2f}, {unclassified_share:.0%} of months '
           f'since 1990 are unclassified: the shocks are all still called, while the slower '
           f'stretches in between, 2010 to 2014 above all, report no clear regime rather than a '
           f'quiet goldilocks.</p>' if IS_V1 else
           f'<p>The tolerance is a judgment, carried over from v1 and not yet re-tuned. At '
           f'{tolerance:.2f}, {unclassified_share:.0%} of months since 1990 are '
           f'unclassified.</p>')
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

# ---------- 9. limitations ----------

section("m-limits", 9, "Limitations")
# v1's limitations are written here; v2 lists its own under meta.limitations
# in its indicator file.
_own_limits = "".join(f'<li><b>{ui.esc(x["title"])}.</b> {ui.rich(x["text"])}</li>'
                      for x in META.get("limitations") or [])
prose(
    '<ul>' + (_own_limits or
    '<li><b>Archetypes do not match the labeled history.</b> Where each regime sits in driver '
    'space is a judgment, informed but not estimated. Against the labeled history on the Regime '
    'history page (1968 onward, on revised data), the months in each regime sit closer to '
    'neutral than the archetypes, and for stagflation and hard landing the monetary and fiscal '
    'positions point the other way: by the time those regimes arrive the Fed has usually eased '
    'and the deficit widened, where the archetypes place tight money and a fiscal pullback. '
    'They have not been re-placed yet.</li>'
    '<li><b>Stepwise inputs.</b> Quarterly, annual and irregular series, such as unit labor '
    'costs, the forecasters\' ten-year projections, CBO\'s revisions and the age of the capital '
    'stock, are carried forward, so their drivers move in steps.</li>'
    '<li><b>Lagging inputs.</b> CBO\'s estimate of the noncyclical unemployment rate, behind the '
    'unemployment gap and the Taylor rule, is revised slowly and lags.</li>'
    '<li><b>Fiscal is federal and not cyclically adjusted.</b> State and local budgets are '
    'left out, and recessions widen the deficit automatically, which reads as looser '
    'fiscal even when policy has not changed.</li>'
    '<li><b>Fixed centers drift.</b> Where a score is measured against an anchor rather than '
    'against the series\' own past, the anchor can stop describing the economy. Capacity '
    'utilization has averaged 76.6% since 2016, three points below its 79.6% center: either '
    'the economy has changed or the center has, and only judgment separates the two.</li>'
    '<li><b>Four regimes do not cover the space.</b> There is no box for a capex recession '
    'with a resilient consumer, which is why mid-2016 reports as a hard landing, or for a '
    'slow recovery under emergency policy, which is why much of 2010 to 2014 reports as no '
    'clear regime. A fifth regime would close both; it has not been added because it should be '
    'argued from a labeled history rather than fitted to these two episodes.</li>'
    '<li><b>Persistence can hold a call through a turn.</b> The rule that stops the call '
    'flipping on noise also delays it at a genuine turning point. In 2008 the monthly readings '
    'moved to stagflation in July and to a hard landing from August, but the call only became '
    'hard landing in October, once the new leader had held three months in a row.</li>')
    + '<li><b>Calibration risk.</b> Tuning thresholds on 2021 to 2023 would overfit an '
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
    '<p><b>October 2026 — two indicators out (v2).</b> The change in core inflation’s distance '
    'from 2% leaves credibility: its information overlaps the miss rate and the Taylor gap, and '
    'the call did slightly better without it. The Treasury’s primary balance leaves the fiscal '
    'impulse: it overlaps the OECD’s cyclically adjusted balance. Each block keeps its share. '
    '33 indicators; graded on the three years that followed, 1976 to 2023, 68% held out and '
    '57% in real time, against 65% and 56%.</p>'
    '<p><b>October 2026 — core PCE replaces the trimmed mean (v2).</b> Underlying inflation in '
    'the inflation signpost is now core PCE against the forecasters’ ten-year CPI, not the '
    'Dallas Fed trimmed mean. The trimmed mean drops each month’s largest price moves, which '
    'strips out a shock such as tariffs; core keeps them in, which matters when the question '
    'is whether such a shock spreads. On its own the trimmed mean predicted the next three '
    'years slightly better; the model as a whole did slightly better on core: 65% held out '
    'and 56% in real time, 1976 to 2023, against 63% and 55%.</p>'
    '<p><b>October 2026 — four scenarios, and a three-year outlook page (v2).</b> Baseline '
    'leaves the call: it could not be graded against the record, sat in the middle of every '
    'signpost and took about a quarter of the odds. Graded on the three years that followed, '
    '1976 to 2023, the average hit rate across the regimes rose from 59% to 63% held out and '
    'from 41% to 55% in real time. The new Three-year outlook page shows where growth and '
    'inflation are heading, what is pushing them, a policy what-if and the track record.</p>'
    '<p><b>October 2026 — history back to 1968.</b> The labeled history and the signposts now '
    'start in 1968, not 1980. Before the forecasters’ ten-year survey begins in October 1979, '
    'the inflation anchor is the Federal Reserve Board’s FRB/US measure of long-run expected '
    'inflation (PTR, the Kozicki-Tinsley estimate before 1979), joined to the survey on its '
    'basis. Underlying inflation falls back to core PCE before the trimmed mean starts. The '
    'record now holds fifteen run-ups to a regime instead of nine, with three or four of each, '
    'so stagflation can be graded for the first time. The v2 positions were relearned on it. '
    'Graded on the three years that followed, 1976 to 2023, the average hit rate across the '
    'regimes is 59% held out and 41% in real time, or 53% in real time with Baseline calls set '
    'aside.</p>'
    '<p><b>October 2026 — the regime call reads the next three years (v2).</b> Each '
    'scenario’s position is now where the signposts sat in the run-up to that regime, '
    '1983 to 2023, shrunk toward a judgment of where the scenario sits, scaled to the '
    'run-ups’ spread and counted as half an episode. Baseline has no label in the answer key '
    'and takes the average month. The positions sit near the middle, so the temperature is '
    '0.20, the confidence floor 25%, and the fit gate is off. Graded on the three years that '
    'followed, 1993 to 2023, the average hit rate across the regimes is 57% held out and 43% '
    'in real time, against 37% for assuming the current regime continues and 33% for the mix '
    'of regimes seen so far. Before, the same positions as judgments scored 8%: they describe '
    'a regime while it happens, not what comes before it.</p>'
    '<p><b>October 2026 — reforms in four blocks (v2).</b> Reforms now reads whether the policy '
    'environment is turning more or less friendly to long-run growth, rather than how private '
    'investment responds. <b>Regulation</b>: the five-year growth of restrictions in the Code '
    'of Federal Regulations (QuantGov RegData) and the five-year change in the Fraser '
    'Institute’s business regulation score. <b>Policy uncertainty</b>: the Baker, Bloom and '
    'Davis index. <b>Trade policy</b>: the three-year change in the effective tariff rate '
    '(customs duties over goods imports) and trade policy uncertainty. <b>Public research</b>: '
    'the five-year change in federal non-defense R&amp;D as a share of GDP. A quarter each. The '
    'investment measures it replaces mostly tracked the business cycle: an investment share '
    'above its norm came before hard landings. The new signpost goes with faster productivity '
    'over the following decade (+0.47, trend removed). For the next three years it runs the '
    'wrong way, so it takes half weight in the regime call.</p>'
    '<p><b>October 2026 — fiscal policy in three blocks (v2).</b> Fiscal policy now reads how '
    'supportive the budget is, rather than the sustainability of the debt path. <b>Impulse</b>, '
    '37.5%: the three-year change in the OECD’s cyclically adjusted primary balance and in the '
    'Treasury’s primary balance, and federal outlays growth less receipts growth. <b>Debt '
    'dynamics</b>, 37.5%: the 10-year yield less expected nominal growth (trend growth plus '
    'forecasters’ ten-year inflation), and the yearly rise in the interest bill as debt rolls '
    'over at market rates, given its average maturity; both low reading as supportive. These '
    'replaced the effective rate on the debt less past nominal growth, which leaned on old '
    'cheap coupons and on growth lifted by inflation. <b>Composition</b>, 25%: federal investment as a share of non-interest '
    'spending, the long-run part, kept by design. The old multiplier axis is folded into '
    'composition: its three-year shifts did not predict growth, and weighting spending and '
    'taxes by CBO’s multipliers did no better than the plain total. Fiscal policy now takes '
    'half weight in the regime call: its growth signal overlaps the growth signpost’s slack, '
    'and its own contribution is inflation (+0.65 with the next three years’ inflation gap). '
    'Graded on the three years that followed, 1993 to 2023, the average hit rate across the '
    'regimes rose from 42% to 48% held out, with all three hard-landing run-ups caught, and '
    'from 28% to 36% in real time.</p>'
    '<p><b>October 2026 — monetary policy in two blocks (v2).</b> Monetary policy now reads '
    'how hawkish the Fed is, rather than its credibility alone. The <b>stance</b> block, 50%: '
    'the 10-year yield less fed funds, the change in fed funds over three years, and the real '
    'fed funds rate less the neutral rate. The <b>credibility</b> block, 50%: the Taylor-rule '
    'gap, now its current reading rather than a three-year average, the share of months off '
    'target, inflation volatility, inflation’s trajectory and forecaster disagreement. Tight '
    'and credible both count as hawkish: hard landings followed tight policy, and followed '
    'spells of low, steady inflation. Graded on the three years that followed, 1993 to 2023, '
    'growth, inflation and monetary policy together caught all three hard-landing run-ups held '
    'out, against none for growth and inflation alone, with the average hit rate across the '
    'regimes rising from 41% to 51%. With all five signposts the gain is small (41% to 42%), '
    'because fiscal policy and reforms have not been reworked yet.</p>'
    '<p><b>October 2026 — inflation in two blocks (v2).</b> Inflation now reads whether '
    'inflation will run above or below expected inflation over the next three years, and '
    'whether long-run expectations are drifting from 2%. The <b>pressure</b> block, 75%: M2 '
    'growth and house-price growth over three years, and trimmed-mean PCE inflation against '
    'the forecasters’ ten-year CPI forecast. The <b>anchoring</b> block, 25%: the forecasters’ '
    'CPI revision, the 5y5y breakeven, the Cleveland Fed’s ten-year expectation and the '
    'inflation risk premium. Out: core inflation over three years against 2%, which looks '
    'back. Graded on the three years that followed, 1993 to 2023, with each episode held out '
    'of its own fit, the average hit rate across the four regimes rose from 36% to 41%, with '
    'all three hard-landing run-ups still caught; graded in real time it fell from 34% to 29%, '
    'within noise but not an improvement.</p>'
    '<p><b>October 2026 — growth in two blocks (v2).</b> Growth now reads whether growth will '
    'run above or below potential over the next three years, and whether potential growth '
    'itself is rising. The <b>cycle</b> block, 75% of the driver: unemployment against CBO’s '
    'long-run rate, capacity utilization against its 10-year average, and real growth in '
    'company and in household debt over three years. The <b>potential growth</b> block, 25%: '
    'CBO’s potential-growth revision, the forecasters’ productivity revision, the capex '
    'consensus revision and CBO’s labor force revision. Out: the forecasters’ ten-year GDP '
    'revision (a near-copy of the productivity revision), the five-year pace of total factor '
    'and labor productivity (they look back, and strong runs were followed by slower decades) '
    'and CBO’s participation revision (from 2019 only). Graded on the three years that '
    'followed, 1993 to 2023, with each episode held out of its own fit, the average hit rate '
    'across the four regimes rose from 23% to 36%, and all three hard-landing run-ups were '
    'caught, against two.</p>'
    '<p><b>October 2026 — five signposts.</b> The default framework is now v2: growth, read '
    'as long-run prospects; inflation, read as anchoring; monetary policy, read as credibility; '
    'fiscal policy, read as the sustainability of the debt path; and reforms and industrial '
    'policy. Supply is folded into growth, and a Baseline scenario joins the four. The six-driver '
    'framework below remains available as v1 under the Framework menu.</p>'
    '<p><b>October 2026 — reworked after review, and renamed.</b> Three drivers '
    'took new names: demand is now <b>Growth</b>, inflation expectations <b>Inflation</b>, and '
    'investment spending <b>Reforms and industrial policy</b>, whose indicators still measure the '
    'investment rather than the policies. Against the September entry below:</p><ul>'
    '<li><b>Growth:</b> payroll growth and jobless claims out; the forecasters’ revision to '
    'ten-year GDP growth and retailers against consumer staples in.</li>'
    '<li><b>Inflation:</b> the 10-year breakeven out; the forecasters’ ten-year CPI forecast, its '
    'revision and the inflation risk premium in.</li>'
    '<li><b>Supply:</b> oil inventories, productivity and global supply chain pressure out; the '
    'revision to the energy sector’s three-year capex and the CBO’s revisions to the potential '
    'labor force and participation rate in.</li>'
    '<li><b>Monetary:</b> the Taylor gap on expected inflation, the real rate against neutral and '
    'financial conditions out; inflation volatility, forecaster disagreement and the share of '
    'months off target in, as measures of credibility.</li>'
    '<li><b>Fiscal:</b> the primary balance level, debt growth and r minus g out; the three-year '
    'change in the primary balance, interest cost already locked in and sovereign CDS in.</li>'
    '<li><b>Reforms and industrial policy:</b> building permits and bank lending standards out; '
    'the change in the average age of the capital stock and the change in high-yield spreads in.</li>'
    '</ul>'
    '<p><b>September 2026 — longer horizons.</b> The set had been built to read the latest '
    'month and was mostly reporting it. Six inputs were added, each looking past the current '
    'quarter, and four short-horizon ones dropped to make room:</p><ul>'
    '<li><b>Supply:</b> crude oil inventories against their five-year average, and labor '
    'productivity. Energy is the supply shock that has actually caused stagflation; '
    'productivity is the other half of whether capacity can expand.</li>'
    '<li><b>Monetary:</b> the yield curve, and the Fed\'s securities portfolio as a share of '
    'GDP. The rate against a rule says what policy is; the curve says what the market thinks '
    'it will do, and the portfolio is the part of the stance a rate cannot show.</li>'
    '<li><b>Reforms and industrial policy</b> (then Investment): the equity risk premium, the excess bond premium and building '
    'permits — the cost of equity, the price of credit risk, and the longest-leading sector '
    'in the economy, none of which the set had.</li>'
    '<li><b>Fiscal:</b> r minus g, the term that decides whether debt compounds.</li>'
    '<li><b>Growth</b> (then Demand): wage growth, from the employment cost index.</li>'
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
