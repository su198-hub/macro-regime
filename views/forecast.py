"""The three-year page: what is already determined, and what would change it.

PROTOTYPE. The shape is meant to be argued with.

WHY THIS IS NOT THE DASHBOARD WITH A LONGER ARROW. The dashboard asks what
regime we are in this month and answers it by scoring six drivers and matching
them to archetypes. Pointing that same machinery three years out does not work,
and the reason is measurable rather than philosophical: simulate this exact test
on 55 years of monthly data and a TRUE three-year correlation of 0.3 is found
only 38% of the time. Almost nothing can be shown to predict three years ahead,
so a page that produced a regime call for 2029 would be presenting noise with a
confidence interval wide enough to contain the opposite answer.

WHAT IT DOES INSTEAD. Three things, in descending order of how much they can be
trusted:

  1. ARITHMETIC. Quantities already committed -- interest costs locked into the
     debt schedule, capacity being built now that arrives later, the workforce
     demographics already imply. These need no backtest because they are not
     forecasts; they are accounting for decisions already taken.
  2. REVERSION. Measures that are stretched against their own history and have
     historically come back. Weaker than arithmetic, because it rests on a
     statistical regularity rather than a mechanism, and labelled as such.
  3. LEVERS. For each driver, what would have to change, and which indicator
     would show it first. This is where a scenario gets built, and it carries
     no claim about what WILL happen.

The same six drivers as the dashboard, then, but only the part of each that can
speak about years rather than months. Most of the dashboard's weight is
deliberately short-horizon -- credit spreads, orders, surveys -- and that
content is excellent at one year and silent at three. This page shows what is
left when you take it away, which for two drivers is a great deal and for two
others is almost nothing. Saying which is which is the point.
"""

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from src import ui
from src.drivers import compute, load_config
from views.common import INDICATORS, get_store, country, settings_banner

cfg = load_config(INDICATORS())
store = get_store()
res = compute(store, cfg, dt.date.today())
drivers, ind = res["drivers"], res["indicators"]
horizon = ui.horizons(cfg)

st.html(f'<div class="mr-head"><h1 class="mr-h1">The next three years</h1>'
        f'<p class="mr-sub">What is already determined, what is stretched, and what '
        f'would have to change — for {ui.esc(country()["label"])}</p></div>')
st.html('<hr class="mr-rule">')
settings_banner()

st.html(
    '<div class="m-body" style="max-width:46rem">'
    '<p><b>This page does not forecast the regime.</b> It cannot: simulate the '
    'three-year test on 55 years of monthly data and a true correlation of 0.3 is '
    'detected 38% of the time. Anything presented here as a prediction of 2029 would '
    'be noise wearing an interval wide enough to contain the opposite answer.</p>'
    '<p>What it shows is the part of each driver that speaks in years rather than '
    'months, in descending order of trust: <b>arithmetic</b> already committed, '
    '<b>reversion</b> from levels that are historically stretched, and the '
    '<b>levers</b> a scenario would move.</p></div>')


# ---------- 1. arithmetic ----------

def latest(col: str):
    s = ind.get(col)
    return None if s is None else s.dropna()


st.html(ui.section_head(
    "1 · Already committed",
    caption="Quantities determined by decisions already taken. No backtest is offered "
            "because none is needed: these are not forecasts but accounting for things "
            "that have happened."))

rows = []

li = latest("fiscal::locked_in_refinancing")
if li is not None and len(li):
    d = store.as_of(["DEBTPUB", "GDPNOM", "MTSNETINT", "AVGMAT", "DGS5"], dt.date.today())
    for c in d.columns:
        d[c] = d[c].astype(float)
    d.index = pd.to_datetime(d.index)
    debt = d["DEBTPUB"].dropna().resample("MS").last().ffill(limit=3)
    gdpn = d["GDPNOM"].dropna().resample("MS").last().ffill(limit=3)
    int12 = d["MTSNETINT"].dropna().resample("MS").last().rolling(12, min_periods=12).sum()
    mat = d["AVGMAT"].dropna().iloc[-1]
    y5 = d["DGS5"].dropna().iloc[-1]
    eff = float(100 * int12.dropna().iloc[-1] / debt.dropna().iloc[-1])
    share = float(debt.dropna().iloc[-1] / gdpn.dropna().iloc[-1])
    burden = float(100 * int12.dropna().iloc[-1] / gdpn.dropna().iloc[-1])
    add = (12 / mat) * (y5 - eff) * share
    rows.append([
        "Fiscal", "Interest already locked in",
        f"{add:+.2f} pts of GDP a year",
        f"{100 / (mat / 12):.0f}% of the debt rolls each year at {mat / 12:.1f}-year average "
        f"maturity, replacing {eff:.1f}% coupons with {y5:.1f}% yields on a stock of "
        f"{100 * share:.0f}% of GDP. The interest bill is {burden:.1f}% of GDP now; on this "
        f"arithmetic alone it reaches {burden + 3 * add:.1f}% in three years with no further "
        f"move from the Fed."])

cx = latest("supply::energy_capex_revision")
if cx is not None and len(cx):
    rows.append([
        "Supply", "Energy investment now, supply later",
        f"score {cx.iloc[-1]:+.2f}",
        "What the majors commit to wells and LNG becomes barrels about five years out. "
        "Directionally consistent across four regions (+0.24 to +0.41 against world oil "
        "supply at five years) but no interval clears zero: Bloomberg capex history "
        "starts in 2007, leaving two or three independent windows. Mechanism, not proof."])

for ind_id, label, note in (
        ("supply::cbo_labour_force_revision", "How many people there will be to employ",
         "CBO's projection of the labour force five years out, revised. Demographics are "
         "the most forecastable thing in the set: the people who will be working in 2031 "
         "have already been born and mostly already hired."),
        ("supply::cbo_lfpr_revision", "How many of them will choose to work",
         "The participation half, isolated — where a swing in female participation shows "
         "and population growth cannot see it. 20 usable publications since 2019, so it "
         "is here on construct grounds and cannot be validated.")):
    s = latest(ind_id)
    if s is not None and len(s):
        rows.append(["Supply", label, f"score {s.iloc[-1]:+.2f}", note])

st.html(ui.table(["Driver", "What", "Now", "The arithmetic"], rows))


# ---------- 2. reversion ----------

st.html(ui.section_head(
    "2 · Stretched against its own history",
    caption="Measures far from their own norms, which have historically come back. This is "
            "a statistical regularity rather than a mechanism, so it is weaker than the "
            "section above and is kept separate for that reason."))

st.html(
    '<div class="m-body" style="max-width:46rem"><p><b>Business investment intensity.</b> '
    'Real business investment relative to real GDP, against its own ten-year norm — how '
    'much faster capex has been outgrowing the economy than it usually does. Against '
    'investment growth three years out it scores <b>−0.56</b>, and across five cuts of '
    'Oct 1976 – Apr 2023 quarterly data (187 quarters, roughly 15 independent three-year '
    'windows) it ranges only <b>−0.52 to −0.57</b>: removing the financial crisis gives '
    '−0.52, removing covid −0.57, removing both −0.53, and the pre-2013 period alone '
    '−0.57. At five years it is −0.65.</p>'
    '<p>It is <b>not</b> in the scored driver, deliberately. The investment driver reads '
    '"how strong is capex now", and a reversion measure would push it toward <i>Low '
    'investment</i> during a boom — right about 2029 and wrong about today. It belongs '
    'here, where it needs no sign convention to fight with.</p></div>')


# ---------- 3. what each driver can and cannot say ----------

st.html(ui.section_head(
    "3 · Where each driver can speak in years",
    caption="Share of each driver's weight that is long-horizon, and what carries it. The "
            "dashboard is deliberately weighted toward fast data; this is what is left "
            "when that is set aside."))

rows = []
for name, block in cfg["drivers"].items():
    inds = block["indicators"]
    long_w = sum(i["weight"] for i in inds if horizon.get(i["id"]) == "long")
    longs = [i for i in inds if horizon.get(i["id"]) == "long"]
    names = ui.join_words([i["label"].split(",")[0] for i in longs]) if longs else "nothing"
    rows.append([block.get("label", name), f"{long_w:.0%}", names])
st.html(ui.table(["Driver", "Long-horizon weight", "Carried by"], rows, numeric={1}))

st.html(
    '<div class="m-body" style="max-width:46rem"><p>The asymmetry is the finding. '
    '<b>Fiscal and supply</b> have committed quantities — debt schedules, capacity under '
    'construction, demographics — so they can say something about 2029. '
    '<b>Demand and investment</b> largely cannot: spending decisions are not made three '
    'years ahead in any observable way, and every long-horizon candidate tested for them '
    'failed. <b>Monetary</b> is worse still, because the policy rate is set meeting by '
    'meeting and the measures that appear to predict it are reading the Fed\'s reaction '
    'to inflation rather than its effect on it.</p>'
    '<p>That is a real limit, not a gap to be filled by trying harder. It is more useful '
    'to your reader stated than hidden.</p></div>')


# ---------- 4. levers ----------

st.html(ui.section_head(
    "4 · Levers",
    caption="What would have to change for each driver to move, and the indicator that "
            "would show it first. A scenario is built by moving these, not by forecasting "
            "them."))

LEVERS = [
    ["Fiscal", "Rates stay high / the Treasury shortens issuance",
     "Locked-in refinancing, monthly",
     "Each point of yield above the 3.2% being paid adds about 0.16 pts of GDP a year "
     "to the interest bill at current maturity."],
    ["Supply", "The majors cut capex",
     "Energy sector capex revision, monthly",
     "Roughly five years from spending to barrels, so a cut now shows in supply around "
     "2031. Four regions agree on direction; none of the intervals clears zero."],
    ["Supply", "Immigration or participation policy shifts",
     "CBO labour force and participation revisions, quarterly",
     "The clearest arithmetic in the set: the 2024 immigration upgrade moved the "
     "labour force projection directly."],
    ["Investment", "The capex boom matures",
     "Investment intensity vs its 10-year norm, quarterly",
     "Currently +26.9 against a 23.2 norm. Historically −0.56 with growth three years "
     "out, Oct 1976 – Apr 2023."],
    ["Inflation", "The anchor moves",
     "SPF ten-year CPI, level and revision, quarterly",
     "Neither predicts realised inflation; together they say whether expectations are "
     "where they should be and whether they are moving."],
    ["Monetary", "The Fed's reaction function changes",
     "Taylor gap, monthly",
     "The only stance measure that nets out the reaction function, which is why it is "
     "the one with the right sign."],
]
st.html(ui.table(["Driver", "What would change", "Watch", "What it would mean"], LEVERS))

st.caption("Prototype. The sections are ordered by how much they can be trusted, not by "
           "how interesting they are — arithmetic first, reversion second, and scenarios "
           "last because they carry no claim about what will happen.")
