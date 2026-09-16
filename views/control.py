"""Control room: change the model's judgment calls and watch the call move.

Every number here is an opinion someone had to pick. The page shows the
default, the argument for it, and a control to overrule it. Changes live in
this browser session only, so the published dashboard and everyone else's view
stay on the defaults.

Layout follows the order a reader argues in: what the engine does with the
drivers, how much each driver counts, then the indicators inside them.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from src import overrides as ovr
from src import ui
from views.common import (INDICATORS, REGIMES, base_config, country, country_picker,
                          get_overrides, get_results, overrides_active, reset_overrides,
                          set_overrides)

st.session_state["_mr_page"] = "control"

cfg, reg = base_config()
# Widget keys carry the country: "demand" exists in every country's config, and
# a slider's value must not follow the reader from one to another.
wk = country()["code"]
inds = ovr.indicator_defaults(cfg)
rationale = reg.get("rationale") or {}
sal_why = rationale.get("driver_salience") or {}
set_why = rationale.get("settings") or {}
current = get_overrides()
vintage = dt.date.today()

st.html('<hr class="mr-mast-rule">')
head_left, head_country = st.columns([3, 1], vertical_alignment="bottom")
here = country_picker(head_country)
head_left.html('<div class="mr-mast bare"><h1 class="mr-title">Control room</h1>'
               f'<p class="mr-sub">Change the assumptions behind the '
               f'{ui.esc(here["label"])} call and see what moves</p></div>')
st.html('<hr class="mr-rule">')

st.html(
    '<div class="m-body"><p>Feel free to update any assumption below &mdash; the dashboard '
    'rescores accordingly. Note that your settings stay in this browser session and leave '
    'the published defaults untouched.</p></div>')


def value_row(default, why: str) -> None:
    """The published default and the argument for it, above its control."""
    shown = f"{default:g}" if isinstance(default, (int, float)) else str(default)
    st.html(f'<p class="mr-ctl-def">Default: <b>{ui.esc(shown)}</b></p>'
            + (f'<p class="mr-ctl-why">{ui.esc(why)}</p>' if why else ""))


def staged() -> dict:
    """Overrides being edited: start from what is already applied."""
    return {group: dict(current.get(group) or {}) for group in ovr.GROUPS}


# ---------- what the reader has changed ----------

rows = ovr.summary(current, cfg, reg)
if rows:
    def fmt(v):
        return f"{v:g}" if isinstance(v, (int, float)) else str(v)
    st.html(ui.section_head("Your changes", f"{len(rows)} from the defaults"))
    st.html(ui.table(["Setting", "Default", "Yours"],
                     [[r["label"], fmt(r["default"]), fmt(r["custom"])] for r in rows],
                     numeric={1, 2}))
    if st.button("Reset everything to defaults", icon=":material/restart_alt:"):
        reset_overrides()
        st.rerun()

# ---------- impact ----------

custom_res = get_results(vintage)
base_res = get_results(vintage, use_overrides=False) if overrides_active() else custom_res
if custom_res is not None and base_res is not None and not custom_res["probabilities"].empty:
    def read(res):
        probs = res["probabilities"].dropna(how="all")
        month = probs.index[-1]
        call = res["calls"].loc[month]
        grade = call.get("fit")
        return {"month": month, "called": ui.call_label(call["called"], res["regimes"]),
                "top": probs.loc[month].max(),
                "grade": ui.FIT_WORDS.get(grade, "—") if isinstance(grade, str) else "—",
                "calls": res["calls"]["called"]}
    a, b = read(base_res), read(custom_res)
    differ = (a["calls"].reindex(b["calls"].index) != b["calls"]).mean()
    st.html(ui.section_head(
        "Impact", f"Latest confirmed month · {b['month']:%B %Y}",
        "The same data, scored both ways." if overrides_active() else
        "Nothing changed yet, so both columns are the published defaults."))
    st.html(ui.table(
        ["", "Default", "Your settings"],
        [["Called regime", a["called"], b["called"]],
         ["Leading probability", f"{a['top']:.0%}", f"{b['top']:.0%}"],
         ["Fit to nearest regime", a["grade"], b["grade"]],
         ["History that differs", "—", f"{differ:.0%} of months"]],
        numeric={1, 2}))

# ---------- engine ----------

st.html(ui.section_head(
    "Regime engine", "Five settings",
    "How distances become probabilities, how fast the call is allowed to change, and "
    "whether a month has to fit anything at all."))

with st.form("engine"):
    stage = staged()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Temperature**")
        value_row(reg["settings"]["temperature"], set_why.get("temperature", ""))
        stage[ovr.SETTINGS]["temperature"] = st.slider(
            "Temperature", *ovr.SETTING_LIMITS["temperature"],
            value=float(current[ovr.SETTINGS].get("temperature", reg["settings"]["temperature"])),
            step=0.05, label_visibility="collapsed",
            help="Lower puts more probability on the nearest regime.")

        st.markdown("**Persistence, months**")
        value_row(reg["settings"]["persistence_months"],
                  set_why.get("persistence_months", ""))
        stage[ovr.SETTINGS]["persistence_months"] = st.slider(
            "Persistence", *ovr.SETTING_LIMITS["persistence_months"],
            value=int(current[ovr.SETTINGS].get("persistence_months",
                                                reg["settings"]["persistence_months"])),
            step=1, label_visibility="collapsed",
            help="Consecutive months a new leader needs before the call changes.")

        st.markdown("**Confidence floor**")
        value_row(reg["settings"]["min_confidence"],
                  set_why.get("min_confidence", ""))
        stage[ovr.SETTINGS]["min_confidence"] = st.slider(
            "Confidence floor", *ovr.SETTING_LIMITS["min_confidence"],
            value=float(current[ovr.SETTINGS].get("min_confidence",
                                                  reg["settings"]["min_confidence"])),
            step=0.05, label_visibility="collapsed",
            help="Below this the call is reported as transitional.")
    with c2:
        st.markdown("**Fit gate**")
        value_row("on" if str(reg["settings"].get(ovr.GATE_KEY)) ==
                  "closer_than_neutral" else "off", set_why.get("fit_gate", ""))
        gate_now = str(current[ovr.SETTINGS].get(ovr.GATE_KEY,
                                                 reg["settings"].get(ovr.GATE_KEY, "none")))
        gate_on = st.toggle("Require a month to fit a regime",
                            value=gate_now == "closer_than_neutral",
                            help="Off: every month is called whichever regime is nearest, "
                                 "however far away that is.")
        stage[ovr.SETTINGS][ovr.GATE_KEY] = "closer_than_neutral" if gate_on else "none"

        st.markdown("**Fit tolerance**")
        value_row(reg["settings"].get("fit_tolerance", 1.0),
                  set_why.get("fit_tolerance", ""))
        stage[ovr.SETTINGS]["fit_tolerance"] = st.slider(
            "Fit tolerance", *ovr.SETTING_LIMITS["fit_tolerance"],
            value=float(current[ovr.SETTINGS].get("fit_tolerance",
                                                  reg["settings"].get("fit_tolerance", 1.0))),
            step=0.05, label_visibility="collapsed", disabled=not gate_on,
            help="1.00 calls no clear regime for anything further from a regime than a "
                 "neutral economy. Higher tolerates a weak fit before giving up.")
    if st.form_submit_button("Apply engine settings", type="primary"):
        set_overrides(stage)
        st.rerun()

# ---------- salience ----------

st.html(ui.section_head(
    "Driver salience", f"{ui.count_word(len(reg['driver_salience']))} drivers",
    "How much each driver counts when measuring the distance from this month to a regime. "
    "Set one to zero and the model stops caring about that driver entirely."))

with st.form("salience"):
    stage = staged()
    for driver, default in reg["driver_salience"].items():
        label = cfg["drivers"].get(driver, {}).get("label", driver)
        left, right = st.columns([1.55, 1], gap="large", vertical_alignment="center")
        left.markdown(f"**{label}**")
        left.html(f'<p class="mr-ctl-def">Default: <b>{float(default):g}</b></p>'
                  f'<p class="mr-ctl-why">{ui.esc(sal_why.get(driver, ""))}</p>')
        stage[ovr.SALIENCE][driver] = right.slider(
            label, *ovr.SALIENCE_LIMITS,
            value=float(current[ovr.SALIENCE].get(driver, default)), step=0.05,
            label_visibility="collapsed", key=f"sal::{wk}::{driver}")
    if st.form_submit_button("Apply salience", type="primary"):
        set_overrides(stage)
        st.rerun()

# ---------- indicators ----------

st.html(ui.section_head(
    "Indicator weights and centers", f"{len(inds)} indicators",
    "Weight is how much an indicator counts inside its driver; only the ratios matter, "
    "because weights are renormalized over whatever has reported. Center is the reading "
    "that scores zero, and scale is how many units of the series make one point of score."))
st.caption("One driver at a time. Apply each panel separately.")

for driver, spec in cfg["drivers"].items():
    keys = [k for k, v in inds.items() if v["driver"] == driver]
    changed = sum(1 for k in keys for g in (ovr.WEIGHTS, ovr.CENTERS, ovr.SCALES)
                  if k in current[g])
    title = spec.get("label", driver) + (f" · {changed} changed" if changed else "")
    # Weights are relative: the driver divides by the sum of whatever is
    # present, so raising one indicator lowers everything else's share without
    # the other numbers moving. Show the share each weight actually buys.
    live = {k: float(current[ovr.WEIGHTS].get(k, inds[k]["weight"])) for k in keys}
    live_total = sum(live.values()) or 1.0
    with st.expander(title):
        st.caption(spec.get("description", ""))
        if changed:
            st.caption(f"Weights here sum to {live_total:g}, against "
                       f"{sum(inds[k]['weight'] for k in keys):g} by default. Only the ratios "
                       f"matter: each indicator counts for its share of that sum.")
        with st.form(f"ind::{driver}"):
            stage = staged()
            for k in keys:
                spec_i = inds[k]
                share_now = live[k] / live_total
                st.markdown(f"**{spec_i['label']}**"
                            + (" · anchor" if spec_i["anchor"] else ""))
                if spec_i["why"]:
                    st.html(f'<p class="mr-ctl-why">{ui.esc(spec_i["why"])}</p>')
                cols = st.columns(3, gap="medium")
                stage[ovr.WEIGHTS][k] = cols[0].number_input(
                    f"Weight — {share_now:.0%} of the driver"
                    + (f" (default {spec_i['weight']:g}, {spec_i['share']:.0%})"
                       if abs(share_now - spec_i["share"]) > 5e-3
                       else f" (default {spec_i['weight']:g})"),
                    *ovr.WEIGHT_LIMITS,
                    value=float(current[ovr.WEIGHTS].get(k, spec_i["weight"])),
                    step=0.05, format="%.2f", key=f"w::{wk}::{k}")
                if spec_i["method"] == "gap":
                    lo, hi = ovr.center_limits(spec_i["center"], spec_i["scale"])
                    step = max(round(abs(spec_i["scale"]) / 10, 4), 0.01)
                    fmt = ("%.1f" if abs(spec_i["center"]) >= 10 else
                           "%.2f" if abs(spec_i["scale"]) >= 0.1 else "%.3f")
                    stage[ovr.CENTERS][k] = cols[1].number_input(
                        f"Center (default {spec_i['center']:g})", lo, hi,
                        value=float(current[ovr.CENTERS].get(k, spec_i["center"])),
                        step=step, format=fmt, key=f"c::{wk}::{k}",
                        help="The reading that scores zero.")
                    slo, shi = ovr.scale_limits(spec_i["scale"])
                    stage[ovr.SCALES][k] = cols[2].number_input(
                        f"Scale (default {spec_i['scale']:g})", slo, shi,
                        value=float(current[ovr.SCALES].get(k, spec_i["scale"])),
                        step=step, format=fmt, key=f"s::{wk}::{k}",
                        help="Units of the series per point of score. Smaller is more "
                             "sensitive.")
                else:
                    cols[1].caption(f"Z-score over {spec_i['window']} months. No center "
                                    f"or scale to argue about: the window defines both.")
                st.html('<hr class="mr-rule" style="margin:0.5rem 0 0.9rem">')
            if st.form_submit_button(f"Apply {spec.get('label', driver).lower()}",
                                     type="primary"):
                set_overrides(stage)
                st.rerun()

# ---------- what is fixed ----------
# The full indicator set with its weights lives on the methodology page; no
# need to print it twice.

st.html(ui.section_head("What cannot be changed here"))
st.html('<div class="m-body"><ul>'
        '<li><b>Series, transforms, direction, anchors and carry-forward rules.</b> These '
        'define what a number is, not how much it counts. Change them and the labels on '
        'the dashboard stop describing the data underneath, so they stay in '
        f'<code>{ui.esc(INDICATORS())}</code> where a change is reviewable.</li>'
        '<li><b>Regime archetypes.</b> The coordinates of each regime are what the words '
        '"goldilocks" and "stagflation" mean here. Editing them privately would leave two '
        'readers using the same name for different economies. Argue them in '
        f'<code>{ui.esc(REGIMES())}</code>, in public.</li>'
        '<li><b>The regime set itself.</b> Adding or removing a regime changes every '
        'probability on the page and needs the methodology rewritten with it.</li>'
        '</ul><p>Values you do set are clamped to ranges that keep the model meaningful: '
        'weights cannot go negative, a scale cannot reach zero because it divides, and a '
        'center cannot move so far that every month clips to the same score.</p></div>')
st.page_link("views/methodology.py", label="How these numbers are used",
             icon=":material/menu_book:")

