"""Indicator bench.

A place to argue about the set with the evidence on screen. Every row says what
the indicator refers to, how it is collected, and how far ahead it looks;
ticking one changes nothing the model does.

That is deliberate. The dashboard is scored from config/indicators.yml and the
published weights; a bench that silently rescored would turn a discussion into a
series of accidental commitments. What the page does give back is the thing a
selection has to survive — three to six per driver, a horizon mix that is not
all news, and a count of how much of it needs a vendor we do not have.
"""

from __future__ import annotations

import streamlit as st

from src import bench as bn
from src import ui
from src.drivers import load_config
from views.common import INDICATORS, country_picker, settings_banner

KEY = "bench::"
SEEDED = "bench_seeded"

bench = bn.load()
cfg = load_config(INDICATORS())
all_rows = bn.rows(bench)
live_ids = set(bn.in_set_ids(bench))


def key_of(row_id: str) -> str:
    return f"{KEY}{row_id}"


def selected() -> set[str]:
    return {r["id"] for r in all_rows if st.session_state.get(key_of(r["id"]), False)}


def seed(ids: set[str]) -> None:
    for r in all_rows:
        st.session_state[key_of(r["id"])] = r["id"] in ids


# The set we actually score is the starting point, so the discussion begins from
# what is true rather than from an empty page.
if not st.session_state.get(SEEDED):
    seed(live_ids)
    st.session_state[SEEDED] = True


# ---------- header ----------

st.html('<h1 class="mr-title">Indicator bench</h1>'
        '<p class="mr-sub">Every indicator we score today, and every candidate from the '
        'September review, with what it measures and how far ahead it looks. '
        'Tick and untick to build a set out loud.</p>')

country_picker(st.sidebar)
settings_banner()

st.html('<div class="mr-custom"><b>Nothing here is scored.</b> Selections stay in this '
        'browser session and never reach the model, so the dashboard and the published '
        'defaults are untouched whatever you tick. The counts below are the only '
        'feedback: they say whether a selection could actually ship.</div>')

# Every row carries two tags because the two questions come apart, and the gap
# between them is the case for changing the set.
st.html(
    '<div style="display:grid;gap:0.6rem 1.5rem;grid-template-columns:repeat(auto-fit,'
    f'minmax(15rem,1fr));border-left:2px solid {ui.MUTED};padding:0.5rem 0 0.5rem 1rem;'
    'margin:0.4rem 0 0.2rem">'
    f'<div><b>Horizon</b> — how far ahead it looks. <span style="color:{ui.INK_2}">'
    'ST weeks to a quarter · MT one to three years · LT structural or a market\'s '
    'multi-year view.</span></div>'
    f'<div><b>Nature</b> — what kind of thing it measures. <span style="color:{ui.INK_2}">'
    'NEWS what just happened · CYC where output sits relative to capacity · '
    'STR what capacity is and where it is drifting.</span></div>'
    f'<div style="grid-column:1/-1;color:{ui.INK_2}">The test for the second: does it '
    'move the speed limit, or tell you where you are against it? Capacity utilisation '
    'tells you where you are. The growth of capacity moves the limit. Four live '
    'indicators are tagged LT but mean-revert inside a cycle, which is how the set came '
    'to read 27% long-horizon while carrying 18% structural.</div></div>')


# ---------- toolbar ----------

bar = st.columns([1.35, 1.35, 1.5, 1.5, 1.9, 1.0, 1.0])
show = bar[0].selectbox("Show", ["Everything", "In the set", "Candidates"],
                        help="Narrow the list without losing what is already ticked.")
natures = bar[1].multiselect("Nature", list(bn.NATURE_ORDER), default=[],
                             format_func=lambda n: bn.NATURE_LABEL[n],
                             placeholder="Any kind",
                             help="Pick Structural to see only what moves the speed limit.")
horizons = bar[2].multiselect("Horizon", list(bn.HORIZON_ORDER), default=[],
                              format_func=lambda h: bn.HORIZON_LABEL[h],
                              placeholder="Any horizon")
sources = bar[3].multiselect("Source", list(bn.SOURCE_LABEL), default=[],
                             format_func=lambda s: bn.SOURCE_LABEL[s],
                             placeholder="Any source")
query = bar[4].text_input("Search", "", placeholder="name or description")

# Buttons run before any checkbox is drawn: Streamlit will not let a widget's
# state be rewritten in the same run that renders it.
if bar[5].button("Reset", use_container_width=True,
                 help="Back to the set that is scored today."):
    seed(live_ids)
    st.rerun()
if bar[6].button("Clear", use_container_width=True,
                 help="Untick everything and build up from nothing."):
    seed(set())
    st.rerun()

detail = st.toggle("Show how each one is measured", value=True,
                   help="Turn off to fit more rows on screen once everyone knows the list.")


def visible(row: dict) -> bool:
    if show == "In the set" and row.get("status") != "in_set":
        return False
    if show == "Candidates" and row.get("status") == "in_set":
        return False
    if natures and row.get("nature", "cyclical") not in natures:
        return False
    if horizons and row.get("horizon", "medium") not in horizons:
        return False
    if sources and row.get("where", "macrobond_check") not in sources:
        return False
    if query:
        hay = " ".join(str(row.get(f, "")) for f in ("name", "refers", "measured", "code", "vendor"))
        if query.lower() not in hay.lower():
            return False
    return True


# ---------- tally ----------

def horizon_bar(mix: dict, width: int = 96) -> str:
    parts = "".join(
        f'<span title="{bn.HORIZON_LABEL[h]} {mix[h]:.0%}" style="flex:{mix[h]:.4f};'
        f'background:{ui.HORIZON_CHIP[h][0]};border-radius:1px"></span>'
        for h in bn.HORIZON_ORDER if mix[h] > 0.004)
    return (f'<span style="display:inline-flex;gap:2px;width:{width}px;height:8px;'
            f'vertical-align:middle">{parts}</span>')


picked = selected()
blocks = bn.tally(bench, picked)
d = bn.diff(bench, picked)
needs_source = sum(1 for r in all_rows if r["id"] in picked and r.get("where") == "external")

whole = [r for r in all_rows if r["id"] in picked]
nat = bn.nature_mix(whole)
flat = bn.flattered(whole)

st.html(ui.section_head(
    "Where the selection stands",
    meta=f"{len(picked)} selected · {len(d['added'])} added · {len(d['dropped'])} dropped",
    caption="A driver has to carry three to six indicators to stay readable. Red means "
            "the selection could not ship as it is; amber means the driver has nothing "
            "structural in it at all."))

AMBER = "#b7791f"
cards = []
for b in blocks:
    tone = ui.INK if b["ok"] else "#c0392b"
    delta = b["n"] - b["was"]
    move = (f'<span style="color:{ui.MUTED}">was {b["was"]}</span>' if delta == 0
            else f'<span style="color:{ui.MUTED}">was {b["was"]}, '
                 f'{"+" if delta > 0 else "−"}{abs(delta)}</span>')
    mix = " / ".join(f'{bn.HORIZON_SHORT[h]} {b["horizons"][h]:.0%}'.replace("%", "")
                     for h in bn.HORIZON_ORDER if b["horizons"][h] >= 0.005) or "—"
    # A driver with no structural content is the thing this page exists to show,
    # so it gets its own colour rather than being one number among several.
    s_tone = AMBER if not b["structural"] else ui.INK
    s_move = "" if b["structural"] == b["was_structural"] else f' (was {b["was_structural"]})'
    new = (f'<div style="font-size:0.72rem;color:{ui.MUTED}">{b["new_sources"]} need a new source</div>'
           if b["new_sources"] else "")
    cards.append(
        f'<div style="border-top:2px solid {tone};padding:0.5rem 0.2rem 0.2rem">'
        # Two lines of room whether the name needs them or not, so the numbers
        # sit on one baseline and the row can be read across.
        f'<div style="font-size:0.78rem;color:{ui.MUTED};letter-spacing:0.04em;'
        f'text-transform:uppercase;line-height:1.15;min-height:2.3em">{ui.esc(b["label"])}</div>'
        f'<div style="font-size:1.6rem;font-weight:600;line-height:1.1;color:{tone}">{b["n"]}</div>'
        f'<div style="font-size:0.74rem">{move}</div>'
        f'<div style="margin-top:0.35rem">{horizon_bar(b["horizons"])}</div>'
        f'<div style="font-size:0.72rem;color:{ui.MUTED}">{mix}</div>'
        f'<div style="font-size:0.74rem;margin-top:0.3rem;color:{s_tone};'
        f'font-weight:{600 if not b["structural"] else 400}">'
        f'{b["structural"]} structural{s_move}</div>{new}</div>')
# One row of six on a laptop, wrapping only when the window is genuinely narrow:
# the six cards are meant to be compared at a glance, and a driver that falls to
# a second row stops being part of the comparison.
st.html('<div style="display:grid;gap:0.9rem;'
        'grid-template-columns:repeat(auto-fit,minmax(7.5rem,1fr))">'
        f'{"".join(cards)}</div>')

notes = [f"Across the whole selection: **news {nat['news']:.0%} · cyclical "
         f"{nat['cyclical']:.0%} · structural {nat['structural']:.0%}**."]
blank = [b["label"] for b in blocks if not b["structural"]]
if blank:
    notes.append(f"No structural content at all in **{ui.join_words(blank)}**.")
if flat:
    # Separated by middots, not commas: half these names contain a comma of
    # their own and a comma-joined list reads as twice as many items.
    names = " · ".join(f"**{r['name']}**" for r in flat)
    notes.append(f"Tagged long-horizon but cyclical in nature: {names} — these are why "
                 f"the horizon mix reads longer-dated than the set really is.")
if needs_source:
    notes.append(f"{needs_source} of the {len(picked)} selected would need a source we do "
                 f"not have today.")
st.caption("  \n".join(notes))


# ---------- the bench ----------

st.html(ui.section_head("The bench", caption=
        "Ticked rows are what we score today. Everything else is a candidate from the "
        "review; nothing has been backtested, so inclusion here is a proposal, not a result."))

for key, block in bn.drivers(bench).items():
    items = [i for i in (block.get("items") or []) if visible(i)]
    if not items:
        continue
    b = next(x for x in blocks if x["driver"] == key)
    label = block.get("label", key)
    hidden = len(block.get("items") or []) - len(items)
    suffix = f" · {hidden} hidden by filters" if hidden else ""
    with st.expander(f"{label} — {b['n']} selected{suffix}", expanded=True):
        for item in items:
            live = item.get("status") == "in_set"
            tag = bn.HORIZON_SHORT.get(item.get("horizon", "medium"), "MT")
            kind = item.get("nature", "cyclical")
            nat_tag = bn.NATURE_SHORT.get(kind, "CYC")
            src = bn.SOURCE_SHORT.get(item.get("where", "macrobond_check"), "MB?")
            mark = "" if live else " · new"
            st.checkbox(f"**{item['name']}**  `{tag}`  `{nat_tag}`  `{src}`{mark}",
                        key=key_of(item["id"]),
                        help=bn.NATURE_HELP.get(kind, ""))
            note = f"{item.get('refers', '')}"
            if detail and item.get("measured"):
                note += f"  \n*How it is measured:* {item['measured'].strip()}"
            note += f"  \n<small>{ui.esc(bn.source_note(item))}</small>"
            st.caption(note, unsafe_allow_html=True)


# ---------- what was decided ----------

st.html(ui.section_head("What this changes", caption=
        "Against the set that is scored today. Copy this into the notes before the "
        "tab is closed — the selection lives in the browser session and nowhere else."))

if not d["added"] and not d["dropped"]:
    st.caption("No change yet. This is the set as it ships.")
else:
    left, right = st.columns(2)
    with left:
        st.html(f'<div class="mr-caption"><b>Added ({len(d["added"])})</b></div>')
        st.html(ui.table(["Indicator", "Driver", "Source"],
                         [[r["name"], r["driver_label"], bn.source_note(r)] for r in d["added"]])
                if d["added"] else '<p class="mr-caption">Nothing added.</p>')
    with right:
        st.html(f'<div class="mr-caption"><b>Dropped ({len(d["dropped"])})</b></div>')
        st.html(ui.table(["Indicator", "Driver", "Was"],
                         [[r["name"], r["driver_label"],
                           bn.HORIZON_LABEL.get(r.get("horizon", "medium"), "Medium")]
                          for r in d["dropped"]])
                if d["dropped"] else '<p class="mr-caption">Nothing dropped.</p>')

st.code(bn.summary(bench, picked), language="text")
st.caption("Everything on this page is read from config/bench.yml. Adding a candidate, or "
           "correcting how one is described, is an edit to that file and needs no code.")
