"""Presentation: HTML fragments and chart styling for app.py.

Pure functions of config and frames with no Streamlit calls, so the layout
logic can be tested. The one that matters is where a regime or a reading lands
on a signpost track: a flipped sign there draws a confident, wrong picture.
"""

from __future__ import annotations

import html

import altair as alt
import pandas as pd

HEADING_FONT = ("'Sabon Next LT', 'Sabon LT Pro', 'Sabon LT Std', Sabon, "
                "Georgia, 'Times New Roman', serif")
BODY_FONT = "Arial, Helvetica, sans-serif"

INK = "#111111"
INK_2 = "#4d4d4d"
MUTED = "#8c8c8c"
GRID = "#e3e3e3"
AXIS = "#bdbdbd"
NAVY = "#12203f"      # driver labels on the signpost chart only, as in the deck
TRACK = "#eeeeee"
TRANSITIONAL = "#b3b3b3"
UNCLASSIFIED_COLOR = "#d9d9d9"
CALL_STATES = {"transitional": ("Transitional", TRANSITIONAL),
               "unclassified": ("No clear regime", UNCLASSIFIED_COLOR)}


def call_label(called, reg_cfg: dict) -> str:
    """Display name for a call: a regime label, 'Transitional' or 'No clear regime'."""
    if called in reg_cfg["regimes"]:
        return reg_cfg["regimes"][called]["label"]
    return CALL_STATES.get(called, ("No call", TRANSITIONAL))[0]


def call_color(called, reg_cfg: dict) -> str:
    if called in reg_cfg["regimes"]:
        return reg_cfg["regimes"][called]["color"]
    return CALL_STATES.get(called, ("", TRANSITIONAL))[1]
TONES = {  # end-of-scale boxes: (fill, text)
    "risk": ("#b83232", "#ffffff"),
    "good": ("#1f8a1f", "#ffffff"),
    "neutral": ("#52514e", "#ffffff"),
}

TRACK_PAD = 7.0     # % of track kept clear at each end so ±1 labels don't touch the end boxes
CLUSTER_GAP = 11.0  # % of track; regimes closer than this share one label slot


# ---------- geometry ----------

def track_position(value: float, reverse: bool = False) -> float:
    """Map a driver score in [-1, 1] to a left offset in percent of the track."""
    v = max(-1.0, min(1.0, float(value)))
    if reverse:
        v = -v
    return TRACK_PAD + (v + 1.0) / 2.0 * (100.0 - 2.0 * TRACK_PAD)


def cluster(points: list[tuple[float, str]], gap: float = CLUSTER_GAP):
    """Group (position, key) pairs whose neighbours sit closer than `gap`.

    Returns [(mean_position, [keys in left-to-right order])]. Chained, so
    three regimes each a little apart share one slot rather than overlapping.
    """
    groups: list[list[tuple[float, str]]] = []
    for pos, key in sorted(points):
        if groups and pos - groups[-1][-1][0] < gap:
            groups[-1].append((pos, key))
        else:
            groups.append([(pos, key)])
    return [(sum(p for p, _ in g) / len(g), [k for _, k in g]) for g in groups]


def text_on(fill: str) -> str:
    """White or ink, whichever reads on the fill."""
    r, g, b = (int(fill.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return INK if (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) else "#ffffff"


def signed(x: float) -> str:
    # Round first: a score of -0.001 printed as "−0.00", which reads as a
    # negative reading that is somehow zero.
    value = 0.0 if abs(x) < 0.005 else x
    return f"{value:+.2f}".replace("-", "−")


# ---------- page chrome ----------

CSS = f"""
<style>
/* Editorial, data-first: black type on white, hairline rules instead of boxes,
   one strong rule above each section, small consistent radii. */
[data-testid="stMainBlockContainer"] {{ max-width: 1200px; padding-top: 4.6rem; }}
.mr-mast {{ border-top: 3px solid {INK}; padding-top: 0.7rem; }}
.mr-mast-rule {{ border: 0; border-top: 3px solid {INK}; margin: 0 0 0.2rem; }}
.mr-mast.bare {{ border-top: 0; padding-top: 0; }}
.mr-eyebrow {{ font: 0.9rem {BODY_FONT}; color: {INK_2}; margin: 0 0 0.2rem; }}
.mr-title {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 2.35rem;
  color: {INK}; line-height: 1.08; margin: 0; letter-spacing: -0.005em; }}
.mr-sub {{ font: 0.92rem/1.45 {BODY_FONT}; color: {INK_2}; margin: 0.3rem 0 0; }}
.mr-rule {{ border: 0; border-top: 1px solid {GRID}; margin: 0.9rem 0 1.1rem; }}
.mr-h2 {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 1.45rem; color: {INK};
  margin: 2.2rem 0 0.15rem; padding-top: 0.55rem; border-top: 2px solid {INK};
  display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; flex-wrap: wrap; }}
.mr-h2-meta {{ font: 0.8rem {BODY_FONT}; color: {INK_2}; font-variant-numeric: tabular-nums; }}
.mr-links {{ font: 0.8rem/1.5 {BODY_FONT}; color: {MUTED}; margin: 0.35rem 0 0; }}
.mr-links a, .mr-source a {{ color: {INK_2}; text-decoration: underline; text-decoration-color: {AXIS};
  text-underline-offset: 2px; }}
.mr-caption {{ font: 0.88rem/1.45 {BODY_FONT}; color: {INK_2}; margin: 0 0 0.9rem; max-width: 60rem; }}
.mr-source {{ font: 0.76rem/1.4 {BODY_FONT}; color: {MUTED}; margin: 0.35rem 0 0; }}

.mr-call {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 2.7rem;
  line-height: 1.05; color: {INK}; margin: 0.05rem 0 0.55rem; }}
.mr-call-swatch {{ display: inline-block; width: 0.5em; height: 0.5em;
  border-radius: 50%; margin-right: 0.3em; vertical-align: 0.1em; }}
.mr-lede {{ font: 1rem/1.55 {BODY_FONT}; color: {INK}; margin: 0 0 0.45rem; max-width: 38rem; }}
.mr-lede-muted {{ font: 0.86rem/1.5 {BODY_FONT}; color: {INK_2}; margin: 0 0 0.35rem;
  max-width: 38rem; }}
.mr-lede-muted:last-child {{ margin-bottom: 0; }}
.mr-demo {{ font: 0.86rem/1.5 {BODY_FONT}; color: {INK}; margin: 0.7rem 0 0; }}
.mr-demo b {{ color: #b83232; }}

/* Custom settings: a reader has moved the model off its published defaults.
   Marked on every page, because nothing else on the page would show it. */
.mr-custom {{ font: 0.86rem/1.55 {BODY_FONT}; color: {INK}; background: #fbf6e8;
  border-left: 3px solid #c99a1e; padding: 0.55rem 0.8rem; margin: 0 0 0.3rem; }}
.mr-ctl-why {{ font: 0.82rem/1.5 {BODY_FONT}; color: {INK_2}; margin: 0.1rem 0 0.9rem;
  max-width: 56rem; }}
.mr-ctl-def {{ font: 0.78rem {BODY_FONT}; color: {MUTED}; margin: 0.1rem 0 0; }}
.mr-ctl-def b {{ color: {INK}; font-weight: 700; }}

/* The month being examined: the call for that month is the context a reader
   needs before reading any breakdown, so it is set like a call, not a caption. */
.mr-focus-call {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 1.5rem;
  color: {INK}; line-height: 1.15; margin: 0.1rem 0 0.25rem; }}
.mr-focus-note {{ font: 0.86rem/1.5 {BODY_FONT}; color: {INK_2}; margin: 0; }}

.mr-probs {{ font-family: {BODY_FONT}; }}
.mr-probs-head {{ font: 700 0.9rem {BODY_FONT}; color: {INK}; margin: 0.35rem 0 0.5rem; }}
.mr-prob {{ display: grid; grid-template-columns: minmax(9rem, 13rem) 1fr 3rem;
  align-items: center; gap: 0.75rem; padding: 0.4rem 0; border-bottom: 1px solid {GRID}; }}
.mr-prob-label {{ font-size: 0.92rem; color: {INK}; }}
.mr-prob-label b {{ font-weight: 700; }}
.mr-prob-code {{ color: {MUTED}; font-size: 0.76rem; margin-left: 0.3rem; }}
.mr-prob-track {{ position: relative; height: 9px; background: {TRACK}; }}
.mr-prob-bar {{ height: 9px; }}
.mr-prob-val {{ font-size: 0.95rem; color: {INK}; text-align: right; font-variant-numeric: tabular-nums; }}
.mr-prob.two {{ grid-template-columns: minmax(9rem, 13rem) 1fr 2.8rem 2.8rem; }}
.mr-prob-cols {{ border-bottom: 1px solid {INK}; align-items: end; padding: 0 0 0.35rem; }}
.mr-prob-colhead {{ font-size: 0.8rem; font-weight: 700; color: {INK}; text-align: right; }}
.mr-prob-tick {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: {INK};
  box-shadow: 0 0 0 1px #fff; }}
.mr-prob-prov {{ color: {INK_2}; }}
.mr-key-tick {{ display: inline-block; width: 2px; height: 11px; background: {INK};
  margin: 0 0.2rem 0 0.1rem; vertical-align: -1px; }}
.mr-prob-note {{ font-size: 0.76rem; line-height: 1.5; color: {MUTED}; margin: 0.45rem 0 0; }}

.mr-prov {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 2.5rem;
  border-top: 1px solid {INK}; border-bottom: 1px solid {GRID}; padding: 0.9rem 0 1rem;
  margin: 1.4rem 0 0; font-family: {BODY_FONT}; }}
.mr-prov p {{ margin: 0; }}
.mr-prov-title {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 1.2rem; color: {INK};
  margin: 0 0 0.35rem; }}
.mr-prov-title span {{ font-family: {BODY_FONT}; font-weight: 400; font-size: 0.86rem; color: {INK_2};
  margin-left: 0.4rem; }}
.mr-prov-main {{ font-size: 1.05rem; font-weight: 700; color: {INK}; margin: 0 0 0.4rem !important; }}
.mr-prov-body {{ font-size: 0.9rem; line-height: 1.55; color: {INK}; max-width: 38rem; }}
/* Mechanics of the reading: how much data, when it confirms. Secondary to
   what the reading says, so it is set back and given air above it. */
.mr-prov-note {{ font: 0.84rem/1.5 {BODY_FONT}; color: {INK_2}; max-width: 38rem;
  margin: 0.85rem 0 0 !important; }}
.mr-cal-title {{ font-weight: 700; font-size: 0.9rem; color: {INK}; margin: 0 0 0.35rem !important; }}
.mr-cal {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; color: {INK}; }}
.mr-cal th {{ text-align: left; font-weight: 400; color: {MUTED}; font-size: 0.76rem;
  border-bottom: 1px solid {AXIS}; padding: 0 0.6rem 0.25rem 0; }}
.mr-cal td {{ border-bottom: 1px solid {GRID}; padding: 0.32rem 0.6rem 0.32rem 0; vertical-align: top; }}
.mr-cal td.date {{ white-space: nowrap; font-variant-numeric: tabular-nums; width: 4.2rem; }}
.mr-cal td.drv {{ color: {INK_2}; white-space: nowrap; }}
.mr-cal .conf {{ font-size: 0.76rem; color: {INK_2}; }}

.sp {{ font-family: {BODY_FONT}; }}
.sp-legend {{ display: flex; flex-wrap: wrap; gap: 0.45rem 1.1rem; align-items: center;
  font-size: 0.82rem; color: {INK_2}; margin: 0 0 0.8rem; }}
.sp-legend span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
.sp-legend:has(+ .sp-legend-regimes) {{ margin-bottom: 0.4rem; }}
.sp-row {{ display: grid; grid-template-columns: 13.5rem 1fr; gap: 1rem;
  align-items: center; padding: 0.4rem 0; }}
.sp-pill {{ background: {NAVY}; color: #fff; border-radius: 3px; padding: 0.62rem 0.7rem;
  text-align: center; font-weight: 700; font-size: 1rem; line-height: 1.2; }}
.sp-read {{ font-size: 0.76rem; color: {INK_2}; text-align: center; margin-top: 0.28rem;
  font-variant-numeric: tabular-nums; }}
.sp-scale {{ display: grid; grid-template-columns: 7rem 1fr 7rem; min-height: 4.4rem; }}
.sp-end {{ display: flex; align-items: center; justify-content: center; text-align: center;
  font-weight: 700; font-size: 0.84rem; line-height: 1.2; padding: 0.3rem 0.45rem; }}
.sp-end.left {{ border-radius: 3px 0 0 3px; }}
.sp-end.right {{ border-radius: 0 3px 3px 0; }}
.sp-track {{ position: relative; background: {TRACK}; margin: 0 2px; }}
.sp-zero {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: {AXIS}; }}
.sp-star {{ position: absolute; top: 0.2rem; transform: translateX(-50%);
  font-size: 1.4rem; line-height: 1; color: {INK}; z-index: 3; }}
.sp-star-prov {{ color: {INK}; z-index: 4; }}
.sp-prov-key {{ color: {INK}; font-size: 1.15rem; line-height: 1; }}
.sp-then {{ position: absolute; top: 0.45rem; width: 11px; height: 11px; border-radius: 50%;
  transform: translateX(-50%); border: 2px solid {MUTED}; background: {TRACK}; z-index: 2; }}
.sp-group {{ position: absolute; bottom: 0.45rem; transform: translateX(-50%);
  display: flex; gap: 2px; white-space: nowrap; z-index: 1; }}
.sp-chip {{ font-size: 0.74rem; font-weight: 700; padding: 0.14rem 0.34rem; border-radius: 2px; }}
.sp-note {{ font-size: 0.78rem; color: {INK_2}; margin-top: 0.5rem; }}

.mr-foot {{ font: 0.78rem/1.6 {BODY_FONT}; color: {INK_2}; border-top: 1px solid {INK};
  margin-top: 2.6rem; padding-top: 0.7rem; }}
.mr-foot b {{ color: {INK}; }}

/* Prose measure. Wide enough to use the page next to the tables it sits among,
   short enough that the eye still finds the next line. */
.m-body p, .m-body li {{ font: 0.98rem/1.62 {BODY_FONT}; color: {INK}; max-width: 62rem; }}
.m-body p {{ margin: 0 0 0.75rem; }}
.m-body ul, .m-body ol {{ margin: 0 0 0.9rem; padding-left: 1.3rem; }}
.m-body li {{ margin: 0 0 0.35rem; }}
.m-body code {{ font-size: 0.86em; background: {TRACK}; padding: 0.05rem 0.25rem; border-radius: 2px; }}
.m-body a, .mr-caption a, .m-toc a, .m-table a, .mr-foot a {{ color: {INK}; text-decoration: underline;
  text-decoration-color: {AXIS}; text-underline-offset: 2px; }}
.m-section {{ scroll-margin-top: 5rem; }}
.m-h3 {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 1.15rem; color: {INK};
  margin: 1.3rem 0 0.3rem; }}
.m-toc {{ list-style: none; padding: 0; margin: 0.4rem 0 0; columns: 2; column-gap: 2.5rem;
  font: 0.95rem/1.5 {BODY_FONT}; max-width: 58rem; }}
.m-toc li {{ margin: 0 0 0.3rem; break-inside: avoid; }}
.m-toc span {{ color: {MUTED}; display: inline-block; width: 1.6rem; font-variant-numeric: tabular-nums; }}
.m-callout {{ border-top: 1px solid {INK}; border-bottom: 1px solid {GRID}; padding: 0.7rem 0;
  margin: 0.8rem 0 1rem; max-width: 62rem; }}
.m-callout p:last-child {{ margin-bottom: 0; }}
.m-wrap {{ overflow-x: auto; margin: 0.3rem 0 1rem; }}
.m-table {{ width: 100%; border-collapse: collapse; font: 0.87rem/1.45 {BODY_FONT}; color: {INK}; }}
.m-table th {{ text-align: left; font-weight: 700; color: {INK}; border-bottom: 1px solid {INK};
  padding: 0.4rem 0.9rem 0.35rem 0; white-space: nowrap; vertical-align: bottom; }}
.m-table td {{ border-bottom: 1px solid {GRID}; padding: 0.45rem 0.9rem 0.45rem 0; vertical-align: top; }}
.m-table td.num, .m-table th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.m-table td.muted {{ color: {INK_2}; }}
.m-table tr.group td {{ font-weight: 700; color: {INK}; border-bottom: 1px solid {AXIS}; padding-top: 1rem; }}
.m-dl dt {{ font: 700 0.95rem {BODY_FONT}; color: {INK}; margin-top: 0.7rem; }}
.m-dl dd {{ font: 0.95rem/1.55 {BODY_FONT}; color: {INK_2}; margin: 0.15rem 0 0; max-width: 62rem; }}

@media (max-width: 760px) {{
  .m-toc {{ columns: 1; }}
  .mr-prov {{ grid-template-columns: 1fr; gap: 1rem; }}
  .sp-row {{ grid-template-columns: 1fr; gap: 0.4rem; }}
  .sp-end {{ font-size: 0.66rem; padding: 0.15rem; overflow-wrap: anywhere; }}
  .sp-scale {{ grid-template-columns: 3.9rem 1fr 3.9rem; }}
  .sp-chip {{ font-size: 0.6rem; padding: 0.1rem 0.2rem; }}
  .sp-group {{ gap: 1px; }}
  .mr-prob {{ grid-template-columns: 8rem 1fr 2.6rem; }}
  .mr-prob.two {{ grid-template-columns: 7rem 1fr 2.4rem 2.4rem; gap: 0.5rem; }}
  .mr-call {{ font-size: 2.1rem; }}
}}
</style>
"""


def esc(s) -> str:
    return html.escape(str(s))


def section_head(title: str, meta: str = "", caption: str = "") -> str:
    """Section title on a strong rule, with frequency and dates right-aligned on the same line."""
    meta_html = f'<span class="mr-h2-meta">{esc(meta)}</span>' if meta else ""
    cap = f'<p class="mr-caption">{caption}</p>' if caption else ""
    return f'<h2 class="mr-h2"><span>{esc(title)}</span>{meta_html}</h2>{cap}'


def table(header: list[str], rows: list[list], numeric: set[int] = frozenset()) -> str:
    """A plain HTML table that wraps long text, unlike st.dataframe.

    Cells are escaped unless passed as ui.Raw. A row given as a single string
    renders as a full-width group heading.
    """
    def cell(v):
        return v.html if isinstance(v, Raw) else esc(v)

    head = "".join(f'<th class="num">{esc(h)}</th>' if i in numeric else f"<th>{esc(h)}</th>"
                   for i, h in enumerate(header))
    body = []
    for r in rows:
        if isinstance(r, str):
            body.append(f'<tr class="group"><td colspan="{len(header)}">{esc(r)}</td></tr>')
            continue
        body.append("<tr>" + "".join(
            f'<td class="num">{cell(v)}</td>' if i in numeric else f"<td>{cell(v)}</td>"
            for i, v in enumerate(r)) + "</tr>")
    return (f'<div class="m-wrap"><table class="m-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


class Raw:
    """Marks trusted markup for ui.table."""
    def __init__(self, html_: str):
        self.html = html_


# ---------- headline ----------

def probability_panel(probs_row: pd.Series, reg_cfg: dict, called: str,
                      confirmed_label: str = "", provisional: pd.Series | None = None,
                      provisional_label: str = "") -> str:
    """One thin bar per regime in config order, so a regime never changes place.

    Bars are the confirmed month. A provisional month, if any, shows as a thin
    outlined tick on the same track and its own column of numbers, so the two
    can be compared without the provisional one looking settled.
    """
    has_prov = provisional is not None
    rows = []
    for name, spec in reg_cfg["regimes"].items():
        p = float(probs_row.get(name, float("nan")))
        width = 0 if pd.isna(p) else max(0.0, min(1.0, p)) * 100
        label = f"<b>{esc(spec['label'])}</b>" if name == called else esc(spec["label"])
        tick = prov_val = ""
        title = f"{esc(spec['label'])}: {p:.0%} in {esc(confirmed_label)}"
        if has_prov:
            q = float(provisional.get(name, float("nan")))
            tick = (f'<div class="mr-prob-tick" style="left:calc({max(0.0, min(1.0, q)) * 100:.1f}% - 1px)"></div>'
                    if not pd.isna(q) else "")
            prov_val = f'<div class="mr-prob-val mr-prob-prov">{q:.0%}</div>'
            title += f", {q:.0%} provisional in {esc(provisional_label)}"
        rows.append(
            f'<div class="mr-prob{" two" if has_prov else ""}" title="{title}">'
            f'<div class="mr-prob-label">{label}'
            f'<span class="mr-prob-code">{esc(spec.get("short", ""))}</span></div>'
            f'<div class="mr-prob-track"><div class="mr-prob-bar" '
            f'style="width:{width:.1f}%;background:{spec["color"]}"></div>{tick}</div>'
            f'<div class="mr-prob-val">{p:.0%}</div>{prov_val}</div>'
        )
    head = '<div class="mr-probs-head">Probability</div>'
    note = ""
    if has_prov:
        head = (f'<div class="mr-prob two mr-prob-cols"><div class="mr-probs-head" style="margin:0">'
                f'Probability</div><div></div>'
                f'<div class="mr-prob-colhead">{esc(confirmed_label)}</div>'
                f'<div class="mr-prob-colhead">{esc(provisional_label)}*</div></div>')
        note = (f'<p class="mr-prob-note">Bars are {esc(confirmed_label)}, confirmed. '
                f'<span class="mr-key-tick"></span> Ticks are {esc(provisional_label)}, '
                f'provisional.<br>* {esc(provisional_label)} can still change as data '
                f'arrives.</p>')
    return '<div class="mr-probs">' + head + "".join(rows) + note + "</div>"


def provisional_box(reading: dict, reg_cfg: dict, called: str, confirm_by, confirm_with: list[str],
                    upcoming: list[dict], moved: str = "", since=None) -> str:
    """The provisional reading as a full-width band under the confirmed call.

    Left: what the early data says and when the month should be confirmed.
    Right: the releases still to come, as a small calendar table.
    upcoming: dicts with release, date, driver, confirms (bool).
    """
    month = reading["month"]
    lead = reading["leading"]
    spec = reg_cfg["regimes"][lead]
    p = float(reading["probabilities"][lead])
    fits = bool(reading.get("fits", True))
    weak = ", a weak fit" if reading.get("fit") == "weak" else ""
    called_name = call_label(called, reg_cfg)
    if not fits:
        verdict = (f"No clear regime. Nearest is {esc(spec['label'])} ({p:.0%}), but not close "
                   f"enough to call it.")
    elif lead == called:
        verdict = f"Leaning {esc(spec['label'])}, {p:.0%}{weak}, in line with the call."
    elif called == "unclassified":
        verdict = (f"Leaning {esc(spec['label'])}, {p:.0%}{weak}, while the confirmed call is "
                   f"no clear regime.")
    else:
        verdict = f"Leaning {esc(spec['label'])}, {p:.0%}{weak}, away from the {esc(called_name)} call."
    confirm = ""
    if confirm_by is not None and not pd.isna(confirm_by):
        confirm = f" {month:%B} should be confirmed around {day_month(confirm_by)}"
        if confirm_with:
            confirm += f", once the {esc(join_words(confirm_with))} {'is' if len(confirm_with) == 1 else 'are'} out"
        confirm += "."
    cal = ""
    if upcoming:
        rows = "".join(
            f'<tr><td class="date">{day_month(u["date"], short=True)}</td>'
            f'<td>{esc(u["release"])}'
            + (f' <span class="conf">· confirms {month:%B}</span>' if u.get("confirms") else "")
            + f'</td><td class="drv">{esc(u["driver"])}</td></tr>'
            for u in upcoming)
        cal = (f'<div><p class="mr-cal-title">Still to come for {month:%B}</p>'
               f'<table class="mr-cal"><thead><tr><th>Expected</th><th>Release</th><th>Driver</th></tr>'
               f'</thead><tbody>{rows}</tbody></table>'
               f'<p class="mr-source">Dates estimated from each series\' recent release timing.</p></div>')
    change = ""
    if moved:
        opener = f"Since {since:%B}, " if since is not None else ""
        text = esc(moved[0].lower() + moved[1:]) if opener else esc(moved[0].upper() + moved[1:])
        change = f'<p class="mr-prov-body">{opener}{text}.</p>'
    return (
        f'<div class="mr-prov"><div>'
        f'<h3 class="mr-prov-title">Provisional reading, {month:%B %Y}</h3>'
        f'<p class="mr-prov-main"><span class="mr-call-swatch" style="background:{spec["color"]}"></span>'
        f'{verdict}</p>'
        f'{change}'
        f'<p class="mr-prov-note">Based on {reading["share"]:.0%} of {month:%B} data released so far. '
        f'Inputs not yet released carry their latest value, so this can change as releases '
        f'arrive.{confirm}</p></div>{cal}</div>'
    )


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def count_word(n: int) -> str:
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def ordinal_size(rank: int, total: int) -> str:
    """'largest', 'second largest', ... 'smallest' for a 0-based rank among total."""
    if rank >= total - 1:
        return "smallest"
    words = ["largest", "second largest", "third largest", "fourth largest", "fifth largest",
             "sixth largest", "seventh largest", "eighth largest", "ninth largest"]
    return words[rank] if rank < len(words) else f"number {rank + 1}"


def day_month(when, short: bool = False) -> str:
    """'September 16' or 'Sep 16', without platform-specific strftime flags."""
    return f"{when:%b} {when.day}" if short else f"{when:%B} {when.day}"


def long_date(when) -> str:
    """'September 15, 2026'."""
    return f"{when:%B} {when.day}, {when.year}"


def join_words(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


STATUS_TEXT = {"reported": "Released", "carried": "Carried forward", "pending": "Not yet released",
               "not_started": "Not available"}
LATEST_PERIOD = {"Quarterly": "quarter", "Annual": "year"}


def status_table(schedule: pd.DataFrame, ind_cfg: dict, month: pd.Timestamp) -> str:
    """Per-indicator status for the provisional month, grouped by driver."""
    labels = {f"{d}::{i['id']}": i.get("label") or i["id"]
              for d, spec in ind_cfg["drivers"].items() for i in spec["indicators"]}
    order = {"pending": 0, "carried": 1, "reported": 2, "not_started": 3}
    rows = []
    for driver, block in schedule.groupby("driver", sort=False):
        spec = ind_cfg["drivers"][driver]
        started = block[block["status"] != "not_started"]["weight"].sum()
        rep = block[block["status"] == "reported"]["weight"].sum()
        share = rep / started if started else 0
        rows.append(f"{spec.get('label', driver)}: {share:.0%} released")
        block = block.assign(o=block["status"].map(order)).sort_values(["o", "weight"], ascending=[True, False])
        for r in block.itertuples():
            name = esc(labels.get(r.key, r.id))
            if r.anchor:
                name += (f'<br><span style="color:{INK_2};font-size:0.8rem">'
                         f'Needed to confirm the month</span>')
            status = STATUS_TEXT.get(r.status, r.status)
            if r.status == "carried" and not pd.isna(r.last_month):
                status = f"Carried from {r.last_month:%B}"
            if r.status == "reported" and r.frequency in LATEST_PERIOD:
                status = f"Released, latest {LATEST_PERIOD[r.frequency]} held"
            expected = "–" if pd.isna(r.expected) else f"~{day_month(r.expected, short=True)}"
            rows.append([Raw(name), r.frequency, status, expected, f"{r.weight:.0%}"])
    return table(["Indicator", "Frequency", f"Status for {month:%B}", "Expected", "Weight"],
                 rows, numeric={4})


def run_length(series: pd.Series) -> int:
    """How many trailing rows share the last value."""
    s = series.dropna()
    if s.empty:
        return 0
    last = s.iloc[-1]
    n = 0
    for v in reversed(s.tolist()):
        if v != last:
            break
        n += 1
    return n


# ---------- signpost chart ----------

def signpost_html(drivers: pd.DataFrame, ind_cfg: dict, reg_cfg: dict,
                  as_of: pd.Timestamp, then: pd.Timestamp | None,
                  provisional: dict | None = None) -> str:
    regimes = reg_cfg["regimes"]
    then_label = f"{then:%B %Y}" if then is not None else ""
    legend = [
        f'<span><span style="color:{INK};font-size:1.15rem;line-height:1">&#9733;</span>'
        f'{as_of:%B %Y}, confirmed</span>',
    ]
    prov_scores = provisional["drivers"] if provisional else None
    if provisional:
        legend.append(
            f'<span><span class="sp-prov-key">&#9734;</span>'
            f'{provisional["month"]:%B %Y}, provisional</span>')
    if then is not None:
        legend.append(
            f'<span><span class="sp-then" style="position:static;transform:none;'
            f'display:inline-block;background:#fff"></span>A year earlier, {then_label}</span>')
    # Markers on the first row, regime codes on the second, so neither wraps mid-set.
    regime_key = [
        f'<span><span class="sp-chip" style="background:{spec["color"]};'
        f'color:{text_on(spec["color"])}">{esc(spec.get("short", ""))}</span>'
        f'{esc(spec["label"])}</span>'
        for spec in regimes.values()]

    rows = []
    for name, driver in ind_cfg["drivers"].items():
        if name not in drivers.columns:
            continue
        sp = driver.get("signpost") or {}
        reverse = bool(sp.get("reverse", False))
        low = sp.get("low") or {"label": "Low", "tone": "neutral"}
        high = sp.get("high") or {"label": "High", "tone": "neutral"}
        left, right = (high, low) if reverse else (low, high)

        now = drivers.at[as_of, name]
        prior = drivers.at[then, name] if then is not None else float("nan")

        marks = ['<div class="sp-zero" title="Neutral"></div>']
        if not pd.isna(prior):
            marks.append(f'<div class="sp-then" style="left:{track_position(prior, reverse):.2f}%" '
                         f'title="{esc(then_label)}: {signed(prior)}"></div>')
        if not pd.isna(now):
            marks.append(f'<div class="sp-star" style="left:{track_position(now, reverse):.2f}%" '
                         f'title="{as_of:%B %Y}, confirmed: {signed(now)}">&#9733;</div>')
        prov = prov_scores.get(name, float("nan")) if prov_scores is not None else float("nan")
        if not pd.isna(prov):
            marks.append(f'<div class="sp-star sp-star-prov" '
                         f'style="left:{track_position(prov, reverse):.2f}%" '
                         f'title="{provisional["month"]:%B %Y}, provisional: {signed(prov)}">&#9734;</div>')

        points = [(track_position(spec["archetype"][name], reverse), key)
                  for key, spec in regimes.items() if name in spec.get("archetype", {})]
        for pos, keys in cluster(points):
            chips = "".join(
                f'<span class="sp-chip" style="background:{regimes[k]["color"]};'
                f'color:{text_on(regimes[k]["color"])}" '
                f'title="{esc(regimes[k]["label"])} archetype: '
                f'{signed(regimes[k]["archetype"][name])}">'
                f'{esc(regimes[k].get("short", k))}</span>'
                for k in keys)
            left_pct = min(max(pos, 9.0), 91.0)
            marks.append(f'<div class="sp-group" style="left:{left_pct:.2f}%">{chips}</div>')

        def end(spec, side):
            fill, fg = TONES.get(spec.get("tone", "neutral"), TONES["neutral"])
            return (f'<div class="sp-end {side}" style="background:{fill};color:{fg}">'
                    f'{esc(spec["label"])}</div>')

        read = "No reading" if pd.isna(now) else f"{as_of:%b} {signed(now)}"
        if not pd.isna(prov):
            read += f" · {provisional['month']:%b}* {signed(prov)}"
        if not pd.isna(prior):
            read += f" · 12m ago {signed(prior)}"
        rows.append(
            f'<div class="sp-row"><div><div class="sp-pill">{esc(driver.get("label", name))}</div>'
            f'<div class="sp-read">{read}</div></div>'
            f'<div class="sp-scale">{end(left, "left")}'
            f'<div class="sp-track">{"".join(marks)}</div>{end(right, "right")}</div></div>'
        )

    reversed_names = [d.get("label", n) for n, d in ind_cfg["drivers"].items()
                      if (d.get("signpost") or {}).get("reverse")]
    note = ""
    if reversed_names:
        names = join_words([reversed_names[0]] + [n.lower() for n in reversed_names[1:]])
        note = (f'<div class="sp-note">{esc(names)} '
                f'{"is" if len(reversed_names) == 1 else "are"} drawn with the '
                f'tighter end on the left. Scores still count restrictive as positive.</div>')
    return (f'<div class="sp"><div class="sp-legend">{"".join(legend)}</div>'
            f'<div class="sp-legend sp-legend-regimes">{"".join(regime_key)}</div>'
            + "".join(rows) + note + "</div>")


# ---------- charts ----------

def style(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(font=BODY_FONT, background="#ffffff")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=INK_2, gridColor=GRID,
                            domainColor=AXIS, tickColor=AXIS, labelFontSize=11,
                            titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=INK, titleColor=INK_2, labelFontSize=12,
                              symbolStrokeWidth=3, orient="top", title=None)
            .configure_header(labelColor=INK, labelFontSize=13, labelFont=BODY_FONT,
                              labelFontWeight="bold", title=None))


def month_mid(index) -> pd.DatetimeIndex:
    """Where to plot monthly values: mid-month, not the month-end they are stored on.

    Stored on the last day, December lands on the January line and every month
    reads a month late against the year ticks.
    """
    return pd.DatetimeIndex(index).to_period("M").to_timestamp() + pd.Timedelta(days=14)


def month_start(index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).to_period("M").to_timestamp()


def time_x(field: str, start, end, months: tuple[int, ...] = (1,), fmt: str = "%Y",
           every_years: int | None = None) -> alt.X:
    """A temporal x encoding with ticks on real period starts, labeled to their right.

    Dates reach the browser as UTC, so the scale and labels are UTC too: a
    viewer's time zone can never move a month into the one before.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    span = end.year - start.year + 1
    step = every_years or (1 if span <= 16 else 2 if span <= 30 else 5)
    values = [alt.DateTime(year=y, month=m, date=1, utc=True)
              for y in range(start.year, end.year + 2) if y % step == 0 or step == 1
              for m in months
              if start <= pd.Timestamp(year=y, month=m, day=1) <= end + pd.Timedelta(days=31)]
    return alt.X(f"{field}:T", title=None, scale=alt.Scale(type="utc"),
                 axis=alt.Axis(values=values, format=fmt, labelAlign="left", labelPadding=4,
                               labelOffset=3, ticks=True, tickSize=6, tickColor=AXIS,
                               domain=True, domainColor=AXIS, grid=False, labelFlush=False))


def month_tip(field: str, title: str = "Month") -> alt.Tooltip:
    # A utc time unit makes Vega format in UTC; formatType="utc" on a temporal
    # tooltip renders NaN.
    return alt.Tooltip(field, timeUnit="utcyearmonth", type="temporal", title=title,
                       format="%B %Y")


FIT_WORDS = {"clear": "Clear", "weak": "Weak", "none": "None"}
FIT_PHRASE = {"clear": "clear fit", "weak": "weak fit", "none": "no fit"}


def fit_grade(calls: pd.DataFrame) -> pd.Series:
    """Each month's fit grade: clear, weak or none. All clear with the gate off."""
    if "fit" in calls:
        return calls["fit"]
    return pd.Series("clear", index=calls.index, dtype=object)


MID_BAND = 0.15   # scores inside this read as "close to normal"

HORIZON_SHORT = {"short": "ST", "medium": "MT", "long": "LT"}
HORIZON_LABEL = {"short": "Short", "medium": "Medium", "long": "Long"}
HORIZON_ORDER = ("short", "medium", "long")
# Horizon is ordered, not categorical, so it takes one hue getting darker
# rather than three competing colors. A neutral slate, because green, blue, red
# and amber already mean regimes here and red and blue mean direction: a fourth
# color system would collide with all of them. The letters stay inside the
# chip, so the color is never carrying the meaning on its own.
HORIZON_CHIP = {"short": ("#eef1f4", INK_2), "medium": ("#c3ccd6", INK),
                "long": ("#5a6875", "#ffffff")}


def horizon_chip(horizon: str) -> str:
    bg, fg = HORIZON_CHIP.get(horizon, HORIZON_CHIP["medium"])
    return (f'<span title="{HORIZON_LABEL.get(horizon, "Medium")} horizon" '
            f'style="background:{bg};color:{fg};font-size:0.72rem;font-weight:700;'
            f'letter-spacing:0.02em;padding:0.1rem 0.35rem;border-radius:2px;'
            f'white-space:nowrap">{HORIZON_SHORT.get(horizon, "MT")}</span>')


def horizon_bar(ind_cfg: dict, driver: str, width: int = 110) -> str:
    """The driver's horizon mix as one small stacked bar, light to dark.

    Segments run short to long so the bar itself reads as the ramp: the further
    right, the further out it looks. Two pixels of surface between segments so
    neighbouring shades do not merge into one block.
    """
    mix = horizon_mix(ind_cfg, driver)
    parts = "".join(
        f'<span title="{HORIZON_LABEL[h]} {mix[h]:.0%}" style="flex:{mix[h]:.4f};'
        f'background:{HORIZON_CHIP[h][0]};border-radius:1px"></span>'
        for h in HORIZON_ORDER if mix[h] > 0.004)
    return (f'<span style="display:inline-flex;gap:2px;width:{width}px;height:9px;'
            f'vertical-align:middle">{parts}</span>')


def horizons(ind_cfg: dict) -> dict:
    """Indicator id to horizon, from meta.horizons. Anything unlisted is medium."""
    listed = ((ind_cfg.get("meta") or {}).get("horizons") or {})
    return {i: h for h, ids in listed.items() for i in (ids or [])}


def horizon_mix(ind_cfg: dict, driver: str) -> dict:
    """Share of a driver's weight by horizon, for saying how far ahead it looks."""
    tags = horizons(ind_cfg)
    inds = (ind_cfg["drivers"].get(driver) or {}).get("indicators", [])
    total = sum(float(i.get("weight", 0.0)) for i in inds) or 1.0
    mix = {h: 0.0 for h in HORIZON_ORDER}
    for i in inds:
        mix[tags.get(i["id"], "medium")] += float(i.get("weight", 0.0)) / total
    return mix


def horizon_text(ind_cfg: dict, driver: str) -> str:
    """'MT 55 / LT 25 / ST 20', heaviest first, zero shares dropped."""
    mix = {h: v for h, v in horizon_mix(ind_cfg, driver).items() if v >= 0.005}
    return " / ".join(f"{HORIZON_SHORT[h]} {v:.0%}".replace("%", "")
                      for h, v in sorted(mix.items(), key=lambda kv: -kv[1]))


def driver_phrase(driver: str, score: float, ind_cfg: dict) -> str:
    """How one driver reads in a sentence: 'capex is expanding'.

    Written in config next to the signpost labels, so the wording lives with
    the driver definition rather than in code, and a second country can say
    something different.
    """
    sp = (ind_cfg["drivers"].get(driver) or {}).get("signpost") or {}
    label = (ind_cfg["drivers"].get(driver) or {}).get("label", driver).lower()
    if pd.isna(score):
        return f"{label} is not scored"
    if score > MID_BAND:
        end = sp.get("high") or {}
    elif score < -MID_BAND:
        end = sp.get("low") or {}
    else:
        end = sp.get("mid") or {}
    return end.get("phrase") or f"{label} is {end.get('label', 'mixed').lower()}"


def indicator_evidence(parts: pd.DataFrame, indicators: list[dict]) -> str:
    """The reading behind a driver, for a sentence: 'capex orders up 6.6% on a year earlier'.

    Takes whichever indicator moved the driver most this month. Labels carry
    their own transform ("Core capital goods orders, year over year"), which
    reads badly mid-sentence, so the label is cut at the first comma and the
    transform supplies the wording instead.
    """
    if parts is None or parts.empty:
        return ""
    live = parts.dropna(subset=["contribution"])
    if live.empty:
        return ""
    row = live.loc[live["contribution"].abs().idxmax()]
    spec = next((i for i in indicators if i["id"] == row["id"]), None)
    if spec is None or pd.isna(row["value"]):
        return ""
    # Only the first letter drops: "Fed funds rate less Taylor rule rate" keeps
    # its proper nouns.
    base = (spec.get("label") or row["id"]).split(",")[0]
    base = base[0].lower() + base[1:]
    value, transform = float(row["value"]), spec.get("transform", "level")
    if transform == "yoy_pct":
        return f"{base} {'up' if value >= 0 else 'down'} {abs(value):.1f}% on a year earlier"
    if transform == "pct_change_3m_ann":
        return f"{base} running at {format_reading(value, transform)} annualized"
    if transform == "diff_12m":
        return f"{base} {format_reading(value, transform)} over the past year"
    return f"{base} at {format_reading(value, transform)}"


def _gaps(row: pd.Series, reg_cfg: dict, regime: str) -> pd.Series:
    """Weighted squared distance from one regime's archetype, per driver."""
    arche = reg_cfg["regimes"][regime]["archetype"]
    salience = reg_cfg.get("driver_salience", {})
    return pd.Series({d: (float(row[d]) - float(a)) ** 2 * float(salience.get(d, 1.0))
                      for d, a in arche.items() if d in row and pd.notna(row[d])})


def why_called(row: pd.Series, ind_cfg: dict, reg_cfg: dict, regime: str,
               weak: bool = False, evidence: dict | None = None) -> str:
    """Why this month looks like this regime, in two or three sentences.

    Opens with what the regime needs and the month has, backed by the reading
    that moved that driver most. Then covers the rest of the picture, the
    drivers saying something whether or not they support the call, so a reader
    sees where inflation and policy sit rather than one flattering fact. The
    driver arguing hardest against is left to objection(), which the page puts
    last. Distances stay in the drill-down and the methodology.
    """
    if regime not in reg_cfg["regimes"] or row is None:
        return ""
    gaps = _gaps(row, reg_cfg, regime)
    if gaps.empty:
        return ""
    arche = reg_cfg["regimes"][regime]["archetype"]
    label = reg_cfg["regimes"][regime]["label"]
    # A driver supports the call only if the regime takes a real position on it
    # *and* the month is meaningfully closer to that position than a blank
    # reading would be. Without the second test a driver sitting nowhere near
    # the archetype gets listed as a reason for the call.
    salience = reg_cfg.get("driver_salience", {})
    def neutral_gap(d):
        return float(arche[d]) ** 2 * float(salience.get(d, 1.0))
    committed = [d for d in gaps.index if abs(float(arche[d])) >= 0.25]
    support = sorted([d for d in committed if gaps[d] < 0.5 * neutral_gap(d)],
                     key=lambda d: gaps[d])[:3]
    against = gaps.idxmax()

    # Plain text, not HTML; the caller escapes it.
    evidence = evidence or {}
    said = set()
    if support:
        first = f"{label} because {driver_phrase(support[0], row[support[0]], ind_cfg)}"
        if evidence.get(support[0]):
            first += f", with {evidence[support[0]]}"
        said.add(support[0])
    else:
        first = f"Nearest to {label}, though no driver is close to what it expects"
    sentences = [f"{first}."]

    # Then the rest of the picture, loudest first, so inflation and policy get
    # said whether or not they flatter the call. The objection is held back.
    said.add(against)
    sentences += _picture(row, ind_cfg, gaps.index, said, limit=2)
    # Kept as its own sentence: tacked onto the first one it collides with the
    # reading quoted there.
    if weak:
        sentences.append("Overall the match is loose.")
    return " ".join(sentences)


def _picture(row: pd.Series, ind_cfg: dict, drivers, said: set, limit: int = 2) -> list[str]:
    """Sentences covering the drivers not yet mentioned, loudest first."""
    others = sorted([d for d in drivers if d not in said],
                    key=lambda d: abs(float(row[d])), reverse=True)
    out = []
    for group in (others[:2], others[2:3]):
        if not group or len(out) >= limit:
            break
        clause = join_words([driver_phrase(d, row[d], ind_cfg) for d in group])
        out.append(f"{clause[0].upper()}{clause[1:]}.")
    return out


def objection(row: pd.Series, ind_cfg: dict, reg_cfg: dict, regime: str,
              evidence: dict | None = None) -> str:
    """The driver arguing hardest against the call, as a sentence.

    Kept apart from why_called so a page can put it after the call's history
    rather than interrupting the reason with it.
    """
    if regime not in reg_cfg["regimes"] or row is None:
        return ""
    gaps = _gaps(row, reg_cfg, regime)
    if gaps.empty or gaps.max() < 0.05:
        return ""
    # The phrase is already a clause, so it is introduced rather than inflected.
    worst = gaps.idxmax()
    text = f"The main argument against: {driver_phrase(worst, row[worst], ind_cfg)}"
    if (evidence or {}).get(worst):
        text += f", with {evidence[worst]}"
    return f"{text}."


def why_not_called(row: pd.Series, ind_cfg: dict, reg_cfg: dict, regime: str) -> str:
    """Why a month fits nothing, named by the two drivers furthest from the nearest regime."""
    if regime not in reg_cfg["regimes"] or row is None:
        return ""
    gaps = _gaps(row, reg_cfg, regime)
    if gaps.empty:
        return ""
    worst = list(gaps.sort_values(ascending=False).index[:2])
    label = reg_cfg["regimes"][regime]["label"]
    first = (f"Nearest is {label}, but "
             + join_words([driver_phrase(d, row[d], ind_cfg) for d in worst]) + ".")
    # The rest of the picture still gets said: a month that fits nothing is
    # still doing something.
    return " ".join([first] + _picture(row, ind_cfg, gaps.index, set(worst), limit=2))


def what_moved(now: pd.Series, before: pd.Series, ind_cfg: dict,
               min_move: float = 0.08) -> str:
    """What changed between two months' driver scores, and what did not."""
    if now is None or before is None:
        return ""
    shared = [d for d in now.index if d in before.index
              and pd.notna(now[d]) and pd.notna(before[d]) and d in ind_cfg["drivers"]]
    if not shared:
        return ""
    delta = pd.Series({d: float(now[d]) - float(before[d]) for d in shared})
    movers = delta[delta.abs() >= min_move].sort_values(key=abs, ascending=False)[:2]
    steady = [d for d in shared if d not in movers.index]
    labels = {d: ind_cfg["drivers"][d].get("label", d).lower() for d in shared}

    if movers.empty:
        # Say what is holding the reading up, not merely that nothing moved:
        # that is the reason the call has not changed.
        held = delta.abs().sort_values().index[:3]
        return ("little has moved: " + join_words([labels[d] for d in held])
                + " are all close to where they were")
    parts = []
    for d, v in movers.items():
        phrase, verb = driver_phrase(d, now[d], ind_cfg), "rising" if v > 0 else "easing"
        # Most phrases name the driver already, so saying it twice reads badly.
        parts.append(f"{phrase} after {verb}" if phrase.lower().startswith(labels[d])
                     else f"{labels[d]} has {'risen' if v > 0 else 'eased'}, so {phrase}")
    moved = join_words(parts)
    if steady:
        rest = (join_words([labels[d] for d in steady]) + " are"
                if len(steady) <= 3 else "the other drivers are")
        moved += f", while {rest} little changed"
    return moved


def regime_runs(called: pd.Series) -> pd.DataFrame:
    """Collapse a monthly called-regime series into contiguous spans."""
    s = called.dropna()
    rows = []
    start = prev = None
    last_date = None
    for date, name in s.items():
        if name != prev:
            if prev is not None:
                rows.append((start, date, prev))
            start, prev = date, name
        last_date = date
    if prev is not None:
        rows.append((start, last_date + pd.offsets.MonthEnd(1), prev))
    return pd.DataFrame(rows, columns=["start", "end", "regime"])


def history_chart(probs: pd.DataFrame, calls: pd.DataFrame, reg_cfg: dict) -> alt.Chart:
    names = [n for n in probs.columns]
    labels = {n: reg_cfg["regimes"][n]["label"] for n in names}
    domain = [labels[n] for n in names] + [label for label, _ in CALL_STATES.values()]
    colors = [reg_cfg["regimes"][n]["color"] for n in names] + [c for _, c in CALL_STATES.values()]

    wide = probs.copy()
    wide.index = month_mid(wide.index)
    wide.index.name = "date"
    wide = wide.reset_index()
    wide["called"] = calls["called"].reindex(probs.index).map(
        lambda c: call_label(c, reg_cfg)).to_numpy()
    grade = fit_grade(calls).reindex(probs.index)
    wide["fit"] = grade.map(FIT_WORDS).to_numpy()
    long = wide.melt(id_vars=["date", "called", "fit"], value_vars=names,
                     var_name="regime", value_name="probability")
    long["label"] = long["regime"].map(labels)
    # Stack in config order with the first regime on top, so the legend reads
    # in the same order as the bands.
    long["stack"] = long["regime"].map({n: i for i, n in enumerate(names)})

    x = time_x("date", month_start(probs.index)[0], probs.index[-1])
    color = alt.Color("label:N", scale=alt.Scale(domain=domain, range=colors),
                      legend=alt.Legend(values=[labels[n] for n in names],
                                        orient="bottom", columns=2, offset=14,
                                        symbolType="square"))
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover",
                                clear="pointerout", empty=False)

    # Probabilities sum to one, so the bands are the whole: part-to-whole over
    # time. A 1px surface-colored edge separates neighboring bands.
    areas = alt.Chart(long).mark_area(opacity=0.9, stroke="#ffffff", strokeWidth=1).encode(
        x=x,
        y=alt.Y("probability:Q", title=None, stack="zero", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], domain=False,
                              ticks=False)),
        color=color,
        order=alt.Order("stack:Q", sort="descending"))
    tooltip = [month_tip("date"), alt.Tooltip("called:N", title="Called"),
               alt.Tooltip("fit:N", title="Fit to nearest regime")] + [
        alt.Tooltip(f"{n}:Q", title=labels[n], format=".0%") for n in names]
    rule = alt.Chart(wide).mark_rule(color=INK, strokeWidth=1).encode(
        x="date:T", opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
        tooltip=tooltip).add_params(hover)

    # The called-regime band is drawn in pixel space above the plot, inside the
    # same layer, so it shares the x scale and the chart can fill its container.
    # Months called a regime on only a weak fit are drawn paler, so the band
    # separates a clear regime from a lean.
    called = calls["called"].reindex(probs.index)
    regime_call = called.isin(names)
    weak = regime_call & grade.isin(["weak", "none"])
    runs = regime_runs(called.where(called.isna(), called.astype(str) + weak.map({True: "|weak", False: ""})))
    runs["weak"] = runs["regime"].str.endswith("|weak")
    runs["regime"] = runs["regime"].str.removesuffix("|weak")
    runs["label"] = runs["regime"].map(lambda c: call_label(c, reg_cfg))
    runs["fit"] = runs["weak"].map({True: "Weak", False: "Clear"}).where(runs["regime"].isin(names), "")
    # Runs cover whole months: from the first day of the first month to the
    # first day of the month after the last.
    runs["start"] = month_start(runs["start"])
    runs["end"] = month_start(runs["end"])
    runs["until"] = runs["end"] - pd.Timedelta(days=1)
    ribbon = alt.Chart(runs).mark_rect().encode(
        x=alt.X("start:T", scale=alt.Scale(type="utc")), x2="end:T",
        y=alt.value(-26), y2=alt.value(-12),
        color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=colors)),
        opacity=alt.condition(alt.datum.weak, alt.value(0.4), alt.value(1.0)),
        tooltip=[alt.Tooltip("label:N", title="Called"), alt.Tooltip("fit:N", title="Fit"),
                 month_tip("start", "From"), month_tip("until", "Until")])
    ribbon_label = alt.Chart(pd.DataFrame({"t": ["Called regime"]})).mark_text(
        align="left", baseline="bottom", fontSize=11, color=INK_2, font=BODY_FONT,
    ).encode(text="t:N", x=alt.value(0), y=alt.value(-30))

    return style(alt.layer(ribbon, ribbon_label, areas, rule).properties(
        # At container width Vega fits the whole chart, legend and band
        # included, into this height; about 300px of it is plot.
        height=430, width="container", padding={"top": 38, "left": 5, "right": 10}))


def drivers_chart(drivers: pd.DataFrame, ind_cfg: dict) -> alt.Chart:
    names = [c for c in drivers.columns if not c.endswith("__coverage")]
    labels = {n: ind_cfg["drivers"].get(n, {}).get("label", n) for n in names}
    long = drivers[names].copy()
    first, last = month_start(long.index)[0], long.index[-1]
    long.index = month_mid(long.index)
    long.index.name = "date"
    long = long.reset_index().melt(id_vars="date", var_name="driver", value_name="score")
    long["driver"] = long["driver"].map(labels)

    # Small panels: one labeled tick every five years keeps labels from colliding.
    base = alt.Chart().encode(x=time_x("date", first, last, every_years=5))
    line = base.mark_line(strokeWidth=2, color=INK).encode(
        y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[-1, 1]),
                axis=alt.Axis(values=[-1, -0.5, 0, 0.5, 1], domain=False, ticks=False)),
        tooltip=[month_tip("date"), alt.Tooltip("score:Q", title="Score", format="+.2f")])
    zero = alt.Chart().mark_rule(color=AXIS).encode(y=alt.datum(0))
    return style(alt.layer(zero, line, data=long).properties(width=290, height=150).facet(
        facet=alt.Facet("driver:N", sort=[labels[n] for n in names], title=None),
        columns=3, spacing={"row": 22, "column": 26}))


# ---------- drill-down: one regime x driver cell ----------

TRANSFORM_TEXT = {
    "level": "Level, as published",
    "yoy_pct": "Change from a year earlier, %",
    "pct_change_3m_ann": "Change over 3 months, annualized %",
    "pct_change_5y_ann": "Change over 5 years, annualized %",
    "diff_12m": "Change from a year earlier, in units",
    "sum_12m": "Sum over the last twelve months",
    "dev_5y_pct": "Difference from its own five-year average, %",
}
UP, DOWN = "#e34948", "#2a78d6"   # diverging poles: pushes the driver up / down


def format_reading(value: float, transform: str) -> str:
    if pd.isna(value):
        return "–"
    if transform in ("yoy_pct", "pct_change_3m_ann"):
        return f"{value:.1f}%".replace("-", "−")
    if transform == "diff_12m":
        return signed(value)
    return f"{value:,.2f}".replace("-", "−")


def gap_chart(drivers: pd.DataFrame, driver: str, archetype: float, regime_label: str,
              color: str, months: int = 36) -> alt.Chart:
    """The driver's recent path against where one regime expects it to sit."""
    df = drivers[[driver]].tail(months).rename(columns={driver: "score"})
    first, last = month_start(df.index)[0], df.index[-1]
    df.index = month_mid(df.index)
    df.index.name = "date"
    df = df.reset_index()
    df["archetype"] = archetype
    df["gap"] = df["score"] - archetype
    x = time_x("date", first, last, months=(1, 7), fmt="%b %Y")
    y = alt.Y("score:Q", title=None, scale=alt.Scale(domain=[-1, 1]),
              axis=alt.Axis(values=[-1, -0.5, 0, 0.5, 1], domain=False, ticks=False))
    wash = alt.Chart(df).mark_area(opacity=0.12, color=color).encode(
        x=x, y=y, y2="archetype:Q")
    zero = alt.Chart(df).mark_rule(color=AXIS).encode(y=alt.datum(0))
    target = alt.Chart(df).mark_rule(color=color, strokeWidth=2).encode(y="archetype:Q")
    label = alt.Chart(df.tail(1)).mark_text(
        align="right", baseline="bottom", dy=-4, fontSize=11, color=INK_2, font=BODY_FONT,
    ).encode(x="date:T", y="archetype:Q",
             text=alt.value(f"{regime_label} expects {signed(archetype)}"))
    line = alt.Chart(df).mark_line(strokeWidth=2, color=INK).encode(
        x=x, y=y,
        tooltip=[month_tip("date"),
                 alt.Tooltip("score:Q", title="Driver score", format="+.2f"),
                 alt.Tooltip("gap:Q", title="Gap to regime", format="+.2f")])
    end = alt.Chart(df.tail(1)).mark_circle(size=70, color=INK, stroke="#ffffff",
                                            strokeWidth=2, opacity=1).encode(x=x, y=y)
    return style(alt.layer(wash, zero, target, label, line, end).properties(
        height=210, width="container"))


def breakdown_table(parts: pd.DataFrame, indicators: list[dict], clipped_to: float | None,
                    horizon_of: dict | None = None) -> str:
    """Indicators behind one driver this month, heaviest weight first.

    Ordered by weight rather than by contribution: the reader's first question
    is which indicators the driver leans on, and a contribution ordering hides
    that by putting whichever indicator happened to move at the top. Within one
    weight, the largest contribution leads; anything with no data this month
    sinks to the bottom.
    """
    spec = {i["id"]: i for i in indicators}
    parts = parts.assign(
        by_weight=parts["share"].fillna(-1),
        by_move=parts["contribution"].abs().fillna(-1),
    ).sort_values(["by_weight", "by_move"], ascending=False)
    biggest = parts["contribution"].abs().max()
    biggest = biggest if biggest and not pd.isna(biggest) else 1.0

    rows = []
    for r in parts.itertuples():
        ind = spec[r.id]
        label = esc(ind.get("label") or r.id)
        if (horizon_of or {}).get(r.id):
            label += " " + horizon_chip(horizon_of[r.id])
        if int(ind["direction"]) < 0:
            label += (f'<br><span style="color:{INK_2};font-size:0.8rem">'
                      f'Inverted: higher pulls down</span>')
        transform = ind.get("transform", "level")
        norm = ind["normalize"]
        center = (f'{float(norm["center"]):g}'.replace("-", "−")
                  if norm.get("method") == "gap"
                  else f'Own {int(norm.get("window", 240)) / 12:g}-year history')
        reading = (f'{format_reading(r.value, transform)}<br><span style="color:{INK_2};'
                   f'font-size:0.8rem">{esc(TRANSFORM_TEXT.get(transform, transform))}</span>')
        if pd.isna(r.contribution):
            bar, contrib = "", "No data"
        else:
            half = abs(r.contribution) / biggest * 50
            side = "left:50%" if r.contribution >= 0 else f"left:{50 - half:.1f}%"
            fill = UP if r.contribution >= 0 else DOWN
            bar = (f'<div style="position:relative;height:10px;width:120px;background:{TRACK};'
                   f'border-radius:2px;display:inline-block;vertical-align:middle;margin-right:8px">'
                   f'<div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:{AXIS}"></div>'
                   f'<div style="position:absolute;{side};width:{half:.1f}%;top:0;bottom:0;'
                   f'background:{fill}"></div></div>')
            contrib = signed(r.contribution)
        rows.append([Raw(label), Raw(reading), center,
                     "–" if pd.isna(r.score) else signed(r.score),
                     "–" if pd.isna(r.share) else f"{r.share:.0%}",
                     Raw(f'<span style="white-space:nowrap">{bar}{contrib}</span>')])

    total = parts["contribution"].sum(min_count=1)
    total_text = "–" if pd.isna(total) else signed(total)
    if clipped_to is not None and not pd.isna(total) and abs(total) > 1:
        total_text += f" → clipped to {signed(clipped_to)}"
    rows.append([Raw("<b>Driver score</b>"), "", "", "", "100%",
                 Raw(f'<b style="white-space:nowrap">{total_text}</b>')])
    return table(["Indicator", "Latest reading", "Center", "Score", "Weight",
                  "Contribution to driver"], rows, numeric={3, 4})
