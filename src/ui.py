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

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NAVY = "#12203f"
TRACK = "#ecebe6"
TRANSITIONAL = "#b5b3ab"
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
    return f"{x:+.2f}".replace("-", "−")


# ---------- page chrome ----------

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 4rem; }}
.mr-eyebrow {{ font: 600 0.78rem {BODY_FONT}; color: {INK_2};
  letter-spacing: 0.04em; text-transform: uppercase; margin: 0 0 0.25rem; }}
.mr-title {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 2.3rem;
  color: {NAVY}; line-height: 1.1; margin: 0; }}
.mr-sub {{ font: 0.95rem/1.45 {BODY_FONT}; color: {INK_2}; margin: 0.35rem 0 0; }}
.mr-rule {{ border: 0; border-top: 1px solid {GRID}; margin: 1.1rem 0 1.4rem; }}
.mr-h2 {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 1.55rem;
  color: {NAVY}; margin: 0 0 0.2rem; }}
.mr-caption {{ font: 0.88rem/1.45 {BODY_FONT}; color: {INK_2}; margin: 0 0 0.9rem; }}

.mr-call {{ font-family: {HEADING_FONT}; font-weight: 700; font-size: 2.6rem;
  line-height: 1.08; color: {INK}; margin: 0.1rem 0 0.6rem; }}
.mr-call-swatch {{ display: inline-block; width: 0.55em; height: 0.55em;
  border-radius: 50%; margin-right: 0.35em; vertical-align: 0.08em; }}
.mr-lede {{ font: 1rem/1.55 {BODY_FONT}; color: {INK}; margin: 0 0 0.5rem; max-width: 36rem; }}
.mr-lede-muted {{ font: 0.88rem/1.5 {BODY_FONT}; color: {INK_2}; margin: 0; max-width: 36rem; }}
.mr-demo {{ font: 0.85rem {BODY_FONT}; color: {INK}; background: #fdf3dc;
  border-left: 3px solid #eda100; padding: 0.45rem 0.7rem; margin: 0.9rem 0 0; max-width: 36rem; }}

.mr-probs {{ font-family: {BODY_FONT}; }}
.mr-probs-head {{ font: 600 0.78rem {BODY_FONT}; color: {INK_2}; letter-spacing: 0.04em;
  text-transform: uppercase; margin: 0.35rem 0 0.7rem; }}
.mr-prob {{ display: grid; grid-template-columns: minmax(9rem, 13rem) 1fr 3rem;
  align-items: center; gap: 0.75rem; padding: 0.42rem 0; border-bottom: 1px solid {GRID}; }}
.mr-prob:last-child {{ border-bottom: 0; }}
.mr-prob-label {{ font-size: 0.92rem; color: {INK}; }}
.mr-prob-label b {{ font-weight: 700; }}
.mr-prob-code {{ color: {INK_2}; font-size: 0.78rem; margin-left: 0.3rem; }}
.mr-prob-track {{ height: 10px; background: {TRACK}; border-radius: 0 4px 4px 0; }}
.mr-prob-bar {{ height: 10px; border-radius: 0 4px 4px 0; }}
.mr-prob-val {{ font-size: 0.95rem; color: {INK}; text-align: right;
  font-variant-numeric: tabular-nums; }}

.sp {{ font-family: {BODY_FONT}; }}
.sp-legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.2rem; align-items: center;
  font-size: 0.84rem; color: {INK_2}; margin: 0 0 1rem; }}
.sp-legend span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
.sp-row {{ display: grid; grid-template-columns: 12.5rem 1fr; gap: 1.1rem;
  align-items: center; padding: 0.55rem 0; }}
.sp-pill {{ background: {NAVY}; color: #fff; border-radius: 10px; padding: 0.7rem 0.8rem;
  text-align: center; font-weight: 700; font-size: 1.02rem; line-height: 1.2; }}
.sp-read {{ font-size: 0.78rem; color: {INK_2}; text-align: center; margin-top: 0.3rem;
  font-variant-numeric: tabular-nums; }}
.sp-scale {{ display: grid; grid-template-columns: 7.2rem 1fr 7.2rem; min-height: 4.6rem; }}
.sp-end {{ display: flex; align-items: center; justify-content: center; text-align: center;
  font-weight: 700; font-size: 0.86rem; line-height: 1.2; padding: 0.3rem 0.45rem; }}
.sp-end.left {{ border-radius: 6px 0 0 6px; }}
.sp-end.right {{ border-radius: 0 6px 6px 0; }}
.sp-track {{ position: relative; background: {TRACK}; margin: 0 2px; }}
.sp-zero {{ position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: {AXIS}; }}
.sp-star {{ position: absolute; top: 0.2rem; transform: translateX(-50%);
  font-size: 1.45rem; line-height: 1; color: {NAVY}; z-index: 3; }}
.sp-then {{ position: absolute; top: 0.45rem; width: 12px; height: 12px; border-radius: 50%;
  transform: translateX(-50%); border: 2px solid {MUTED}; background: {TRACK}; z-index: 2; }}
.sp-group {{ position: absolute; bottom: 0.45rem; transform: translateX(-50%);
  display: flex; gap: 3px; white-space: nowrap; z-index: 1; }}
.sp-chip {{ font-size: 0.76rem; font-weight: 700; padding: 0.16rem 0.38rem; border-radius: 4px;
  letter-spacing: 0.02em; }}
.sp-note {{ font-size: 0.8rem; color: {INK_2}; margin-top: 0.6rem; }}

.mr-foot {{ font: 0.8rem/1.55 {BODY_FONT}; color: {INK_2}; border-top: 1px solid {GRID};
  margin-top: 2.4rem; padding-top: 0.9rem; }}
.mr-foot b {{ color: {INK}; }}

@media (max-width: 760px) {{
  .sp-row {{ grid-template-columns: 1fr; gap: 0.4rem; }}
  .sp-end {{ font-size: 0.66rem; padding: 0.15rem; overflow-wrap: anywhere; }}
  .sp-scale {{ grid-template-columns: 3.9rem 1fr 3.9rem; }}
  .sp-chip {{ font-size: 0.6rem; padding: 0.1rem 0.2rem; }}
  .sp-group {{ gap: 1px; }}
  .mr-prob {{ grid-template-columns: 8rem 1fr 2.6rem; }}
  .mr-call {{ font-size: 2rem; }}
}}
</style>
"""


def esc(s) -> str:
    return html.escape(str(s))


# ---------- headline ----------

def probability_panel(probs_row: pd.Series, reg_cfg: dict, called: str) -> str:
    """One thin bar per regime in config order, so a regime never changes place."""
    rows = []
    for name, spec in reg_cfg["regimes"].items():
        p = float(probs_row.get(name, float("nan")))
        width = 0 if pd.isna(p) else max(0.0, min(1.0, p)) * 100
        label = f"<b>{esc(spec['label'])}</b>" if name == called else esc(spec["label"])
        rows.append(
            f'<div class="mr-prob" title="{esc(spec["label"])}: {p:.0%}">'
            f'<div class="mr-prob-label">{label}'
            f'<span class="mr-prob-code">{esc(spec.get("short", ""))}</span></div>'
            f'<div class="mr-prob-track"><div class="mr-prob-bar" '
            f'style="width:{width:.1f}%;background:{spec["color"]}"></div></div>'
            f'<div class="mr-prob-val">{p:.0%}</div></div>'
        )
    return ('<div class="mr-probs"><div class="mr-probs-head">Probability this month</div>'
            + "".join(rows) + "</div>")


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
                  as_of: pd.Timestamp, then: pd.Timestamp | None) -> str:
    regimes = reg_cfg["regimes"]
    then_label = f"{then:%B %Y}" if then is not None else ""
    legend = [
        f'<span><span style="color:{NAVY};font-size:1.2rem;line-height:1">&#9733;</span>'
        f'Reading for {as_of:%B %Y}</span>',
    ]
    if then is not None:
        legend.append(
            f'<span><span class="sp-then" style="position:static;transform:none;'
            f'display:inline-block;background:#fff"></span>A year earlier, {then_label}</span>')
    for spec in regimes.values():
        legend.append(
            f'<span><span class="sp-chip" style="background:{spec["color"]};'
            f'color:{text_on(spec["color"])}">{esc(spec.get("short", ""))}</span>'
            f'{esc(spec["label"])}</span>')

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
                         f'title="{as_of:%B %Y}: {signed(now)}">&#9733;</div>')

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

        read = "No reading" if pd.isna(now) else f"Now {signed(now)}"
        if not pd.isna(prior):
            read += f" · a year ago {signed(prior)}"
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
        note = (f'<div class="sp-note">{esc(", ".join(reversed_names))} '
                f'{"is" if len(reversed_names) == 1 else "are"} drawn with the '
                f'tighter end on the left. Scores still count restrictive as positive.</div>')
    return (f'<div class="sp"><div class="sp-legend">{"".join(legend)}</div>'
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
    domain = [labels[n] for n in names] + ["Transitional"]
    colors = [reg_cfg["regimes"][n]["color"] for n in names] + [TRANSITIONAL]

    wide = probs.copy()
    wide.index.name = "date"
    wide = wide.reset_index()
    wide["called"] = calls["called"].reindex(probs.index).map(
        lambda c: labels.get(c, "Transitional")).to_numpy()
    long = wide.melt(id_vars=["date", "called"], value_vars=names,
                     var_name="regime", value_name="probability")
    long["label"] = long["regime"].map(labels)

    x = alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount=8, grid=False))
    color = alt.Color("label:N", scale=alt.Scale(domain=domain, range=colors),
                      legend=alt.Legend(values=[labels[n] for n in names],
                                        orient="bottom", columns=2, offset=14))
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover",
                                clear="pointerout", empty=False)

    lines = alt.Chart(long).mark_line(strokeWidth=2, strokeCap="round",
                                      strokeJoin="round").encode(
        x=x, y=alt.Y("probability:Q", title=None,
                     scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format="%", tickCount=5, domain=False, ticks=False)),
        color=color)
    dots = alt.Chart(long).mark_circle(size=64, stroke="#ffffff", strokeWidth=2).encode(
        x="date:T", y="probability:Q", color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    tooltip = [alt.Tooltip("date:T", title="Month", format="%B %Y"),
               alt.Tooltip("called:N", title="Called")] + [
        alt.Tooltip(f"{n}:Q", title=labels[n], format=".0%") for n in names]
    rule = alt.Chart(wide).mark_rule(color=AXIS, strokeWidth=1).encode(
        x="date:T", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=tooltip).add_params(hover)

    # The called-regime band is drawn in pixel space above the plot, inside the
    # same layer, so it shares the x scale and the chart can fill its container.
    runs = regime_runs(calls["called"].reindex(probs.index))
    runs["label"] = runs["regime"].map(lambda c: labels.get(c, "Transitional"))
    ribbon = alt.Chart(runs).mark_rect().encode(
        x="start:T", x2="end:T", y=alt.value(-26), y2=alt.value(-12),
        color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=colors)),
        tooltip=[alt.Tooltip("label:N", title="Called"),
                 alt.Tooltip("start:T", title="From", format="%B %Y"),
                 alt.Tooltip("end:T", title="Until", format="%B %Y")])
    ribbon_label = alt.Chart(pd.DataFrame({"t": ["Called regime"]})).mark_text(
        align="left", baseline="bottom", fontSize=11, color=INK_2, font=BODY_FONT,
    ).encode(text="t:N", x=alt.value(0), y=alt.value(-30))

    return style(alt.layer(ribbon, ribbon_label, lines, rule, dots).properties(
        # At container width Vega fits the whole chart, legend and band
        # included, into this height; about 300px of it is plot.
        height=430, width="container", padding={"top": 38, "left": 5, "right": 10}))


def drivers_chart(drivers: pd.DataFrame, ind_cfg: dict) -> alt.Chart:
    names = [c for c in drivers.columns if not c.endswith("__coverage")]
    labels = {n: ind_cfg["drivers"].get(n, {}).get("label", n) for n in names}
    long = drivers[names].copy()
    long.index.name = "date"
    long = long.reset_index().melt(id_vars="date", var_name="driver", value_name="score")
    long["driver"] = long["driver"].map(labels)

    base = alt.Chart().encode(x=alt.X("date:T", title=None,
                                      axis=alt.Axis(format="%Y", tickCount=5, grid=False)))
    line = base.mark_line(strokeWidth=2, color=NAVY).encode(
        y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[-1, 1]),
                axis=alt.Axis(values=[-1, -0.5, 0, 0.5, 1], domain=False, ticks=False)),
        tooltip=[alt.Tooltip("date:T", title="Month", format="%B %Y"),
                 alt.Tooltip("score:Q", title="Score", format="+.2f")])
    zero = alt.Chart().mark_rule(color=AXIS).encode(y=alt.datum(0))
    return style(alt.layer(zero, line, data=long).properties(width=290, height=150).facet(
        facet=alt.Facet("driver:N", sort=[labels[n] for n in names], title=None),
        columns=3, spacing={"row": 22, "column": 26}))
