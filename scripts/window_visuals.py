"""Two explainer charts for the steadier nowcast, for emails and slides.

    python scripts/window_visuals.py [--out FOLDER] [--month 2023-02]

1. How the 18-month window turns one month into a label, with that month's
   real numbers (the twelve-month changes in core PCE and GDP inside the window,
   averaged, against expectations and potential).
2. How the 12-month minimum regime length absorbs a short dip (Feb-Oct 1995).

Writes PNGs to FOLDER (default: output/ in the repo, which git ignores).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.window_sweep import load_inputs  # noqa: E402
from src import truth  # noqa: E402

INK, MUTED, LINE = "#1b2127", "#5f6a74", "#d5dade"
PICK, CUR = "#1f6f5c", "#8a949c"
COL = {"goldilocks": "#008300", "high_growth_high_inflation": "#2a78d6", "stagflation": "#e34948",
       "hard_landing": "#eda100"}
NAME = {"goldilocks": "Goldilocks", "high_growth_high_inflation": "High growth, high inflation",
        "stagflation": "Stagflation", "hard_landing": "Hard landing"}
SPAN = {6: "half-year", 9: "three-quarter", 12: "four-quarter"}
BOX = dict(facecolor="white", edgecolor="none", pad=2.5)
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "text.color": INK})


def quarterly(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.groupby(s.index.asfreq("Q")).mean()


def window_chart(m: pd.DataFrame, t: pd.Period, path: str) -> None:
    steady = truth.VERSIONS["now"]["steadier"]
    small, big = truth.AVERAGED[(steady["years"], "centered")]
    half = big // 2
    ax_ = truth.axes(m, steady["years"], "centered", steady["measure"])
    lab = truth.build(m, **steady)
    band = truth.band(steady["years"])
    c = m["core_pce"]
    a, b = t - half, t + half
    ends = pd.period_range(a + small, b, freq="M")          # each change ends here
    rates = pd.Series({e: ((c[e] / c[e - small]) ** (12 / small) - 1) * 100 for e in ends})
    infl = ax_.loc[t, "core_pce"]
    q, qs = t.asfreq("Q"), small // 3
    q_ends = pd.period_range(q - big // 6 + qs, q + big // 6, freq="Q")
    g_rate, p_rate = ax_.loc[t, "gdp_growth"], ax_.loc[t, "potential_growth"]
    gap_i, gap_g = ax_.loc[t, "inflation"], ax_.loc[t, "growth"]
    word = lambda v, up, down: up if v >= band else down if v <= -band else "inside the band, so last month's reading"

    months = pd.period_range(t - 13, t + 13, freq="M")
    x = {p_: i for i, p_ in enumerate(months)}
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(-1, len(months)); ax.set_ylim(-0.5, 8.1); ax.axis("off")
    for p_ in months:
        if p_.month in (1, 4, 7, 10):
            ax.text(x[p_], -0.1, p_.strftime("%b\n%Y") if p_.month == 1 else p_.strftime("%b"),
                    ha="center", va="top", color=MUTED, fontsize=9)
            ax.plot([x[p_], x[p_]], [0.05, 7.5], color=LINE, lw=0.6, zorder=0)
    ax.plot([x[t], x[t]], [0.05, 7.55], color=INK, lw=1.6, ls=(0, (3, 2)), zorder=1)
    ax.text(x[t], 7.7, f"Month being labeled: {t.strftime('%b %Y')}", ha="center", fontsize=11.5, fontweight="bold")

    def label(y, text, color):
        ax.text(-1.6, y, text, ha="right", va="center", fontsize=10, color=color, fontweight="bold", clip_on=False)

    def arrow(y, lo, hi, color, lw=2, scale=14):
        ax.add_patch(FancyArrowPatch((x[lo], y), (x[hi], y), arrowstyle="-|>", mutation_scale=scale,
                                     color=color, lw=lw, shrinkA=0, shrinkB=0))

    # Today: one change, start to end.
    label(6.75, "Today: 12 months\n(6 either side)", CUR)
    ax.add_patch(Rectangle((x[t - 6] - 0.5, 6.5), x[t + 6] - x[t - 6] + 1, 0.5, color=CUR, alpha=0.18, lw=0))
    arrow(6.75, t - 6, t + 6, CUR)
    ax.text(x[t - 6] - 0.4, 7.07, f"One change, start to end: {(t - 6).strftime('%b %Y')} → "
            f"{(t + 6).strftime('%b %Y')}", ha="left", va="bottom", fontsize=9.5, bbox=BOX, zorder=3)

    # Proposed: every small change inside the window, one per row.
    top, step = 5.95, min(0.45, 2.6 / (len(ends) - 1))
    label(top - step * (len(ends) - 1) / 2, f"Proposed: {big} months\n({half} either side),\n"
          f"{len(ends)} {small}-month changes", PICK)
    ax.add_patch(Rectangle((x[a] - 0.5, top - step * (len(ends) - 1) - 0.15), x[b] - x[a] + 1,
                           step * (len(ends) - 1) + 0.3, color=PICK, alpha=0.10, lw=0))
    for k, e in enumerate(ends):
        y = top - k * step
        arrow(y, e - small, e, PICK, lw=1.4, scale=8)
        ax.text(x[e] + 0.35, y, f"{rates[e]:.1f}%", va="center", fontsize=7.5, color=MUTED)
    y_note = top - step * (len(ends) - 1) - 0.55
    ax.text(x[a] - 0.4, y_note, f"Core PCE, each change at a yearly rate: {rates.iloc[0]:.1f}% "
            f"({(ends[0] - small).strftime('%b %Y')}–{ends[0].strftime('%b %Y')}) to {rates.iloc[-1]:.1f}% "
            f"({(ends[-1] - small).strftime('%b %Y')}–{ends[-1].strftime('%b %Y')}).  Average: {infl:.2f}% a year",
            ha="left", va="center", fontsize=9.5, bbox=BOX, zorder=3, fontweight="bold")

    label(1.75, "Then subtract\nwhat was expected", PICK)
    ax.text(-0.5, 1.75, f"Expected inflation, averaged over the same {big} months: {infl - gap_i:.2f}%\n"
            f"Gap: {infl:.2f} − {infl - gap_i:.2f} = {gap_i:+.2f} points, against the ±{band:.2f} band  →  "
            f"{word(gap_i, 'inflation high', 'inflation low')}", ha="left", va="center", fontsize=9.5, bbox=BOX, zorder=3)
    label(0.6, "Growth: same\nwindow on GDP", PICK)
    ax.text(-0.5, 0.6, f"Real GDP, the {len(q_ends)} {SPAN[small]} changes from {q_ends[0] - qs}–{q_ends[0]} to "
            f"{q_ends[-1] - qs}–{q_ends[-1]}: average {g_rate:.2f}% a year, against CBO potential {p_rate:.2f}%\n"
            f"Gap: {gap_g:+.2f} points, against the ±{band:.2f} band  →  {word(gap_g, 'growth up', 'growth down')}",
            ha="left", va="center", fontsize=9.5, bbox=BOX, zorder=3)
    fig.text(0.5, 0.02, f"Label for {t.strftime('%b %Y')}: {NAME[lab.loc[t, 'label']].lower()}.  "
             "A gap inside the band keeps last month's reading.", ha="center", fontsize=10.5,
             fontweight="bold", color=PICK)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.98, bottom=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def minimum_chart(m: pd.DataFrame, path: str) -> None:
    steady = truth.VERSIONS["now"]["steadier"]
    rows = [("Today's rule:\n3-month minimum", 3), ("9-month minimum", 9),
            (f"Proposed:\n{steady['min_months']}-month minimum", steady["min_months"])]
    months = pd.period_range("1994-07", "1996-06", freq="M")
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(-0.6, len(months) - 0.4); ax.set_ylim(-0.55, 3.3); ax.axis("off")
    for r, (name, mm) in enumerate(rows):
        lab = truth.build(m, **{**steady, "min_months": mm})["label"]
        y = 2 - r
        ax.text(-0.8, y + 0.4, name, ha="right", va="center", fontsize=10, fontweight="bold",
                color=PICK if r == 2 else INK)
        for i, p_ in enumerate(months):
            ax.add_patch(Rectangle((i - 0.45, y + 0.05), 0.9, 0.7, color=COL[lab[p_]], lw=0))
    for i, p_ in enumerate(months):
        ax.text(i, -0.12, p_.strftime("%b\n%Y") if p_.month == 1 else p_.strftime("%b"),
                ha="center", va="top", fontsize=9, color=MUTED)
    dip = [i for i, p_ in enumerate(months) if pd.Period("1995-02", "M") <= p_ <= pd.Period("1995-10", "M")]
    ax.add_patch(Rectangle((dip[0] - 0.55, -0.02), len(dip) + 0.1, 2.9, fill=False, ec=INK, lw=1.4, ls=(0, (3, 2))))
    ax.text((dip[0] + dip[-1]) / 2, 3.0, "Feb–Oct 1995: growth dips below potential for 9 months, no recession",
            ha="center", fontsize=10, fontweight="bold")
    fig.legend([Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL], [NAME[k] for k in COL], loc="lower center",
               bbox_to_anchor=(0.58, 0.075), ncol=4, frameon=False, fontsize=9.5)
    fig.text(0.5, 0.02, "A regime shorter than the minimum takes the label before it. The 9-month dip passes a "
             "3- or 9-month rule; the 12-month rule keeps 1995 goldilocks.", ha="center", fontsize=10,
             color=PICK, fontweight="bold")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.92, bottom=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output"))
    ap.add_argument("--month", default="2023-02", help="Month for the window example, YYYY-MM.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    m = load_inputs()
    window_chart(m, pd.Period(args.month, "M"), os.path.join(args.out, "1_how_the_18_month_window_works.png"))
    minimum_chart(m, os.path.join(args.out, "2_how_the_12_month_minimum_works.png"))
    print(f"Wrote two charts to {args.out}")
