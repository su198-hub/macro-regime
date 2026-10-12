"""Two explainer charts for the steadier nowcast, for emails and slides.

    python scripts/window_visuals.py [--out FOLDER] [--month 2023-02]

1. How the averaging works: the one-year stretches inside the 18-month window
   around one month, core PCE inflation over each, and their average, with the
   start-to-end change for comparison.
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
STRETCH = {6: "six-month", 9: "nine-month", 12: "one-year"}
BOX = dict(facecolor="white", edgecolor="none", pad=2.5)
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "text.color": INK})


def quarterly(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.groupby(s.index.asfreq("Q")).mean()


def averaging_chart(m: pd.DataFrame, t: pd.Period, path: str) -> None:
    """The averaged window for one month: every one-year stretch inside the
    18 months around it, its inflation, and their average, with the single
    start-to-end change underneath for comparison."""
    steady = truth.VERSIONS["now"]["steadier"]
    small, big = truth.AVERAGED[(steady["years"], "centered")]
    half = big // 2
    c = m["core_pce"]
    a, b = t - half, t + half
    ends = pd.period_range(a + small, b, freq="M")
    rates = pd.Series({e: ((c[e] / c[e - small]) ** (12 / small) - 1) * 100 for e in ends})
    whole = ((c[b] / c[a]) ** (12 / big) - 1) * 100

    months = pd.period_range(a - 1, b + 1, freq="M")
    x = {p_: i for i, p_ in enumerate(months)}
    n = len(ends)
    right = len(months) + 0.6                   # where the numbers column starts
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(-0.8, right + 7.2); ax.set_ylim(-2.75, n + 1.55); ax.axis("off")

    # The window and the month being labeled.
    ax.add_patch(Rectangle((x[a] - 0.5, -0.55), x[b] - x[a] + 1, n + 0.75, color=PICK, alpha=0.08, lw=0))
    ax.text((x[a] + x[b]) / 2, n + 0.75, f"{big}-month window: {a.strftime('%b %Y')} to {b.strftime('%b %Y')}, "
            f"{half} months either side of {t.strftime('%b %Y')}", ha="center", fontsize=10.5, fontweight="bold")
    ax.plot([x[t], x[t]], [-0.55, n + 0.2], color=INK, lw=1.3, ls=(0, (3, 2)), zorder=1)
    ax.text(x[t] + 0.25, n - 0.35, t.strftime("%b %Y"), fontsize=9, color=INK, va="center")
    ticks = [p_ for p_ in months if p_.month in (1, 4, 7, 10)]
    for p_ in ticks:
        ax.plot([x[p_], x[p_]], [-1.9, n + 0.2], color=LINE, lw=0.6, zorder=0)
        ax.text(x[p_], -2.05, p_.strftime("%b\n%Y") if p_.month == 1 or p_ == ticks[0] else p_.strftime("%b"),
                ha="center", va="top", color=MUTED, fontsize=9)

    # One row per one-year stretch, oldest on top.
    ax.text(right, n + 0.2, "Core PCE inflation", fontsize=9.5, color=MUTED, va="center")
    for k, e in enumerate(ends):
        y = n - 1 - k
        s = e - small
        ax.plot([x[s], x[e]], [y, y], color=PICK, lw=2.2, solid_capstyle="butt")
        ax.plot([x[s], x[e]], [y, y], "o", color=PICK, ms=5)
        ax.text(right, y, f"{s.strftime('%b %Y')} → {e.strftime('%b %Y')}", va="center", fontsize=9.5)
        ax.text(right + 6.9, y, f"{rates[e]:.1f}%", va="center", ha="right", fontsize=9.5)
    ax.plot([right, right + 6.9], [-0.45, -0.45], color=INK, lw=0.8)
    ax.text(right, -0.85, f"Average of the {n}", va="center", fontsize=10, fontweight="bold", color=PICK)
    ax.text(right + 6.9, -0.85, f"{rates.mean():.2f}%", va="center", ha="right", fontsize=10,
            fontweight="bold", color=PICK)

    # Start to end, for comparison.
    ax.plot([x[a], x[b]], [-1.55, -1.55], color=CUR, lw=2.2, solid_capstyle="butt")
    ax.plot([x[a], x[b]], [-1.55, -1.55], "o", color=CUR, ms=5)
    ax.text(right, -1.55, "Start to end, for comparison", va="center", fontsize=9.5, color=CUR)
    ax.text(right + 6.9, -1.55, f"{whole:.2f}%", va="center", ha="right", fontsize=9.5, color=CUR)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)
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
    averaging_chart(m, pd.Period(args.month, "M"), os.path.join(args.out, "1_how_the_averaging_works.png"))
    minimum_chart(m, os.path.join(args.out, "2_how_the_12_month_minimum_works.png"))
    print(f"Wrote two charts to {args.out}")
