"""Two explainer charts for the steadier nowcast, for emails and slides.

    python scripts/window_visuals.py [--out FOLDER] [--month 2023-02]

1. How the 18-month window turns one month into a label, with that month's
   real numbers (core PCE and GDP, start to end, against expectations and
   potential).
2. How the 6-month minimum regime length absorbs a short blip (Jul-Sep 2020).

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
BOX = dict(facecolor="white", edgecolor="none", pad=2.5)
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "text.color": INK})


def quarterly(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return s.groupby(s.index.asfreq("Q")).mean()


def window_chart(m: pd.DataFrame, t: pd.Period, path: str) -> None:
    steady = truth.VERSIONS["now"]["steadier"]
    ax_ = truth.axes(m, steady["years"], "centered")
    lab = truth.build(m, **steady)
    band = truth.band(steady["years"])
    c = m["core_pce"]
    a, b = t - 9, t + 9
    infl = ((c[b] / c[a]) ** (12 / 18) - 1) * 100
    q = t.asfreq("Q")
    g, p = quarterly(m["gdp"]), quarterly(m["potential"])
    g_rate = ((g[q + 3] / g[q - 3]) ** (4 / 6) - 1) * 100
    p_rate = ((p[q + 3] / p[q - 3]) ** (4 / 6) - 1) * 100
    gap_i, gap_g = ax_.loc[t, "inflation"], ax_.loc[t, "growth"]
    word = lambda v, up, down: up if v >= band else down if v <= -band else "inside the band, so last month's reading"

    months = pd.period_range(t - 13, t + 13, freq="M")
    x = {p_: i for i, p_ in enumerate(months)}
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(-1, len(months)); ax.set_ylim(-0.5, 6.3); ax.axis("off")
    for p_ in months:
        if p_.month in (1, 4, 7, 10):
            ax.text(x[p_], -0.1, p_.strftime("%b\n%Y") if p_.month == 1 else p_.strftime("%b"),
                    ha="center", va="top", color=MUTED, fontsize=9)
            ax.plot([x[p_], x[p_]], [0.05, 5.7], color=LINE, lw=0.6, zorder=0)
    ax.plot([x[t], x[t]], [0.05, 5.75], color=INK, lw=1.6, ls=(0, (3, 2)), zorder=1)
    ax.text(x[t], 5.9, f"Month being labeled: {t.strftime('%b %Y')}", ha="center", fontsize=11.5, fontweight="bold")

    def label(y, text, color):
        ax.text(-1.6, y, text, ha="right", va="center", fontsize=10, color=color, fontweight="bold", clip_on=False)

    def bar(y, lo, hi, color, note):
        ax.add_patch(Rectangle((x[lo] - 0.5, y), x[hi] - x[lo] + 1, 0.55, color=color, alpha=0.18, lw=0))
        ax.add_patch(FancyArrowPatch((x[lo], y + 0.275), (x[hi], y + 0.275), arrowstyle="-|>",
                                     mutation_scale=14, color=color, lw=2))
        ax.text(x[lo] - 0.4, y + 0.72, note, ha="left", va="bottom", fontsize=9.5, bbox=BOX, zorder=3)

    label(4.55, "Today: 12 months\n(6 either side)", CUR)
    bar(4.3, t - 6, t + 6, CUR, f"{(t - 6).strftime('%b %Y')} → {(t + 6).strftime('%b %Y')}")
    label(2.95, "Proposed: 18 months\n(9 either side)", PICK)
    bar(2.7, a, b, PICK, f"Core PCE index: {c[a]:.2f} in {a.strftime('%b %Y')} → {c[b]:.2f} in "
                         f"{b.strftime('%b %Y')}  =  {infl:.2f}% a year")
    label(1.65, "Then subtract\nwhat was expected", PICK)
    ax.text(-0.5, 1.65, f"Expected inflation, averaged over the same 18 months: {infl - gap_i:.2f}%\n"
            f"Gap: {infl:.2f} − {infl - gap_i:.2f} = {gap_i:+.2f} points, against the ±{band:.2f} band  →  "
            f"{word(gap_i, 'inflation high', 'inflation low')}", ha="left", va="center", fontsize=9.5, bbox=BOX, zorder=3)
    label(0.55, "Growth: same\nwindow on GDP", PICK)
    ax.text(-0.5, 0.55, f"Real GDP {(q - 3)} → {(q + 3)}: {g_rate:.2f}% a year, against CBO potential {p_rate:.2f}%\n"
            f"Gap: {gap_g:+.2f} points, against the ±{band:.2f} band  →  {word(gap_g, 'growth up', 'growth down')}",
            ha="left", va="center", fontsize=9.5, bbox=BOX, zorder=3)
    fig.text(0.5, 0.025, f"Label for {t.strftime('%b %Y')}: {NAME[lab.loc[t, 'label']].lower()}.  "
             "A gap inside the band keeps last month's reading.", ha="center", fontsize=10.5,
             fontweight="bold", color=PICK)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.97, bottom=0.1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def minimum_chart(m: pd.DataFrame, path: str) -> None:
    steady = truth.VERSIONS["now"]["steadier"]
    rows = [("Monthly reading\n(no minimum)", 1), ("Today's rule:\n3-month minimum", 3),
            ("Proposed:\n6-month minimum", steady["min_months"])]
    months = pd.period_range("2020-01", "2021-03", freq="M")
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(-0.6, len(months) - 0.4); ax.set_ylim(-0.55, 3.3); ax.axis("off")
    for r, (name, mm) in enumerate(rows):
        lab = truth.build(m, years=steady["years"], align="centered", measure="endpoints", min_months=mm)["label"]
        y = 2 - r
        ax.text(-0.8, y + 0.4, name, ha="right", va="center", fontsize=10, fontweight="bold",
                color=PICK if r == 2 else INK)
        for i, p_ in enumerate(months):
            ax.add_patch(Rectangle((i - 0.45, y + 0.05), 0.9, 0.7, color=COL[lab[p_]], lw=0))
    for i, p_ in enumerate(months):
        ax.text(i, -0.12, p_.strftime("%b\n%Y") if p_.month == 1 else p_.strftime("%b"),
                ha="center", va="top", fontsize=9, color=MUTED)
    blip = [i for i, p_ in enumerate(months) if pd.Period("2020-07", "M") <= p_ <= pd.Period("2020-09", "M")]
    ax.add_patch(Rectangle((blip[0] - 0.55, -0.02), len(blip) + 0.1, 2.9, fill=False, ec=INK, lw=1.4, ls=(0, (3, 2))))
    ax.text(blip[0] + 1, 3.0, "Jul–Sep 2020: 3 months read stagflation", ha="center", fontsize=10, fontweight="bold")
    fig.legend([Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL], [NAME[k] for k in COL], loc="lower center",
               bbox_to_anchor=(0.58, 0.075), ncol=4, frameon=False, fontsize=9.5)
    fig.text(0.5, 0.02, "A regime shorter than the minimum takes the label before it. The 3-month blip passes "
             "today's rule; the 6-month rule absorbs it into hard landing.", ha="center", fontsize=10,
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
    minimum_chart(m, os.path.join(args.out, "2_how_the_6_month_minimum_works.png"))
    print(f"Wrote two charts to {args.out}")
