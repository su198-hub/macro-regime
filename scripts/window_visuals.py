"""Two explainer charts for the steadier nowcast, for emails.

    python scripts/window_visuals.py [--out FOLDER] [--month 2023-02]

1. How the averaging works: year-over-year core PCE inflation, each rate dated
   by the month it ends in, and the average of the seven from 3 months before
   the month to 3 after.
2. How the 12-month minimum regime length absorbs a short dip (the 1995
   slowdown, which the end-dated labels place from Aug 1995 to Apr 1996).

Writes PNGs to FOLDER (default: output/ in the repo, which git ignores).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

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


def averaging_chart(m: pd.DataFrame, t: pd.Period, path: str) -> None:
    """How one month's inflation reading is built: year-over-year core PCE
    inflation, each rate dated by the month it ends in, and the average of the
    seven ending from 3 months before the month to 3 months after."""
    steady = truth.METHODS["now"]
    n, small = truth.stretches(steady["years"], steady["align"])
    after = truth.months_after(steady["years"], steady["align"], steady["measure"])
    c = m["core_pce"].dropna()
    yoy = ((c / c.shift(small)) ** (12 / small) - 1) * 100
    ends = pd.period_range(t - after, t + after, freq="M")
    rates = yoy.loc[ends]
    avg = float(rates.mean())
    first = ends[0] - small                      # the earliest month the readings reach back to
    shown = yoy.loc[t - 18:t + 12].dropna()

    fig = plt.figure(figsize=(11, 4.9))
    ax = fig.add_axes([0.07, 0.17, 0.58, 0.68])
    xs = shown.index.to_timestamp()
    ax.plot(xs, shown.values, color=CUR, lw=1.8, zorder=2)
    ax.axvspan(ends[0].to_timestamp() - pd.Timedelta(days=15), ends[-1].to_timestamp() + pd.Timedelta(days=15),
               color=PICK, alpha=0.10, lw=0)
    ax.scatter(ends.to_timestamp(), rates.values, color=PICK, s=34, zorder=3)
    ax.hlines(avg, ends[0].to_timestamp(), ends[-1].to_timestamp(), color=PICK, lw=2, zorder=3)
    ax.axvline(t.to_timestamp(), color=INK, lw=1.2, ls=(0, (3, 2)), zorder=1)
    ax.text(t.to_timestamp(), ax.get_ylim()[1], f" {t.strftime('%b %Y')}", va="top", fontsize=9.5, color=INK)
    ax.text(ends[-1].to_timestamp() + pd.Timedelta(days=20), avg, f"average {avg:.2f}%", va="center",
            fontsize=9.5, color=PICK, fontweight="bold")
    ax.set_ylabel("Core PCE inflation, year over year (%)", color=MUTED, fontsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", color=LINE, lw=0.6)
    fig.text(0.07, 0.93, f"The inflation reading for {t.strftime('%b %Y')}: the {n} year-over-year rates from "
             f"{ends[0].strftime('%b %Y')} to {ends[-1].strftime('%b %Y')}, {after} months either side",
             fontsize=10.5, fontweight="bold")

    # The seven rates, listed.
    x0, y0, step = 0.71, 0.80, 0.07
    fig.text(x0, y0 + 0.05, "Year-over-year rate", fontsize=9.5, color=MUTED)
    for k, e in enumerate(ends):
        y = y0 - k * step
        fig.text(x0, y, f"12 months to {e.strftime('%b %Y')}" + ("  (month labeled)" if e == t else ""),
                 fontsize=9.5, fontweight="bold" if e == t else "normal")
        fig.text(0.985, y, f"{rates[e]:.1f}%", fontsize=9.5, ha="right")
    yb = y0 - n * step + 0.015
    fig.add_artist(plt.Line2D([x0, 0.985], [yb + 0.035, yb + 0.035], color=INK, lw=0.8))
    fig.text(x0, yb - 0.01, f"Average of the {n}", fontsize=10, fontweight="bold", color=PICK)
    fig.text(0.985, yb - 0.01, f"{avg:.2f}%", fontsize=10, fontweight="bold", color=PICK, ha="right")
    fig.text(0.07, 0.035, f"Each dot is inflation over the 12 months to that month. Together the {n} rates cover "
             f"the {small + 2 * after} months from {(first + 1).strftime('%b %Y')} to {ends[-1].strftime('%b %Y')}.",
             fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def minimum_chart(m: pd.DataFrame, path: str) -> None:
    steady = truth.METHODS["now"]
    rows = [("3-month minimum", 3), ("9-month minimum", 9),
            (f"{steady['min_months']}-month minimum\n(used)", steady["min_months"])]
    months = pd.period_range("1995-01", "1996-12", freq="M")
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
    dip = [i for i, p_ in enumerate(months) if pd.Period("1995-08", "M") <= p_ <= pd.Period("1996-04", "M")]
    ax.add_patch(Rectangle((dip[0] - 0.55, -0.02), len(dip) + 0.1, 2.9, fill=False, ec=INK, lw=1.4, ls=(0, (3, 2))))
    ax.text((dip[0] + dip[-1]) / 2, 3.0, "Aug 1995 – Apr 1996: the 1995 slowdown, 9 months below potential, no recession",
            ha="center", fontsize=10, fontweight="bold")
    fig.legend([Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL], [NAME[k] for k in COL], loc="lower center",
               bbox_to_anchor=(0.58, 0.075), ncol=4, frameon=False, fontsize=9.5)
    fig.text(0.5, 0.02, "A regime shorter than the minimum takes the label before it. The 9-month dip passes a "
             "3- or 9-month rule; the 12-month rule keeps it goldilocks.", ha="center", fontsize=10,
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
