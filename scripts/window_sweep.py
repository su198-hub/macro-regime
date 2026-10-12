"""Which nowcast window? Scores 27 versions of the regime-at-the-time label.

    python scripts/window_sweep.py            # prints the table, writes a CSV

Versions: windows of 12, 18, 24, 30 and 36 months centred on the month (GDP is
quarterly, so a centred window grows two quarters at a time); measured start
to end, or as the average of the 12-month changes inside the window; and a
minimum regime length of 3, 6 or 9 months.

Checks, for each version:
  flips per decade    regime changes since 1980 per ten years (steadiness)
  median months       median length of a regime
  recession months    share of NBER recession months labeled growth down
  episodes            of the 11 well-known episodes in src/truth.py, how many
                      have most of their months carrying the usual label
  turn lag            months between each recession's start or end and the
                      label switching into or out of growth down, averaged
                      over early and late alike (timeliness)
  provisional         half the window: months that wait for later data

The pick: among versions matching at least 10 of 11 episodes, catching at
least 95% of recession months and leaving at most 12 months provisional, the
one that flips least. That is 18 months, start to end, 6-month minimum, the
steadier nowcast in truth.VERSIONS.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import truth  # noqa: E402
from src.store import Store  # noqa: E402

DB = os.environ.get("MACRO_REGIME_DB",
                    os.path.join(os.environ.get("LOCALAPPDATA", "data"), "macro-regime", "macrobond.duckdb"))


def load_inputs() -> pd.DataFrame:
    store = Store(DB)
    try:
        return truth.from_store(store.latest(list(truth.STORE_IDS.values())))
    finally:
        store.close()


def turn_lag(lab: pd.DataFrame) -> float:
    down = lab["label"].isin(["stagflation", "hard_landing"])
    gaps = []
    for pk, tr in truth.NBER:
        pk, tr = pd.Period(pk, "M"), pd.Period(tr, "M")
        if pk < pd.Period("1980-06", "M"):
            continue
        w = down.loc[pk - 36:tr]
        switches = w[w & ~w.shift(1, fill_value=True)].index
        if len(switches):
            gaps.append(abs((switches[-1] - pk).n))
        after = down.loc[tr:]
        up = after[~after].index
        if len(up):
            gaps.append(abs((up[0] - tr).n))
    return float(np.mean(gaps))


def score(lab: pd.DataFrame, months: int) -> dict:
    run = (lab["label"] != lab["label"].shift()).cumsum()
    lengths = lab.groupby(run).size()
    down = lab["label"].isin(["stagflation", "hard_landing"])
    ep = truth.episode_check(lab)
    return {"flips_per_decade": (len(lengths) - 1) / (len(lab) / 120),
            "median_months": float(lengths.median()),
            "recession_months": float(down[lab["recession"]].mean()),
            "episodes": int((ep["share"] >= 0.5).sum()),
            "turn_lag": turn_lag(lab), "provisional": months // 2}


def sweep(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for months in (12, 18, 24, 30, 36):
        for measure in ("endpoints", "averaged"):
            if months == 12 and measure == "averaged":
                continue   # one 12-month change: identical to start to end
            for min_months in (3, 6, 9):
                lab = truth.build(m, years=months / 12, align="centered", measure=measure, min_months=min_months)
                rows.append({"window": months, "measure": measure, "min_months": min_months,
                             **score(lab, months)})
    out = pd.DataFrame(rows)
    out["passes"] = (out["episodes"] >= 10) & (out["recession_months"] >= 0.95) & (out["provisional"] <= 12)
    return out


if __name__ == "__main__":
    res = sweep(load_inputs())
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    best = res[res["passes"]].sort_values(["flips_per_decade", "turn_lag"]).iloc[0]
    print(f"\nSteadiest that passes: {best.window} months, {best.measure}, {best.min_months}-month minimum")
    path = os.path.join(os.path.dirname(DB), "window_sweep.csv")
    res.to_csv(path, index=False)
    print(f"Wrote {path}")
