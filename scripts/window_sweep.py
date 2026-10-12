"""Which nowcast window? Scores 57 versions of the regime-at-the-time label.

    python scripts/window_sweep.py              # prints the table, writes a CSV
    python scripts/window_sweep.py --realtime   # adds the real-time check (~10 min)

Versions: windows of 12, 18, 24, 30 and 36 months centred on the month (GDP is
quarterly, so a centred window grows two quarters at a time); measured start
to end, or as the average of the 6-, 9- or 12-month changes inside the window;
and a minimum regime length of 3, 6 or 9 months.

Checks, for each version:
  flips per decade    regime changes since 1980 per ten years (steadiness)
  median months       median length of a regime
  recession months    share of NBER recession months labeled growth down
  episodes            of the 11 well-known episodes in src/truth.py, how many
                      have most of their months carrying the usual label
  episode share       the share of episode months labeled as expected, a
                      finer version of the same check
  turn lag            months between each recession's start or end and the
                      label switching into or out of growth down, averaged
                      over early and late alike (timeliness)
  provisional         half the window: months that wait for later data
  realtime            (with --realtime) for every month from Jan 1985 to Dec
                      2023, the label the page would have shown with data to
                      that month, against the final label. Today's revised
                      data throughout, and no publication delay, so it isolates
                      the window's own edge, not data revisions.

The pick: among versions matching at least 10 of 11 episodes, catching at
least 95% of recession months and leaving at most 12 months provisional, the
one that flips least, ties going to the shorter turn lag. That is 18 months,
averaging the 6-month changes inside it, 9-month minimum: the steadier nowcast
in truth.VERSIONS.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import truth  # noqa: E402
from src.store import Store  # noqa: E402

DB = os.environ.get("MACRO_REGIME_DB",
                    os.path.join(os.environ.get("LOCALAPPDATA", "data"), "macro-regime", "macrobond.duckdb"))
CUTOFFS = pd.period_range("1985-01", "2023-12", freq="M")


def load_inputs() -> pd.DataFrame:
    store = Store(DB)
    try:
        return truth.from_store(store.latest(list(truth.STORE_IDS.values())))
    finally:
        store.close()


def build(m: pd.DataFrame, months: int, measure: str, small: int | None, min_months: int) -> pd.DataFrame:
    """The centred label for one version, with `small`-month changes when averaged."""
    key, saved = (months / 12, "centered"), dict(truth.AVERAGED)
    try:
        truth.AVERAGED.pop(key, None)
        if measure == "averaged":
            truth.AVERAGED[key] = (small, months)
        return truth.build(m, years=months / 12, align="centered", measure=measure, min_months=min_months)
    finally:
        truth.AVERAGED.clear()
        truth.AVERAGED.update(saved)


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


def realtime(m: pd.DataFrame, full: pd.DataFrame, *version) -> float:
    """Share of months whose label at the time matched the final one. CBO
    potential is kept whole: it is a projection, published years ahead."""
    hits = []
    cols = [c for c in m.columns if c != "potential"]
    for t in CUTOFFS:
        cut = m.copy()
        cut.loc[cut.index > t, cols] = np.nan
        lab = build(cut, *version)
        if t in lab.index and t in full.index:
            hits.append(lab.loc[t, "label"] == full.loc[t, "label"])
    return float(np.mean(hits))


def score(lab: pd.DataFrame, months: int) -> dict:
    run = (lab["label"] != lab["label"].shift()).cumsum()
    lengths = lab.groupby(run).size()
    down = lab["label"].isin(["stagflation", "hard_landing"])
    ep = truth.episode_check(lab)
    return {"flips_per_decade": (len(lengths) - 1) / (len(lab) / 120),
            "median_months": float(lengths.median()),
            "recession_months": float(down[lab["recession"]].mean()),
            "episodes": int((ep["share"] >= 0.5).sum()),
            "episode_share": float(ep["share"].mean()),
            "missed": "; ".join(ep.loc[ep["share"] < 0.5, "episode"]),
            "turn_lag": turn_lag(lab), "provisional": months // 2}


def versions():
    for months in (12, 18, 24, 30, 36):
        for min_months in (3, 6, 9):
            yield months, "endpoints", None, min_months
            for small in (6, 9, 12):
                if small < months:   # one 12-month change in 12 months is start to end
                    yield months, "averaged", small, min_months


def sweep(m: pd.DataFrame, with_realtime: bool = False) -> pd.DataFrame:
    rows = []
    for v in versions():
        lab = build(m, *v)
        row = {"window": v[0], "measure": v[1], "small": v[2], "min_months": v[3], **score(lab, v[0])}
        if with_realtime:
            row["realtime"] = realtime(m, lab, *v)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["passes"] = (out["episodes"] >= 10) & (out["recession_months"] >= 0.95) & (out["provisional"] <= 12)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realtime", action="store_true", help="Add the real-time check (slow).")
    res = sweep(load_inputs(), ap.parse_args().realtime)
    pd.set_option("display.width", 250)
    print(res.drop(columns="missed").round(2).to_string(index=False))
    best = res[res["passes"]].sort_values(["flips_per_decade", "turn_lag"]).iloc[0]
    how = f"averaging {best.small:.0f}-month changes" if best.measure == "averaged" else "start to end"
    print(f"\nSteadiest that passes: {best.window} months, {how}, {best.min_months}-month minimum")
    path = os.path.join(os.path.dirname(DB), "window_sweep.csv")
    res.to_csv(path, index=False)
    print(f"Wrote {path}")
