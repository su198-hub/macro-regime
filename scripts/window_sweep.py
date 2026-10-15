"""Which nowcast window? Scores every version of the regime-at-the-time label.

    python scripts/window_sweep.py              # prints the table, writes a CSV
    python scripts/window_sweep.py --realtime   # adds the real-time check (slow)

Versions, two ways of dating a 12-month change:

  middle  by the middle of what it covers: windows of 12-36 months centred on
          the month, measured start to end or as the average of the 6-, 9- or
          12-month changes inside them.
  ending  by the month it ends in, as a year-over-year rate is quoted: the
          average of the 12-month changes ending in the months balanced around
          the month -- 1, 7, 13, 19 or 25 of them -- or of 6-month changes
          (1, 7 or 13). Seven 12-month changes run from 3 months before the
          month to 3 after and cover the 18 months to 3 months after it.

Each with a minimum regime length of 3, 6, 9, 12 or 18 months, and the band
from truth.band for its window.

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
  provisional         months at the end that wait for later data. Reported,
                      not judged: the label's job is the past.
  realtime            (with --realtime) for every month from Jan 1985 to Dec
                      2023, the label the page would have shown with data to
                      that month, against the final label.

The script prints the frontier: every version no other beats on flips, episode
share, recession months and turn lag at once.

The nowcast in truth.METHODS dates by end month, because that is how people
read inflation (now against a year ago), and averages the 7 year-over-year
rates ending from 3 months before the month to 3 after, holding a regime at
least 12 months: 3.0 flips a decade, 10 of 11 episodes, 79% of episode months,
95% of recession months, 3 months provisional. Dated by the middle month the
same window does better on history (88% of episode months, 98% of recession
months, 6.8 against 8.1 months of turn lag), because a change ending in a month
mostly describes the year before it, which puts end-dated labels about six
months behind events. Among end-dated year-over-year versions, 7 rates with a
12-month minimum is the steadiest that matches 10 of 11 episodes, though two of
them, the late-80s boom and the 2022 squeeze, sit at exactly half their months.
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


def build(m: pd.DataFrame, months: int, measure: str, small: int | None, min_months: int,
          dating: str = "middle") -> pd.DataFrame:
    """The label for one version. `months` is the span of data the window
    covers; `small` the length of each averaged change; `dating` "middle"
    (centred window) or "ending" (changes balanced around the month by end month)."""
    align = "centered" if dating == "middle" else "ending"
    key, saved = (months / 12, align), dict(truth.AVERAGED)
    try:
        truth.AVERAGED.pop(key, None)
        if measure == "averaged":
            truth.AVERAGED[key] = (small, months)
        return truth.build(m, years=months / 12, align=align, measure=measure, min_months=min_months)
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


def score(lab: pd.DataFrame, months: int | None = None) -> dict:
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
            "turn_lag": turn_lag(lab), "provisional": int(lab["provisional"].sum())}


def versions():
    """(months, measure, small, min_months, dating) for every version scored."""
    for min_months in (3, 6, 9, 12, 18):
        for months in (12, 18, 24, 30, 36):
            yield months, "endpoints", None, min_months, "middle"
            for small in (6, 9, 12):
                if small < months:   # one 12-month change in 12 months is start to end
                    yield months, "averaged", small, min_months, "middle"
        for small, spans in ((12, (12, 18, 24, 30, 36)), (6, (6, 12, 18))):
            for months in spans:     # an odd number of changes, so they balance around the month
                yield months, "averaged", small, min_months, "ending"


def sweep(m: pd.DataFrame, with_realtime: bool = False) -> pd.DataFrame:
    rows = []
    for v in versions():
        lab = build(m, *v)
        row = {"dating": v[4], "window": v[0], "measure": v[1], "small": v[2],
               "changes": (v[0] - v[2] + 1) if v[1] == "averaged" else 1, "min_months": v[3], **score(lab)}
        if with_realtime:
            row["realtime"] = realtime(m, lab, *v)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["frontier"] = [not dominated(out, i) for i in out.index]
    return out


def dominated(res: pd.DataFrame, i: int) -> bool:
    """True if another version is at least as good on every check and better on one."""
    r = res.loc[i]
    ge = ((res["flips_per_decade"] <= r["flips_per_decade"]) & (res["episode_share"] >= r["episode_share"])
          & (res["recession_months"] >= r["recession_months"]) & (res["turn_lag"] <= r["turn_lag"]))
    gt = ((res["flips_per_decade"] < r["flips_per_decade"]) | (res["episode_share"] > r["episode_share"])
          | (res["recession_months"] > r["recession_months"]) | (res["turn_lag"] < r["turn_lag"]))
    return bool((ge & gt).any())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realtime", action="store_true", help="Add the real-time check (slow).")
    res = sweep(load_inputs(), ap.parse_args().realtime)
    pd.set_option("display.width", 250)
    print(res.drop(columns="missed").round(2).to_string(index=False))
    front = res[res["frontier"]].sort_values("flips_per_decade")
    print("\nFrontier, steadiest first:")
    print(front.drop(columns="frontier").round(2).to_string(index=False))
    path = os.path.join(os.path.dirname(DB), "window_sweep.csv")
    res.to_csv(path, index=False)
    print(f"Wrote {path}")
