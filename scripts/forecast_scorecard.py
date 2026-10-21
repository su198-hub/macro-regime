"""The forecast scorecard: how well each approach calls the next three years.

    python scripts/forecast_scorecard.py      # prints the tables, writes a CSV

The answer key is the three-year forward label (src/truth.py, METHODS["next3"]):
for each month, the regime the following 36 months turned out to be. Every
method is graded on the same months, January 1990 to the last month with a full
three years after it, so the rows compare directly.

Methods
  persistence          the latest confirmed regime continues (no signposts)
  base rate            the mix of regimes known so far (no signposts)
  v1 / v2 positions    each framework's scenario positions from its config,
                       leading scenario only (no persistence or floor); v2's
                       are learned from these months, so in-sample
  run-up, fitted       positions learned from every month whose next three
                       years were each regime, graded on those same months:
                       the upper bound, and the overfitting view
  run-up, held out     positions learned without the episode being graded
                       or anything sharing its outcome window
  run-up, real time    positions learned only from labels known at the time

Reported for each: share of months right, the average of each regime's share
(chance is 25%; always calling one regime scores 25% too), each regime's
share, hard-landing run-ups caught (at least half their months called hard
landing), and log loss of the probabilities (an even split is 1.39; lower is
better; Baseline gets no credit because the answer key has no Baseline).

Overfitting is tolerated where the method is sound, so the fitted row is shown,
but the held-out and real-time rows are what a reader should weigh.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.window_sweep import DB, load_inputs  # noqa: E402
from src import drivers as D  # noqa: E402
from src import outlook as ol  # noqa: E402
from src import regimes as R  # noqa: E402
from src import truth  # noqa: E402
from src.store import Store  # noqa: E402

FRAMEWORKS = {"v1": ("config/indicators.yml", "config/regimes.yml"),
              "v2": ("config/indicators_v2.yml", "config/regimes_v2.yml")}
FIRST = "1990-01"
REGIMES = list(truth.QUADRANT.values())


def load() -> dict:
    store = Store(DB)
    try:
        out = {}
        for fw, (ind, reg) in FRAMEWORKS.items():
            cfg, rcfg = D.load_config(ind), R.load_regimes(reg)
            drivers = D.compute(store, cfg, dt.date.today())["drivers"]
            names = list(cfg["drivers"])
            out[fw] = {"cfg": cfg, "reg": rcfg, "names": names,
                       "scores": ol.monthly_drivers(drivers, names)}
        return out
    finally:
        store.close()


def scorecard() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fws = load()
    m = load_inputs()
    fwd = truth.build(m, **truth.METHODS["next3"])
    now = truth.build(m, **truth.METHODS["now"])["label"]

    trains = {fw: ol.training_frame(f["scores"], fwd) for fw, f in fws.items()}
    common = trains["v1"].index.intersection(trains["v2"].index)
    window = common[common >= pd.Period(FIRST, "M")]
    labels = fwd["label"].reindex(window)

    rows, calls_for_episodes, notes = [], {}, {}

    def add(name, probs=None, calls=None, train=None):
        if probs is not None:
            probs = probs.reindex(window).dropna(how="all")
            calls = probs.idxmax(axis=1)
        calls = calls.reindex(window).dropna()
        g = ol.grade(calls, labels, (train if train is not None else trains["v2"]).loc[window], probs)
        rows.append({"method": name, **{k: v for k, v in g.items() if k != "by_regime"},
                     **{f"{r}": g["by_regime"].get(r, np.nan) for r in REGIMES},
                     "share_baseline": float((calls == "baseline").mean())})
        calls_for_episodes[name] = calls

    add("persistence", calls=ol.persistence(now, window))
    add("base rate", probs=ol.base_rate(trains["v2"], REGIMES, first=FIRST))
    for fw, f in fws.items():
        tr = trains[fw]
        # v2's positions are learned from these same months, so its config row
        # is in-sample; v1's are judgments.
        learned = any("prior" in s for s in f["reg"]["regimes"].values())
        add(f"{fw} positions (config{', in-sample' if learned else ''})",
            probs=R.probabilities(f["scores"].loc[window], f["reg"]), train=tr)
        # The same signpost weights the dashboard's regime call uses.
        wts = f["reg"].get("driver_salience") or {}
        # The fallback for a regime with no episode left must not be fitted on
        # the episode being graded, so v2 falls back to its prior.
        key = "prior" if learned else "archetype"
        judgment = pd.DataFrame({r: f["reg"]["regimes"][r][key] for r in REGIMES}).T[f["names"]]
        fitted = ol.positions(tr, f["names"]).reindex(REGIMES).fillna(judgment)
        add(f"{fw} run-up, fitted", probs=ol.probabilities(tr[f["names"]], fitted, weights=wts), train=tr)
        add(f"{fw} run-up, held out", probs=ol.held_out(tr, judgment, f["names"], weights=wts), train=tr)
        add(f"{fw} run-up, real time", probs=ol.real_time(tr, judgment, f["names"], first=FIRST, weights=wts),
            train=tr)
        notes[fw] = {"fitted_positions": fitted.round(2), "support": ol.support(tr)}

    table = pd.DataFrame(rows).set_index("method")
    episodes = ol.by_episode(trains["v2"].loc[window], calls_for_episodes)
    return table, episodes, notes


if __name__ == "__main__":
    table, episodes, notes = scorecard()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    show = table[["first", "last", "n", "months", "balanced", "goldilocks", "high_growth_high_inflation",
                  "stagflation", "hard_landing", "hard_landing_caught", "hard_landing_runups",
                  "log_loss", "share_baseline"]].copy()
    for c in ("months", "balanced", "goldilocks", "high_growth_high_inflation", "stagflation",
              "hard_landing", "share_baseline"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    show["log_loss"] = show["log_loss"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    print("Forecast scorecard: calling the next three years (answer key: the 3-year forward label)\n")
    print(show.to_string())
    print("\nBy episode, share of months called right:")
    ep = episodes.copy()
    for c in ep.columns[4:]:
        ep[c] = ep[c].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    print(ep.to_string(index=False))
    for fw, n in notes.items():
        print(f"\n{fw} run-up positions, fitted on every month:\n{n['fitted_positions'].to_string()}")
        print(f"support (months, episodes):\n{n['support'].to_string()}")
    path = os.path.join(os.path.dirname(DB), "forecast_scorecard.csv")
    table.to_csv(path)
    print(f"\nWrote {path}")
