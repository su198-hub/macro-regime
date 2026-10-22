"""Learn v2's scenario positions from the data, and grade the regime call.

    python scripts/realign_positions.py            # print positions and grades
    python scripts/realign_positions.py --write    # also write them to the config

v2's regime call describes the next three years. Each scenario's position is
where the signposts sat in the run-up to that regime, shrunk toward the prior
in config/regimes_v2.yml (src/outlook.py, realigned_positions). Learned from
1968 to the last month with three full years after it.

Graded through the dashboard's own engine, salience, temperature, persistence,
confidence floor and fit gate as configured, against the three-year forward
label (src/truth.py), January 1993 onward:
  held out    each episode called from positions learned without it or any
              month sharing its outcome window
  real time   each month called from positions learned only from labels
              known by then, 36 months after the month they describe, plus about four for the last GDP quarter
Baseline, unclassified and transitional calls count as misses: the answer key
has only the four growth-inflation regimes. Benchmarks: the latest confirmed
regime continues, and the mix of regimes known so far.
"""

from __future__ import annotations

import copy
import datetime as dt
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.window_sweep import DB, load_inputs  # noqa: E402
from src import drivers as D  # noqa: E402
from src import outlook as ol  # noqa: E402
from src import regimes as R  # noqa: E402
from src import truth  # noqa: E402
from src.store import Store  # noqa: E402

INDICATORS, REGIMES_CFG = "config/indicators_v2.yml", "config/regimes_v2.yml"
FIRST = os.environ.get("REALIGN_FIRST", "1976-01")


def priors(reg: dict, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({r: {n: float(spec["prior"][n]) for n in names}
                         for r, spec in reg["regimes"].items()}).T


def with_positions(reg: dict, pos: pd.DataFrame) -> dict:
    out = copy.deepcopy(reg)
    for r in out["regimes"]:
        out["regimes"][r]["archetype"] = {n: float(pos.at[r, n]) for n in pos.columns}
    return out


def held_out(train, prior, reg, names):
    """Probabilities and fits, each episode from positions learned without it."""
    probs, fits = [], []
    for sp in ol.spells(train["label"]).itertuples():
        test = train.loc[sp.start:sp.end, names]
        rest = train[~ol.purged(train.index, sp.start, sp.end)]
        c = with_positions(reg, ol.realigned_positions(rest, prior, names))
        probs.append(R.probabilities(test, c))
        fits.append(R.fit(test, c)["fits"])
    return pd.concat(probs).sort_index(), pd.concat(fits).sort_index()


def real_time(train, prior, reg, names):
    probs, fits = {}, {}
    for t in train.index[train.index >= pd.Period(FIRST, "M")]:
        known = train[train.index <= t - ol.HORIZON - ol.LABEL_LAG]
        c = with_positions(reg, ol.realigned_positions(known, prior, names))
        x = train.loc[[t], names]
        probs[t] = R.probabilities(x, c).iloc[0]
        fits[t] = bool(R.fit(x, c)["fits"].iloc[0])
    return pd.DataFrame(probs).T, pd.Series(fits)


def grade(probs, fits, reg, train, window, labels) -> dict:
    gate = reg["settings"].get("fit_gate") == "closer_than_neutral"
    calls = R.call_regime(probs, reg, fits if gate else None)["called"].reindex(window)
    out = ol.grade(calls, labels, train.loc[window], probs.reindex(window))
    out["leading"] = ol.score(probs.reindex(window).idxmax(axis=1), labels)["balanced"]
    for state in ("baseline", R.UNCLASSIFIED, "transitional"):
        out[state] = float((calls == state).mean())
    return out


def write(pos: pd.DataFrame, path: str = REGIMES_CFG) -> None:
    """Replace each regime's archetype values in place, keeping the comments."""
    lines = open(path, encoding="utf-8").read().split("\n")
    regime, inside = None, False
    for i, line in enumerate(lines):
        m = re.match(r"^  (\w+):\s*$", line)
        if m:
            regime, inside = m.group(1), False
        elif line.strip() == "archetype:":
            inside = regime in pos.index
        elif inside:
            m = re.match(r"^(\s{6})(\w+): (-?[\d.]+)\s*$", line)
            if m and m.group(2) in pos.columns:
                lines[i] = f"{m.group(1)}{m.group(2)}: {pos.at[regime, m.group(2)]:.2f}"
            else:
                inside = False
    text = "\n".join(lines)
    # Check before replacing the file: every position must read back as written.
    back = yaml.safe_load(text)["regimes"]
    for r in pos.index:
        for n in pos.columns:
            if abs(float(back[r]["archetype"][n]) - round(float(pos.at[r, n]), 2)) > 1e-9:
                raise SystemExit(f"{path}: {r}.{n} did not update; nothing written.")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def main() -> None:
    cfg, reg = D.load_config(INDICATORS), R.load_regimes(REGIMES_CFG)
    names = list(cfg["drivers"])
    store = Store(DB)
    try:
        drivers = D.compute(store, cfg, dt.date.today())["drivers"]
    finally:
        store.close()
    scores = ol.monthly_drivers(drivers, names)
    m = load_inputs()
    fwd = truth.build(m, **truth.METHODS["next3"])
    now = truth.build(m, **truth.METHODS["now"])["label"]
    train = ol.training_frame(scores, fwd)
    window = train.index[train.index >= pd.Period(FIRST, "M")]
    labels = fwd["label"].reindex(window)
    prior = priors(reg, names)

    pd.set_option("display.width", 200)
    print(f"Learned from {train.index[0]} to {train.index[-1]}; episodes:")
    print(ol.support(train).to_string())
    pos = ol.realigned_positions(train, prior, names)
    print("\nRun-up positions, learned only:")
    print(ol.positions(train, names).round(2).to_string())
    print(f"\nRealigned positions (prior strength {ol.PRIOR_STRENGTH} episodes):")
    print(pos.round(2).to_string())

    rows = {}
    for name, (probs, fits) in (("held out", held_out(train, prior, reg, names)),
                                ("real time", real_time(train, prior, reg, names))):
        g = grade(probs, fits, reg, train, window, labels)
        rows[name] = {"regimes right, average": g["balanced"], "months right": g["months"],
                      **{f"  {r}": g["by_regime"].get(r, np.nan) for r in truth.QUADRANT.values()},
                      "hard-landing run-ups caught": f"{g['hard_landing_caught']}/{g['hard_landing_runups']}",
                      "leading scenario, regimes right": g["leading"], "log loss": g["log_loss"],
                      "called baseline": g["baseline"], "transitional": g["transitional"],
                      "unclassified": g[R.UNCLASSIFIED]}
    pers = ol.grade(ol.persistence(now, window), labels, train.loc[window])
    base = ol.base_rate(train, list(truth.QUADRANT.values()), first=FIRST).reindex(window)
    based = ol.grade(base.idxmax(axis=1), labels, train.loc[window], base)
    for name, g in (("regime continues", pers), ("base rate", based)):
        rows[name] = {"regimes right, average": g["balanced"], "months right": g["months"],
                      **{f"  {r}": g["by_regime"].get(r, np.nan) for r in truth.QUADRANT.values()},
                      "hard-landing run-ups caught": f"{g['hard_landing_caught']}/{g['hard_landing_runups']}",
                      "log loss": g["log_loss"]}
    print(f"\nThe regime call against the next three years, {window[0]} to {window[-1]}")
    print(pd.DataFrame(rows).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nEpisodes graded:")
    print(ol.spells(labels.dropna()).to_string(index=False))

    if "--write" in sys.argv:
        write(pos)
        print(f"\nWritten to {REGIMES_CFG}.")


if __name__ == "__main__":
    main()
