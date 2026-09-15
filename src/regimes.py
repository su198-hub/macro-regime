"""Regime classification.

Each regime is a point in driver space. Score by salience-weighted distance,
softmax into probabilities, then require persistence before calling a switch.

No hidden Markov model here on purpose. Start with something you can argue
with in a meeting. Once you have a hand-labelled history to validate against,
statsmodels MarkovRegression or a Gaussian mixture over this same driver space
becomes a reasonable upgrade — and this module's output is what you validate
them against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_regimes(path: str | Path = "config/regimes.yml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _driver_order(reg_cfg: dict) -> list[str]:
    """The drivers a regime call needs, taken from the archetypes.

    Read from config rather than from whichever columns a driver frame happens
    to have: a driver with no inputs at all is missing, not optional.
    """
    order: list[str] = []
    for spec in reg_cfg["regimes"].values():
        order += [d for d in spec["archetype"] if d not in order]
    return order


def _matrix(reg_cfg: dict, driver_order: list[str]):
    names = list(reg_cfg["regimes"].keys())
    arche = np.array([
        [reg_cfg["regimes"][n]["archetype"][d] for d in driver_order]
        for n in names
    ])
    salience = np.array([
        float(reg_cfg["driver_salience"].get(d, 1.0)) for d in driver_order
    ])
    return names, arche, salience


def probabilities(drivers: pd.DataFrame, reg_cfg: dict) -> pd.DataFrame:
    """Monthly probability across regimes.

    Rows where any driver is missing return NaN rather than a guess. A regime
    call built on three of five drivers is not a regime call.
    """
    driver_order = _driver_order(reg_cfg)
    names, arche, salience = _matrix(reg_cfg, driver_order)
    temp = float(reg_cfg["settings"]["temperature"])

    X = drivers.reindex(columns=driver_order).to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1)

    out = np.full((len(X), len(names)), np.nan)
    if valid.any():
        diff = X[valid][:, None, :] - arche[None, :, :]
        dist = np.sqrt(((diff ** 2) * salience).sum(axis=2))
        logits = -dist / temp
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        out[valid] = exp / exp.sum(axis=1, keepdims=True)

    return pd.DataFrame(out, index=drivers.index, columns=names)


UNCLASSIFIED = "unclassified"


def fit(drivers: pd.DataFrame, reg_cfg: dict) -> pd.DataFrame:
    """Whether a month is close enough to its nearest regime to be called it.

    Probabilities are relative: they sum to one whether or not any regime
    describes the month. On their own they turn every middling month into
    whichever archetype sits nearest the centre, which is how 2011-14 came out
    as goldilocks. The fit gate asks an absolute question instead.

    The yardstick is each archetype's own distance from a perfectly neutral
    economy (every driver at zero), so a regime far from neutral, like a hard
    landing, accepts months far from neutral, and one near it, like goldilocks,
    demands a closer match. Three grades:

      clear  closer to the archetype than neutral is
      weak   further than neutral, but within `fit_tolerance` times that distance
      none   beyond the tolerance: no regime is called

    With a tolerance of 1 there is no weak grade. That was too strict in
    practice: more than half of all months since 1990 fitted nothing, most of
    them only just, and a monitor that says "no clear regime" half the time
    says little. Weak fits are still called, but flagged.
    """
    driver_order = _driver_order(reg_cfg)
    names, arche, salience = _matrix(reg_cfg, driver_order)
    tolerance = float(reg_cfg["settings"].get("fit_tolerance", 1.0))
    X = drivers.reindex(columns=driver_order).to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1)
    from_neutral = np.sqrt(((arche ** 2) * salience).sum(axis=1))

    nearest = np.full(len(X), None, dtype=object)
    distance = np.full(len(X), np.nan)
    threshold = np.full(len(X), np.nan)
    if valid.any():
        diff = X[valid][:, None, :] - arche[None, :, :]
        dist = np.sqrt(((diff ** 2) * salience).sum(axis=2))
        k = dist.argmin(axis=1)
        nearest[valid] = np.array(names, dtype=object)[k]
        distance[valid] = dist[np.arange(len(k)), k]
        threshold[valid] = from_neutral[k]
    out = pd.DataFrame({"nearest": nearest, "distance": distance, "threshold": threshold},
                       index=drivers.index)
    out["limit"] = out["threshold"] * tolerance
    out["fits"] = out["distance"] < out["limit"]
    grade = np.where(out["distance"] < out["threshold"], "clear",
                     np.where(out["fits"], "weak", "none"))
    out["fit"] = pd.Series(grade, index=out.index, dtype=object).where(valid)
    return out


def call_regime(probs: pd.DataFrame, reg_cfg: dict, fits: pd.Series | None = None) -> pd.DataFrame:
    """Apply the fit gate, persistence and the confidence floor.

    The leading regime only becomes the called regime after it has led for
    `persistence_months` consecutive months. Without this the dashboard flips
    on noise and people stop opening it.

    With the fit gate on, a month whose leader does not fit counts as
    "unclassified" and goes through the same persistence rule, so the call
    moves to and from "no clear regime" as deliberately as between regimes.
    """
    persistence = int(reg_cfg["settings"]["persistence_months"])
    floor = float(reg_cfg["settings"]["min_confidence"])

    # idxmax raises on all-NaN rows under pandas 3, so take the leader over
    # scored months only and leave unscored months as NaN for the loop below.
    leader = probs.dropna(how="all").idxmax(axis=1).reindex(probs.index)
    top = probs.max(axis=1)
    state = leader.copy()
    if fits is not None:
        misfit = leader.notna() & ~fits.reindex(probs.index).fillna(False).astype(bool)
        state = state.astype(object).where(~misfit, UNCLASSIFIED)

    called: list[str | None] = []
    current: str | None = None   # the regime we are currently calling
    previous: str | None = None  # last month's leader, for the run counter
    run = 0

    for name in state:
        if pd.isna(name):
            called.append(current)
            previous, run = None, 0
            continue
        # Count consecutive months this name has led, not months since the
        # last call. Comparing against `current` here means a challenger's
        # streak resets every month and it can never accumulate enough to win.
        run = run + 1 if name == previous else 1
        previous = name
        if current is None:
            current = name                              # seed
        elif name != current and run >= persistence:
            current = name                              # confirmed switch
        called.append(current)

    result = pd.DataFrame({
        "leading": leader,
        "leading_probability": top,
        "state": state,
        "called": called,
    }, index=probs.index)
    low = (result["leading_probability"] < floor) & (result["called"] != UNCLASSIFIED)
    result.loc[low, "called"] = "transitional"
    return result


def contributions(drivers: pd.DataFrame, reg_cfg: dict, as_of) -> pd.DataFrame:
    """Per-driver distance to each archetype for one month.

    Answers the only question anyone asks when they disagree with the call:
    which driver is pulling it there.
    """
    driver_order = _driver_order(reg_cfg)
    names, arche, salience = _matrix(reg_cfg, driver_order)
    x = drivers.reindex(columns=driver_order).loc[as_of].to_numpy(dtype=float)
    gap = ((x[None, :] - arche) ** 2) * salience
    return pd.DataFrame(gap, index=names, columns=driver_order).round(3)


def run(drivers: pd.DataFrame, reg_cfg: dict) -> dict:
    probs = probabilities(drivers, reg_cfg)
    gate = str(reg_cfg["settings"].get("fit_gate", "none"))
    fitted = fit(drivers, reg_cfg)
    gate_on = gate == "closer_than_neutral"
    calls = call_regime(probs, reg_cfg, fitted["fits"] if gate_on else None)
    if gate_on:
        calls = calls.join(fitted[["distance", "threshold", "limit", "fit", "fits"]])
    return {"probabilities": probs, "calls": calls, "fit": fitted}
