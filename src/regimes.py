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


def call_regime(probs: pd.DataFrame, reg_cfg: dict) -> pd.DataFrame:
    """Apply persistence and the confidence floor.

    The leading regime only becomes the called regime after it has led for
    `persistence_months` consecutive months. Without this the dashboard flips
    on noise and people stop opening it.
    """
    persistence = int(reg_cfg["settings"]["persistence_months"])
    floor = float(reg_cfg["settings"]["min_confidence"])

    # idxmax raises on all-NaN rows under pandas 3, so take the leader over
    # scored months only and leave unscored months as NaN for the loop below.
    leader = probs.dropna(how="all").idxmax(axis=1).reindex(probs.index)
    top = probs.max(axis=1)

    called: list[str | None] = []
    current: str | None = None   # the regime we are currently calling
    previous: str | None = None  # last month's leader, for the run counter
    run = 0

    for name in leader:
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
        "called": called,
    }, index=probs.index)
    result.loc[result["leading_probability"] < floor, "called"] = "transitional"
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
    return {"probabilities": probs, "calls": call_regime(probs, reg_cfg)}
