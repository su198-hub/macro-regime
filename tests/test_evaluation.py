"""The record's measures on calls small enough to check by hand."""

import numpy as np
import pandas as pd

from src import evaluation as ev

G, H, S, L = "goldilocks", "high_growth_high_inflation", "stagflation", "hard_landing"
REGIMES = [G, H, S, L]


def series(values, start="2000-01"):
    return pd.Series(values, index=pd.period_range(start, periods=len(values), freq="M"))


def test_precision_recall_and_f1():
    labels = series([G, G, G, G, L, L])
    calls = series([G, G, L, L, L, L])
    t = ev.by_regime(calls, labels, REGIMES)
    assert t.at[G, "precision"] == 1.0 and t.at[G, "recall"] == 0.5
    assert t.at[L, "precision"] == 0.5 and t.at[L, "recall"] == 1.0
    assert np.isclose(t.at[L, "f1"], 2 * 0.5 * 1.0 / 1.5)
    assert np.isnan(t.at[H, "precision"])          # never called
    assert t.at[G, "base_rate"] == 4 / 6


def test_a_call_outside_the_regimes_counts_against_recall_only():
    labels = series([G, G])
    calls = series([G, "transitional"])
    t = ev.by_regime(calls, labels, REGIMES)
    assert t.at[G, "precision"] == 1.0 and t.at[G, "recall"] == 0.5


def test_axes_pool_the_regimes_on_each_side():
    labels = series([S, L, G, H])
    calls = series([L, L, H, H])
    a = ev.axes(calls, labels)
    assert a.at["growth_below", "precision"] == 1.0 and a.at["growth_below", "recall"] == 1.0
    # Inflation above: called in months 3 and 4, followed in months 1 and 4.
    assert a.at["inflation_above", "precision"] == 0.5
    assert a.at["inflation_above", "recall"] == 0.5


def test_bootstrap_brackets_a_perfect_record_at_one():
    labels = series([G] * 6 + [L] * 6 + [G] * 6)
    b = ev.bootstrap(labels, labels, REGIMES, draws=200)
    assert b["episodes"] == 3
    assert b["balanced"] == (1.0, 1.0)


def test_longest_misses_are_unbroken_runs_of_one_mistake():
    labels = series([G, G, G, G, L, L, L])
    calls = series([L, L, G, H, H, L, G])
    m = ev.longest_misses(calls, labels)
    assert m.iloc[0].to_dict() == {"start": pd.Period("2000-01", "M"), "end": pd.Period("2000-02", "M"),
                                   "called": L, "followed": G, "months": 2}
    assert len(m) == 4


def test_calibration_bands_the_leading_odds():
    labels = series([G, G, L])
    probs = pd.DataFrame({G: [0.9, 0.55, 0.85], L: [0.1, 0.45, 0.15]}, index=labels.index)
    c = ev.calibration(probs, labels)
    top = c.iloc[-1]
    assert top.months == 2 and top.right == 0.5
