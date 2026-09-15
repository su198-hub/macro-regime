"""The drill-down must explain the driver score, not a different number."""

import numpy as np
import pandas as pd
import pytest

from src.drivers import driver_breakdown, driver_scores, indicator_frames

CFG = {"drivers": {"demand": {"indicators": [
    {"id": "a", "source": {"fred": "A"}, "transform": "level",
     "normalize": {"method": "gap", "center": 0.0, "scale": 1.0},
     "direction": 1, "weight": 0.6},
    {"id": "b", "source": {"fred": "B"}, "transform": "level",
     "normalize": {"method": "gap", "center": 1.0, "scale": 0.5},
     "direction": -1, "weight": 0.4},
]}}}


def wide(b_values):
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    return pd.DataFrame({"A": [0.5, 1.0, 1.5], "B": b_values}, index=idx)


def test_contributions_sum_to_the_driver_score():
    w = wide([1.2, 0.9, 1.4])
    inputs, scores = indicator_frames(CFG, w)
    month = w.index[-1]
    parts = driver_breakdown(CFG, "demand", inputs, scores, month)
    assert parts["contribution"].sum() == pytest.approx(
        driver_scores(CFG, w).at[month, "demand"])
    assert parts.set_index("id").at["b", "value"] == pytest.approx(1.4)


def test_missing_indicator_is_excluded_and_weights_renormalise():
    w = wide([1.2, 0.9, np.nan])
    inputs, scores = indicator_frames(CFG, w)
    month = w.index[-1]
    parts = driver_breakdown(CFG, "demand", inputs, scores, month).set_index("id")
    assert np.isnan(parts.at["b", "share"])
    assert parts.at["a", "share"] == pytest.approx(1.0)
    assert parts["contribution"].sum() == pytest.approx(
        driver_scores(CFG, w).at[month, "demand"])
