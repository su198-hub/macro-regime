"""The drill-down must explain the driver score, not a different number."""

import numpy as np
import pandas as pd
import pytest

from src.drivers import driver_breakdown, driver_scores, indicator_frames

CFG = {"drivers": {"demand": {"indicators": [
    {"id": "a", "source": {"series": "A"}, "transform": "level",
     "normalize": {"method": "gap", "center": 0.0, "scale": 1.0},
     "direction": 1, "weight": 0.6},
    {"id": "b", "source": {"series": "B"}, "transform": "level",
     "normalize": {"method": "gap", "center": 1.0, "scale": 0.5},
     "direction": -1, "weight": 0.4},
]}}}


def wide(b_values):
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    return pd.DataFrame({"A": [0.5, 1.0, 1.5], "B": b_values}, index=idx)


def test_taylor_style_derived_inputs_fall_back_and_floor():
    """first_of fills gaps from a second estimate; floor stops a rule going below the lower bound."""
    cfg = {"drivers": {"monetary": {"indicators": [{
        "id": "taylor_gap", "transform": "level", "direction": 1, "weight": 1.0,
        "normalize": {"method": "gap", "center": 0.0, "scale": 1.0},
        "source": {
            "expr": "FF - rule",
            "series": ["FF", "PI", "SEP", "HLW"],
            "derived": {
                "sep_real": {"expr": "SEP - 2"},
                "neutral": {"first_of": ["sep_real", "HLW"]},
                "rule": {"expr": "neutral + PI + 0.5 * (PI - 2)", "floor": 0.125},
            },
        },
    }]}}}
    idx = pd.date_range("2024-01-31", periods=3, freq="ME")
    w = pd.DataFrame({"FF": [5.0, 5.0, 0.1], "PI": [3.0, 3.0, -2.0],
                      "SEP": [np.nan, 3.0, 3.0], "HLW": [0.5, 0.5, 0.5]}, index=idx)
    inputs, _ = indicator_frames(cfg, w)
    gap = inputs["monetary::taylor_gap"]
    # Month 1: no SEP, neutral from HLW 0.5: rule = 0.5 + 3 + 0.5 = 4.0, gap 1.0
    assert gap.iloc[0] == pytest.approx(1.0)
    # Month 2: SEP-implied neutral 1.0: rule = 1 + 3 + 0.5 = 4.5, gap 0.5
    assert gap.iloc[1] == pytest.approx(0.5)
    # Month 3: rule = 1 - 2 - 2 = -3, floored at 0.125: gap 0.1 - 0.125
    assert gap.iloc[2] == pytest.approx(-0.025)


def test_quarterly_year_on_year_survives_the_ragged_edge():
    """A year-on-year change is taken before the carry, not after.

    Carried first, a twelve-month change reads the latest quarter against
    whichever quarter the month a year ago happened to hold. In the months
    after a release that is three quarters back, not four, so steady 3.1% wage
    growth printed as 2.3%.
    """
    cfg = {"drivers": {"demand": {"indicators": [
        {"id": "q", "source": {"series": "Q"}, "transform": "yoy_pct",
         "normalize": {"method": "gap", "center": 0.0, "scale": 1.0},
         "direction": 1, "weight": 1.0, "carry_forward_periods": 2},
        # A monthly series that reaches further than the quarterly one, so the
        # quarterly value has to be carried past its own last observation.
        {"id": "m", "source": {"series": "M"}, "transform": "level",
         "normalize": {"method": "gap", "center": 0.0, "scale": 1.0},
         "direction": 1, "weight": 1.0},
    ]}}}
    # Quarterly index growing 1% a quarter, last observation Q2 2025. Monthly
    # data runs to July, past the month Q3 would start in, which is exactly
    # when July a year earlier already held Q3 2024 and the comparison slips a
    # quarter.
    q_dates = pd.date_range("2023-01-01", "2025-04-01", freq="QS")
    quarterly = pd.Series([100 * 1.01 ** i for i in range(len(q_dates))], index=q_dates)
    m_dates = pd.date_range("2023-01-31", "2025-07-31", freq="ME")
    w = pd.DataFrame({"Q": quarterly.reindex(m_dates.union(q_dates)),
                      "M": pd.Series(1.0, index=m_dates).reindex(m_dates.union(q_dates))})
    inputs, _ = indicator_frames(cfg, w)
    yoy = inputs["demand::q"].dropna()
    four_quarters = (1.01 ** 4 - 1) * 100
    assert yoy.iloc[-1] == pytest.approx(four_quarters, abs=0.01), \
        "the carried months lost a quarter of growth"
    # Every month of the carry reports the same four-quarter change.
    assert yoy.tail(4).round(6).nunique() == 1


def test_contributions_sum_to_the_driver_score():
    w = wide([1.2, 0.9, 1.4])
    inputs, scores = indicator_frames(CFG, w)
    month = w.index[-1]
    parts = driver_breakdown(CFG, "demand", inputs, scores, month)
    assert parts["contribution"].sum() == pytest.approx(
        driver_scores(CFG, w).at[month, "demand"])
    assert parts.set_index("id").at["b", "value"] == pytest.approx(1.4)


def test_ragged_edge_month_is_left_unscored_but_early_history_is_not():
    # a carries 0.6 of the weight, b 0.4. Require 0.7 of started weight.
    cfg = {**CFG, "meta": {"min_reported_share": 0.7}}
    idx = pd.date_range("2024-01-31", periods=4, freq="ME")
    # b has not started in month 1, and has not been released yet in month 4.
    w = pd.DataFrame({"A": [0.5, 1.0, 1.5, 2.0], "B": [np.nan, 0.9, 1.4, np.nan]}, index=idx)
    d = driver_scores(cfg, w)["demand"]
    assert not np.isnan(d.iloc[0]), "a series that had not started should not block a month"
    assert not np.isnan(d.iloc[2])
    assert np.isnan(d.iloc[3]), "a month missing a started input's release should be unscored"


def test_missing_indicator_is_excluded_and_weights_renormalise():
    w = wide([1.2, 0.9, np.nan])
    inputs, scores = indicator_frames(CFG, w)
    month = w.index[-1]
    parts = driver_breakdown(CFG, "demand", inputs, scores, month).set_index("id")
    assert np.isnan(parts.at["b", "share"])
    assert parts.at["a", "share"] == pytest.approx(1.0)
    assert parts["contribution"].sum() == pytest.approx(
        driver_scores(CFG, w).at[month, "demand"])
