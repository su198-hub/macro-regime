"""Tests for the parts that fail silently.

A sign error in a transform or an off-by-one in persistence does not raise.
It just produces a confident, wrong dashboard. These are the tests worth having.
"""

import numpy as np
import pandas as pd
import pytest

from src.regimes import call_regime, probabilities
from src.transform import normalise, pct_change_3m_ann, sum_12m, to_monthly, yoy_pct


def test_sum_12m_needs_a_full_year_and_sums_it():
    s = pd.Series(range(1, 15), index=pd.date_range("2020-01-31", periods=14, freq="ME"),
                  dtype=float)
    out = sum_12m(s)
    assert out.iloc[:11].isna().all()
    assert out.iloc[11] == pytest.approx(sum(range(1, 13)))
    assert out.iloc[13] == pytest.approx(sum(range(3, 15)))


def monthly(values, start="2020-01-31"):
    return pd.Series(values,
                     index=pd.date_range(start, periods=len(values), freq="ME"))


# ---------- transforms ----------

def test_yoy_pct_on_monthly_uses_twelve_periods():
    s = monthly([100.0] * 12 + [110.0])
    assert yoy_pct(s).iloc[-1] == pytest.approx(10.0)


def test_three_month_annualised_compounds():
    s = monthly([100, 100, 100, 101, 102, 103.0301])
    # 1% per month for three months annualises to about 12.68%
    assert pct_change_3m_ann(s).iloc[-1] == pytest.approx(12.68, abs=0.1)


def test_gap_normalisation_is_signed_and_centred():
    s = monthly([2.0, 2.3, 1.7])
    out = normalise(s, {"method": "gap", "center": 2.0, "scale": 0.3})
    assert out.iloc[0] == pytest.approx(0.0)
    assert out.iloc[1] == pytest.approx(1.0)
    assert out.iloc[2] == pytest.approx(-1.0)


def test_normalisation_clips_extremes():
    s = monthly([2.0, 99.0])
    out = normalise(s, {"method": "gap", "center": 2.0, "scale": 0.3})
    assert out.iloc[1] == 3.0


def test_quarterly_series_is_carried_to_the_latest_month_within_its_limit():
    q = pd.Series([1.0, 2.0, 3.0, 4.0],
                  index=pd.to_datetime(["2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01"]))
    carried = to_monthly(q, end=pd.Timestamp("2026-09-15"), carry_periods=2)
    assert carried.index[-1] == pd.Timestamp("2026-09-30")
    assert carried.iloc[-1] == 4.0
    # Two quarters is six months: April's value lasts to October, not beyond.
    stale = to_monthly(q, end=pd.Timestamp("2027-01-15"), carry_periods=2)
    assert stale.loc["2026-10-31"] == 4.0
    assert np.isnan(stale.loc["2026-11-30"])


def test_monthly_series_is_never_extended():
    m = monthly([1.0, 2.0, 3.0])
    out = to_monthly(m, end=m.index[-1] + pd.DateOffset(months=4), carry_periods=2)
    assert out.index[-1] == m.index[-1]


def test_gap_rejects_zero_scale():
    with pytest.raises(ValueError):
        normalise(monthly([1.0]), {"method": "gap", "center": 0, "scale": 0})


# ---------- regime engine ----------

REG = {
    "settings": {"temperature": 0.3, "persistence_months": 3,
                 "min_confidence": 0.0},
    "driver_salience": {"a": 1.0, "b": 1.0},
    "regimes": {
        "hot": {"archetype": {"a": 1.0, "b": 1.0}},
        "cold": {"archetype": {"a": -1.0, "b": -1.0}},
    },
}


def test_probabilities_sum_to_one_and_favour_the_near_archetype():
    d = pd.DataFrame({"a": [0.9], "b": [0.9]},
                     index=pd.to_datetime(["2024-01-31"]))
    p = probabilities(d, REG)
    assert p.sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert p["hot"].iloc[0] > p["cold"].iloc[0]


def test_missing_driver_yields_no_call_rather_than_a_guess():
    d = pd.DataFrame({"a": [0.9], "b": [np.nan]},
                     index=pd.to_datetime(["2024-01-31"]))
    assert probabilities(d, REG).isna().all(axis=1).iloc[0]


def test_a_driver_with_no_column_at_all_also_yields_no_call():
    # Early in a rewind a whole driver can be absent, not just NaN.
    d = pd.DataFrame({"a": [0.9]}, index=pd.to_datetime(["2024-01-31"]))
    assert probabilities(d, REG).isna().all(axis=1).iloc[0]


def test_unscored_months_carry_the_call_without_raising():
    d = pd.DataFrame(
        {"a": [np.nan, 1.0, np.nan, 1.0],
         "b": [np.nan, 1.0, np.nan, 1.0]},
        index=pd.date_range("2024-01-31", periods=4, freq="ME"),
    )
    calls = call_regime(probabilities(d, REG), REG)
    assert pd.isna(calls["called"].iloc[0])
    assert calls["called"].iloc[1:].tolist() == ["hot", "hot", "hot"]
    assert pd.isna(calls["leading"].iloc[2])


def test_switch_requires_persistence_and_then_happens():
    # Two cold months should not flip the call; the third should.
    d = pd.DataFrame(
        {"a": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
         "b": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]},
        index=pd.date_range("2024-01-31", periods=6, freq="ME"),
    )
    called = call_regime(probabilities(d, REG), REG)["called"].tolist()
    assert called[:3] == ["hot"] * 3
    assert called[3] == "hot", "flipped on the first challenging month"
    assert called[4] == "hot", "flipped before persistence was met"
    assert called[5] == "cold", "never flipped despite a sustained run"


def test_a_single_outlier_month_does_not_flip_the_call():
    d = pd.DataFrame(
        {"a": [1.0, 1.0, -1.0, 1.0, 1.0],
         "b": [1.0, 1.0, -1.0, 1.0, 1.0]},
        index=pd.date_range("2024-01-31", periods=5, freq="ME"),
    )
    assert set(call_regime(probabilities(d, REG), REG)["called"]) == {"hot"}


def test_confidence_floor_reports_transitional():
    cfg = {**REG, "settings": {**REG["settings"], "min_confidence": 0.99}}
    d = pd.DataFrame({"a": [0.0], "b": [0.0]},
                     index=pd.to_datetime(["2024-01-31"]))
    assert call_regime(probabilities(d, cfg), cfg)["called"].iloc[0] == \
        "transitional"
