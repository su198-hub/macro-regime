"""Tests for the parts that fail silently.

A sign error in a transform or an off-by-one in persistence does not raise.
It just produces a confident, wrong dashboard. These are the tests worth having.
"""

import numpy as np
import pandas as pd
import pytest

from src.regimes import UNCLASSIFIED, call_regime, fit, probabilities, run
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


def test_a_neutral_month_fits_no_regime_even_though_probabilities_sum_to_one():
    d = pd.DataFrame({"a": [0.0, 0.9], "b": [0.0, 0.9]},
                     index=pd.date_range("2024-01-31", periods=2, freq="ME"))
    f = fit(d, REG)
    # All-neutral is exactly as far from "hot" as neutral itself is: not closer, so no fit.
    assert not f["fits"].iloc[0]
    assert f["fits"].iloc[1] and f["nearest"].iloc[1] == "hot"


def test_fit_gate_moves_the_call_to_unclassified_only_after_persistence():
    cfg = {**REG, "settings": {**REG["settings"], "fit_gate": "closer_than_neutral"}}
    d = pd.DataFrame({"a": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                      "b": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]},
                     index=pd.date_range("2024-01-31", periods=6, freq="ME"))
    calls = run(d, cfg)["calls"]
    assert calls["called"].tolist() == ["hot"] * 5 + [UNCLASSIFIED]
    assert calls["state"].iloc[3] == UNCLASSIFIED


def test_tolerance_grades_near_misses_as_weak_and_still_calls_them():
    # Hot sits sqrt(2) = 1.41 from neutral; a tolerance of 1.25 allows up to 1.77.
    cfg = {**REG, "settings": {**REG["settings"], "fit_gate": "closer_than_neutral",
                               "fit_tolerance": 1.25}}
    # (1.5, -1.5) is 2.55 from both archetypes: beyond the tolerance of either.
    d = pd.DataFrame({"a": [0.9, 0.0, 1.5], "b": [0.9, 0.0, -1.5]},
                     index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    f = fit(d, cfg)
    assert f["fit"].tolist() == ["clear", "weak", "none"]
    assert f["limit"].iloc[0] == pytest.approx(np.sqrt(2) * 1.25)
    held = pd.DataFrame({"a": [0.0] * 4, "b": [0.0] * 4},
                        index=pd.date_range("2024-01-31", periods=4, freq="ME"))
    assert UNCLASSIFIED not in set(run(held, cfg)["calls"]["called"])


def test_gate_off_keeps_the_old_behaviour():
    d = pd.DataFrame({"a": [0.0], "b": [0.0]}, index=pd.to_datetime(["2024-01-31"]))
    assert run(d, REG)["calls"]["called"].iloc[0] in {"hot", "cold"}


def test_confidence_floor_reports_transitional():
    cfg = {**REG, "settings": {**REG["settings"], "min_confidence": 0.99}}
    d = pd.DataFrame({"a": [0.0], "b": [0.0]},
                     index=pd.to_datetime(["2024-01-31"]))
    assert call_regime(probabilities(d, cfg), cfg)["called"].iloc[0] == \
        "transitional"


# ---------- transforms added for the v2 signposts ----------

def _monthly(values, start="2000-01"):
    return pd.Series(values, index=pd.period_range(start, periods=len(values), freq="M").to_timestamp())


def test_pct_change_3y_ann_is_the_annual_rate_over_three_years():
    from src.transform import pct_change_3y_ann
    s = _monthly(100 * 1.05 ** (np.arange(48) / 12))
    assert pct_change_3y_ann(s).iloc[-1] == pytest.approx(5.0)
    assert pct_change_3y_ann(s).iloc[:36].isna().all()


def test_mean_5y_averages_the_rate_once_four_years_are_in():
    from src.transform import mean_5y
    s = pd.Series(np.arange(24, dtype=float),
                  index=pd.period_range("2000Q1", periods=24, freq="Q").to_timestamp())
    out = mean_5y(s)
    assert out.iloc[14] != out.iloc[14]        # 15 quarters: not yet
    assert out.iloc[15] == pytest.approx(np.mean(np.arange(16)))
    assert out.iloc[-1] == pytest.approx(np.mean(np.arange(4, 24)))


def test_mean_36m_needs_the_full_three_years():
    from src.transform import mean_36m
    s = _monthly(np.ones(40))
    out = mean_36m(s)
    assert out.iloc[34] != out.iloc[34] and out.iloc[35] == pytest.approx(1.0)


def test_dev_10y_is_points_above_the_trailing_decade():
    from src.transform import dev_10y
    s = _monthly(np.r_[np.full(100, 10.0), np.full(20, 12.0)])
    out = dev_10y(s)
    assert out.iloc[70] != out.iloc[70]        # under six years: not yet
    assert out.iloc[71] == pytest.approx(0.0)  # six years in
    assert out.iloc[99] == pytest.approx(0.0)
    assert out.iloc[-1] == pytest.approx(12.0 - np.mean(np.r_[np.full(100, 10.0), np.full(20, 12.0)]))


def test_target_distance_change_is_positive_when_inflation_drifts_away():
    from src.transform import target_distance_change_12m
    # Inflation steps from 2% in the first year to 4% in the second.
    monthly = np.r_[np.full(24, 1.02 ** (1 / 12)), np.full(24, 1.04 ** (1 / 12))]
    s = _monthly(100 * np.cumprod(monthly))
    out = target_distance_change_12m(s)
    # Month 36: inflation over the latest year is 4%, over the year before it 2%.
    assert out.iloc[35] == pytest.approx(2.0, abs=1e-6)       # from 0 to 2 points away
    assert out.iloc[-1] == pytest.approx(0.0, abs=1e-6)       # still 2 away a year later
    falling = _monthly(100 * np.cumprod(np.r_[np.full(24, 1.04 ** (1 / 12)), np.full(24, 1.02 ** (1 / 12))]))
    assert target_distance_change_12m(falling).iloc[35] == pytest.approx(-2.0, abs=1e-6)
