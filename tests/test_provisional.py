"""Confirmed versus provisional readings of the latest months."""

import numpy as np
import pandas as pd
import pytest

from src.drivers import (driver_scores, expected_release, indicator_frames, median_release_lag,
                         provisional_months, provisional_reading)

CFG = {
    "meta": {"min_reported_share": 0.6, "provisional": {"fill_months": 2, "min_reported_share": 0.5}},
    "drivers": {"demand": {"indicators": [
        {"id": "hard", "source": {"fred": "H"}, "transform": "level",
         "normalize": {"method": "gap", "center": 0.0, "scale": 1.0},
         "direction": 1, "weight": 0.3, "anchor": True},
        {"id": "fast", "source": {"fred": "F"}, "transform": "level",
         "normalize": {"method": "gap", "center": 0.0, "scale": 1.0},
         "direction": 1, "weight": 0.7},
    ]}},
}
IDX = pd.date_range("2026-01-31", periods=6, freq="ME")


def frames(h, f):
    wide = pd.DataFrame({"H": h, "F": f}, index=IDX)
    return wide, indicator_frames(CFG, wide)[1]


def test_anchor_holds_back_confirmation_even_with_enough_weight():
    wide, scores = frames([1, 1, 1, 1, np.nan, np.nan], [1, 1, 1, 1, 2, 2])
    d = driver_scores(CFG, wide, scores)["demand"]
    assert not np.isnan(d.iloc[3])
    # 70% of weight reported clears 60%, but the anchor has not reported.
    assert np.isnan(d.iloc[4]) and np.isnan(d.iloc[5])


def test_provisional_carries_the_anchor_and_marks_it():
    wide, scores = frames([1, 1, 1, 1, np.nan, np.nan], [1, 1, 1, 1, 2, 2])
    r = provisional_reading(CFG, scores, IDX[4])
    status = r["status"].set_index("id")["status"]
    assert status["hard"] == "carried" and status["fast"] == "reported"
    assert r["reported"]["demand"] == pytest.approx(0.7)
    # (0.3 * 1 + 0.7 * 2) / 1.0, halved by DRIVER_SCALE
    assert r["drivers"]["demand"] == pytest.approx(0.85)


def test_carry_stops_after_fill_months():
    wide, scores = frames([1, 1, np.nan, np.nan, np.nan, np.nan], [1, 1, 1, 1, 2, 2])
    status = provisional_reading(CFG, scores, IDX[5])["status"].set_index("id")["status"]
    assert status["hard"] == "pending"


def test_provisional_months_start_after_the_last_confirmed_month():
    wide, scores = frames([1, 1, 1, 1, np.nan, np.nan], [1, 1, 1, 1, 2, np.nan])
    d = driver_scores(CFG, wide, scores)
    months = [r["month"] for r in provisional_months(CFG, d, scores)]
    # Month 5 has the fast indicator; month 6 has nothing new, so it is not shown.
    assert months == [IDX[4]]


def test_release_lag_and_expected_date():
    pub = pd.DataFrame({"observation_date": pd.to_datetime(["2026-05-01", "2026-06-01", "2026-07-01"]),
                        "published": pd.to_datetime(["2026-06-26", "2026-07-31", "2026-08-28"])})
    lag = median_release_lag(pub, 12)
    assert lag == 28
    assert expected_release(pd.Timestamp("2026-08-31"), 12, lag) == pd.Timestamp("2026-09-28")
    assert expected_release(pd.Timestamp("2026-08-31"), 4, 36) == pd.Timestamp("2026-11-05")
