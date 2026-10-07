import numpy as np
import pandas as pd

from src import truth


def months(start, n):
    return pd.period_range(start, periods=n, freq="M")


def test_hysteresis_holds_inside_band():
    x = pd.Series([1.0, 0.1, -0.1, -0.2, -1.0, 0.2], index=months("2000-01", 6))
    h = truth._hysteresis(x, 0.5)
    assert h["up"].tolist() == [True, True, True, True, False, False]
    assert h["clear"].tolist() == [True, False, False, False, True, False]


def test_short_spells_take_the_label_before():
    s = pd.Series(["a"] * 4 + ["b"] * 2 + ["a"] * 3 + ["c"] * 3, index=months("2000-01", 12))
    assert truth._persist(s, 3).tolist() == ["a"] * 9 + ["c"] * 3


def test_primary_quadrants():
    idx = months("2000-01", 12)
    ax = pd.DataFrame({"growth": [1.0] * 6 + [-1.0] * 6,
                       "inflation": [-1.0] * 3 + [1.0] * 6 + [-1.0] * 3}, index=idx)
    lab = truth.primary(ax, 0.25, 0.5, 3)["label"].tolist()
    assert lab == (["goldilocks"] * 3 + ["high_growth_high_inflation"] * 3
                   + ["stagflation"] * 3 + ["hard_landing"] * 3)


def test_quarterly_surprise_is_realised_less_forecast():
    idx = months("2000-01", 36)
    level = pd.Series(100 * 1.01 ** (np.arange(36) / 3), index=idx)  # 1% a quarter
    fc = pd.Series(3.0, index=pd.period_range("2000-01", periods=12, freq="Q").asfreq("M", how="start"))
    s = truth._quarterly_surprise(level, None, fc)
    assert abs(s.dropna().iloc[-1] - (1.01 ** 4 - 1) * 100 + 3.0) < 1e-6


def test_gdp_gap_growth_is_zero_when_gdp_tracks_potential():
    q = pd.period_range("2000Q1", periods=12, freq="Q").asfreq("M", how="start")
    path = pd.Series(100 * 1.005 ** np.arange(12), index=q)
    g = truth.gdp_gap_growth(path, path * 1.0)
    assert g.abs().max() < 1e-9
