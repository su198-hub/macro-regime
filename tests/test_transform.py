"""Transforms added for the reforms signpost."""

import numpy as np
import pandas as pd
import pytest

from src.transform import TRANSFORMS


def test_mean_12m_is_a_trailing_year_at_any_frequency():
    m = pd.Series(np.arange(24, dtype=float), index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    out = TRANSFORMS["mean_12m"](m)
    assert out.iloc[:11].isna().all()
    assert out.iloc[11] == pytest.approx(5.5)
    q = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2020-03-31", periods=5, freq="QE"))
    assert TRANSFORMS["mean_12m"](q).iloc[-1] == pytest.approx(3.5)


def test_diff_5y_puts_five_yearly_points_on_a_yearly_footing():
    """Fraser-style: every five years to 2000, yearly after. Five steps must
    be five years, not five observations."""
    years = [1990, 1995, 2000, 2001, 2002, 2003, 2004, 2005]
    s = pd.Series([5.0, 6.0, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0],
                  index=pd.to_datetime([f"{y}-01-01" for y in years]))
    out = TRANSFORMS["diff_5y"](s).dropna()
    assert out.index[0] == pd.Timestamp("1995-12-31")
    assert out.loc["2005-12-31"] == pytest.approx(8.0 - 7.0)
    assert out.loc["1997-12-31"] == pytest.approx(6.4 - 5.4)   # interpolated between 1995 and 2000


def test_diff_5y_on_a_monthly_series_is_sixty_months():
    m = pd.Series(np.arange(80, dtype=float), index=pd.date_range("2000-01-31", periods=80, freq="ME"))
    assert TRANSFORMS["diff_5y"](m).dropna().eq(60.0).all()


def test_diff_5y_matches_by_date_across_a_gap():
    """A two-year hole must not stretch the window to seven years."""
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    s = pd.Series(np.arange(120, dtype=float), index=idx).drop(idx[24:48])
    out = TRANSFORMS["diff_5y"](s).dropna()
    assert out.eq(60.0).all()
    assert pd.Timestamp("2007-01-31") not in out.index      # 2002-01 is in the hole


def test_diff_5y_finds_feb_28_five_years_after_feb_29():
    idx = pd.date_range("2016-01-31", "2021-03-31", freq="ME")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    out = TRANSFORMS["diff_5y"](s)
    assert out.loc["2021-02-28"] == pytest.approx(60.0)
