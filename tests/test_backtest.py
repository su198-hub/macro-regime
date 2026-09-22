"""The ex-post benchmark: each rule labels the regime it claims to."""

import numpy as np
import pandas as pd
import pytest

from src import backtest as bt


def economy(start, months, gap_path, inflation):
    """A synthetic economy: an unemployment gap path and a constant inflation rate."""
    idx = pd.date_range(start, periods=months, freq="ME")
    price = 100 * (1 + inflation / 100) ** (np.arange(months) / 12)
    return pd.DataFrame({"UNRATE": 5.0 + np.asarray(gap_path, float), "NROU": 5.0,
                         "PCEPILFE": price}, index=idx)


def label(frame, when):
    return bt.realised(frame).loc[pd.Period(when, "M")]


def test_steady_growth_and_low_inflation_is_goldilocks():
    assert label(economy("1995-01", 60, np.zeros(60), 2.0), "1998-01") == "goldilocks"


def test_steady_growth_and_high_inflation_is_high_growth_high_inflation():
    assert label(economy("1995-01", 60, np.zeros(60), 4.0), "1998-01") == \
        "high_growth_high_inflation"


def test_rising_unemployment_and_high_inflation_is_stagflation():
    rising = np.linspace(0, 3, 60)
    assert label(economy("1995-01", 60, rising, 4.0), "1998-01") == "stagflation"


def test_slow_growth_with_low_inflation_and_no_recession_is_unclassified():
    rising = np.linspace(0, 3, 60)
    assert label(economy("1995-01", 60, rising, 1.5), "1998-01") == bt.UNCLASSIFIED


def test_a_recession_in_the_window_is_a_hard_landing_whatever_else_happens():
    # 2001 recession, March-November; January 2001 sits within six months of it.
    assert label(economy("1998-01", 72, np.zeros(72), 2.0), "2001-01") == "hard_landing"


def test_months_without_a_full_window_are_not_labelled():
    """Five months before and six after are needed, so 30 months give 19 labels."""
    t = bt.realised(economy("1995-01", 30, np.zeros(30), 2.0))
    assert len(t) == 19
    assert t.index.min() == pd.Period("1995-06", "M")
    assert t.index.max() == pd.Period("1996-12", "M")


def test_the_scorecard_counts_what_it_says():
    months = pd.period_range("2000-01", periods=4, freq="M")
    truth = pd.Series(["goldilocks", "hard_landing", "hard_landing", "goldilocks"], index=months)
    calls = {"a": pd.Series(["goldilocks", "hard_landing", "goldilocks", "hard_landing"],
                            index=months.to_timestamp(how="end")),
             "b": pd.Series(["goldilocks", "transitional", "hard_landing", "goldilocks"],
                            index=months.to_timestamp(how="end"))}
    board = bt.scorecard(calls, truth)
    assert board.loc["Months called correctly", "a"] == pytest.approx(0.5)
    assert board.loc["Hard-landing calls, months", "a"] == 2
    assert board.loc["Hard-landing calls that were right", "a"] == pytest.approx(0.5)
    assert board.loc["Recession months caught", "a"] == pytest.approx(0.5)
    # a: one recession called Goldilocks, one hard landing called in Goldilocks
    assert board.loc["Costly errors, months", "a"] == 2
    # b: transitional scores as declining to call, so it is neither right nor costly
    assert board.loc["Costly errors, months", "b"] == 0
    assert board.attrs["naive"][1] == pytest.approx(0.5)
