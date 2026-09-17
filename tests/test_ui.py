"""Signpost geometry. A flipped sign here draws a confident, wrong chart."""

import pandas as pd

PROSE_CFG = {"drivers": {
    "demand": {"label": "Demand", "signpost": {
        "low": {"label": "Weak", "phrase": "demand is running below trend"},
        "high": {"label": "Strong", "phrase": "demand is running above trend"},
        "mid": {"phrase": "demand is close to trend"}}},
    "investment": {"label": "Investment spending", "signpost": {
        "low": {"label": "Low", "phrase": "capex is contracting"},
        "high": {"label": "High", "phrase": "capex is expanding"},
        "mid": {"phrase": "capex is flat"}}},
}}
PROSE_REG = {"driver_salience": {"demand": 1.0, "investment": 1.0},
             "regimes": {"boom": {"label": "Boom",
                                  "archetype": {"demand": 0.6, "investment": 0.6}}}}
import pytest

from src.ui import (cluster, driver_phrase, month_mid, month_start, regime_runs, run_length,
                    objection, track_position, what_moved, why_called,
                    why_not_called)


def test_months_plot_mid_month_not_on_the_next_months_line():
    idx = pd.to_datetime(["2016-12-31", "2017-01-31"])
    assert list(month_mid(idx)) == [pd.Timestamp("2016-12-15"), pd.Timestamp("2017-01-15")]
    assert list(month_start(idx)) == [pd.Timestamp("2016-12-01"), pd.Timestamp("2017-01-01")]


def test_track_position_orders_low_to_high_and_clips():
    assert track_position(-1) < track_position(0) < track_position(1)
    assert track_position(0) == pytest.approx(50.0)
    assert track_position(5) == track_position(1)


def test_reverse_mirrors_about_the_centre():
    assert track_position(0.6, reverse=True) == pytest.approx(100 - track_position(0.6))
    assert track_position(0.6, reverse=True) < 50


def test_cluster_merges_near_neighbours_only():
    groups = cluster([(50.0, "a"), (55.0, "b"), (90.0, "c")], gap=10)
    assert [keys for _, keys in groups] == [["a", "b"], ["c"]]
    assert groups[0][0] == pytest.approx(52.5)


def test_run_length_counts_trailing_repeats():
    assert run_length(pd.Series(["x", "y", "y", "y"])) == 3
    assert run_length(pd.Series([], dtype=object)) == 0


def test_regime_runs_collapse_contiguous_months():
    idx = pd.date_range("2024-01-31", periods=5, freq="ME")
    runs = regime_runs(pd.Series(["a", "a", "b", "b", "a"], index=idx))
    assert runs["regime"].tolist() == ["a", "b", "a"]
    assert runs["start"].iloc[1] == idx[2]
    assert runs["end"].iloc[1] == idx[4]


# ---------- the sentences on the dashboard ----------

def test_driver_phrase_follows_the_score_into_the_right_band():
    assert driver_phrase("demand", 0.8, PROSE_CFG) == "demand is running above trend"
    assert driver_phrase("demand", -0.8, PROSE_CFG) == "demand is running below trend"
    assert driver_phrase("demand", 0.02, PROSE_CFG) == "demand is close to trend"


def test_why_called_names_what_matches_and_what_argues_against():
    row = pd.Series({"demand": 0.05, "investment": 0.62})
    text = why_called(row, PROSE_CFG, PROSE_REG, "boom")
    assert text == "Boom because capex is expanding."
    # Demand is furthest from the archetype, so it is the objection, and it is
    # not listed as a reason for the call.
    assert "demand" not in text
    assert objection(row, PROSE_CFG, PROSE_REG, "boom") == \
        "The main argument against: demand is close to trend."


def test_why_called_marks_a_loose_match_without_arithmetic():
    row = pd.Series({"demand": 0.05, "investment": 0.62})
    text = why_called(row, PROSE_CFG, PROSE_REG, "boom", weak=True)
    assert "though the match is loose" in text
    assert "distance" not in text.lower()


def test_why_not_called_names_the_two_worst_gaps():
    row = pd.Series({"demand": -0.7, "investment": -0.6})
    text = why_not_called(row, PROSE_CFG, PROSE_REG, "boom")
    assert text.startswith("Nearest is Boom, but")
    assert "demand is running below trend" in text and "capex is contracting" in text


def test_what_moved_reports_movers_and_holds_without_repeating_the_driver():
    before = pd.Series({"demand": 0.0, "investment": 0.30})
    now = pd.Series({"demand": 0.40, "investment": 0.31})
    text = what_moved(now, before, PROSE_CFG)
    assert text.startswith("demand is running above trend after rising")
    assert "demand has risen, so demand" not in text
    assert "investment spending are little changed" in text


def test_what_moved_says_what_held_when_nothing_moved():
    before = pd.Series({"demand": 0.20, "investment": 0.30})
    now = pd.Series({"demand": 0.21, "investment": 0.31})
    text = what_moved(now, before, PROSE_CFG)
    assert text.startswith("little has moved")
    assert "demand" in text and "investment spending" in text

