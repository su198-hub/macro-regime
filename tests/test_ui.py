"""Signpost geometry. A flipped sign here draws a confident, wrong chart."""

import pandas as pd
import pytest

from src.ui import cluster, month_mid, month_start, regime_runs, run_length, track_position


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
