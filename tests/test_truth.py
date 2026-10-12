import numpy as np
import pandas as pd
import pytest

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


def test_growth_gap_is_zero_when_gdp_tracks_potential():
    q = pd.period_range("2000Q1", periods=12, freq="Q").asfreq("M", how="start")
    path = pd.Series(100 * 1.005 ** np.arange(12), index=q)
    parts = truth.growth_parts(path, path * 1.0)
    assert (parts["gdp_growth"] - parts["potential_growth"]).abs().max() < 1e-9
    assert abs(parts["gdp_growth"].iloc[6] - (1.005 ** 4 - 1) * 100) < 1e-9


def test_centred_window_runs_to_latest_print_at_the_end():
    s = pd.Series(100 * 1.01 ** np.arange(24), index=months("2000-01", 24))
    c = truth._centred(s, 6, 12)
    assert abs(c.iloc[10] - (1.01 ** 12 - 1) * 100) < 1e-9
    assert abs(c.iloc[-1] - (1.01 ** 12 - 1) * 100) < 1e-9


def test_store_round_trip():
    idx = months("2000-01", 6)
    m = pd.DataFrame({name: np.arange(6, dtype=float) + i for i, name in enumerate(truth.CODES)}, index=idx)
    obs = truth.to_observations(m, pd.Timestamp("2026-10-07").date())
    wide = obs.pivot(index="observation_date", columns="series_id", values="value")
    back = truth.from_store(wide)
    pd.testing.assert_frame_equal(back, m, check_freq=False, check_names=False)


def test_spells_collapse_runs_in_order():
    idx = months("2000-01", 6)
    lab = pd.DataFrame({
        "label": ["goldilocks"] * 3 + ["stagflation"] * 3, "strength": ["clear", "weak"] * 3,
        "recession": [False] * 4 + [True] * 2,
        "provisional": [False] * 5 + [True],
        **{c: 1.0 for c in ("gdp_growth", "potential_growth", "growth",
                            "core_pce", "expected", "inflation")}}, index=idx)
    sp = truth.spells(lab)
    assert sp["label"].tolist() == ["goldilocks", "stagflation"]
    assert sp["months"].tolist() == [3, 3]
    assert sp["recession_months"].tolist() == [0, 2]
    assert sp["provisional"].tolist() == [False, True]
    assert abs(sp["clear"].iloc[0] - 2 / 3) < 1e-9


def test_provisional_months_keep_the_last_confirmed_regime(monkeypatch):
    idx = months("2000-01", 8)
    ax = pd.DataFrame({"growth": [1.0] * 5 + [-1.0] * 3, "inflation": [-1.0] * 8,
                       "provisional": [False] * 5 + [True] * 3}, index=idx)
    monkeypatch.setattr(truth, "axes", lambda m, *a, **k: ax)
    lab = truth.build(None, start="2000-01")
    assert lab["label"].tolist() == ["goldilocks"] * 8
    assert lab["provisional_reading"].tolist()[-3:] == ["hard_landing"] * 3


def test_forward_window_starts_in_the_month_and_stops_short():
    s = pd.Series(100 * 1.01 ** np.arange(48), index=months("2000-01", 48))
    f = truth._windowed(s, 1, 12, "forward")
    assert abs(f.iloc[0] - (1.01 ** 12 - 1) * 100) < 1e-9
    assert f.iloc[-12:].isna().all() and not np.isnan(f.iloc[-13])


def test_band_shrinks_with_the_window():
    assert truth.band(1) == 0.5
    assert abs(truth.band(4) - 0.25) < 1e-12


def test_what_followed_compares_with_the_most_common_label():
    idx = months("2000-01", 24)
    now = pd.Series(["goldilocks"] * 8 + ["hard_landing"] * 16, index=idx)
    fwd = pd.DataFrame({"label": ["goldilocks", "hard_landing"]}, index=idx[:2])
    wf = truth.what_followed(fwd, now, 1)
    assert wf["most_common"].tolist() == ["goldilocks", "goldilocks"]
    assert wf["agree"].tolist() == [True, False]


def test_averaged_measure_damps_a_one_off_jump_at_the_end():
    lvl = pd.Series(100 * 1.002 ** np.arange(60), index=months("2000-01", 60))
    lvl.iloc[36:] *= 1.01   # one-off level jump: every later change carries it once
    ends = truth._windowed(lvl, 1, 12, "forward")
    avg = truth._averaged(lvl, 12, 36, "forward", 12)
    steady = (1.002 ** 12 - 1) * 100
    # A window ending just after the jump: the start-to-end change takes all of it.
    assert abs(ends.iloc[24] - steady) > abs(avg.iloc[0] - steady)


def test_back_to_back_changes_chain_to_the_endpoints():
    lvl = pd.Series([100, 101.6, 103.2, 108.7], index=months("2000-01", 4))
    chained = np.prod([lvl.iloc[k + 1] / lvl.iloc[k] for k in range(3)])
    assert abs(chained - lvl.iloc[3] / lvl.iloc[0]) < 1e-12


def test_nowcast_averages_one_year_stretches_over_18_months_with_a_12_month_minimum():
    now = truth.METHODS["now"]
    assert now == truth.DEFAULT
    assert now["measure"] == "averaged"
    assert truth.window_months(now["years"], now["align"], now["measure"]) == 18
    assert truth.stretches(now["years"], now["align"]) == (7, 12)
    assert now["min_months"] == 12


def test_forward_labels_average_and_hold_a_regime_a_year():
    assert truth.stretches(3, "forward") == (25, 12)
    assert truth.stretches(5, "forward") == (37, 24)
    for view in ("next3", "next5"):
        assert truth.METHODS[view]["measure"] == "averaged"
        assert truth.METHODS[view]["min_months"] == 12


def test_averaged_window_is_the_mean_of_the_small_changes_inside_it():
    idx = pd.period_range("2000-01", periods=40, freq="M")
    level = pd.Series(100 * 1.003 ** np.arange(40) * (1 + 0.01 * (np.arange(40) % 5 == 0)), index=idx)
    out = truth._averaged(level, 6, 18, "centered", 12)
    t = idx[20]
    ends = pd.period_range(t - 9 + 6, t + 9, freq="M")
    want = np.mean([((level[e] / level[e - 6]) ** 2 - 1) * 100 for e in ends])
    assert len(ends) == 13
    assert out[t] == pytest.approx(want)


def test_minimum_regime_length_absorbs_short_spells():
    idx = months("2000-01", 16)
    ax = pd.DataFrame({"growth": [1.0] * 6 + [-1.0] * 4 + [1.0] * 6, "inflation": [-1.0] * 16}, index=idx)
    three = truth.primary(ax, 0.5, 0.5, 3)["label"].tolist()
    six = truth.primary(ax, 0.5, 0.5, 6)["label"].tolist()
    assert three[6:10] == ["hard_landing"] * 4
    assert six == ["goldilocks"] * 16
