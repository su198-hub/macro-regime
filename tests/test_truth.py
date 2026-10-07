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
