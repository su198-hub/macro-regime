"""The forecast scorecard's grading: what makes it fair is what these pin down.

Held-out grading must not see the episode it grades, or anything sharing its
outcome window; real-time grading must not see a label before it was known;
the benchmarks must not use hindsight either.
"""

import numpy as np
import pandas as pd
import pytest

from src import outlook as ol

NAMES = ["g", "i"]


def frame(labels, g=None, i=None, start="2000-01"):
    idx = pd.period_range(start, periods=len(labels), freq="M")
    g = np.zeros(len(labels)) if g is None else np.asarray(g, dtype=float)
    i = np.zeros(len(labels)) if i is None else np.asarray(i, dtype=float)
    return pd.DataFrame({"g": g, "i": i, "label": labels}, index=idx)


JUDGMENT = pd.DataFrame({"g": [9.0, -9.0], "i": [0.0, 0.0]}, index=["up", "down"])


def test_purge_covers_every_month_sharing_an_outcome_window():
    idx = pd.period_range("2000-01", periods=200, freq="M")
    mask = ol.purged(idx, idx[100], idx[110])
    assert mask[100 - 35] and mask[110 + 35]
    assert not mask[100 - 36] and not mask[110 + 36]


def test_held_out_never_learns_from_the_episode_it_grades():
    # Two long episodes. Holding out "up" purges it and the 35 months after,
    # leaving only "down" months: "up" must fall back to its judgment position.
    tr = frame(["up"] * 60 + ["down"] * 100, g=[1.0] * 60 + [-1.0] * 100)
    p = ol.held_out(tr, JUDGMENT, NAMES, temperature=1.0)
    pos = pd.DataFrame({"g": [9.0, -1.0], "i": [0.0, 0.0]}, index=["up", "down"])
    want = ol.probabilities(tr.loc[tr.index[:60], NAMES], pos, 1.0)
    assert np.allclose(p.loc[tr.index[:60]].to_numpy(), want[["up", "down"]].to_numpy())


def test_real_time_uses_only_labels_known_at_the_time():
    # "down" first appears at month 60, so it is not known before month 96.
    tr = frame(["up"] * 60 + ["down"] * 100, g=[1.0] * 60 + [-1.0] * 100)
    p = ol.real_time(tr, JUDGMENT, NAMES, temperature=1.0, first="2000-01")
    early = tr.index[95]           # down not yet known: its judgment position stands
    later = tr.index[96]           # month 60's label known now
    pos_early = pd.DataFrame({"g": [1.0, -9.0], "i": [0.0, 0.0]}, index=["up", "down"])
    want = ol.probabilities(tr.loc[[early], NAMES], pos_early, 1.0).iloc[0]
    assert p.loc[early].to_numpy() == pytest.approx(want[["up", "down"]].to_numpy())
    assert p.loc[later, "down"] > p.loc[early, "down"]
    assert tr.index[35] not in p.index        # nothing known yet: no call


def test_base_rate_counts_only_known_labels():
    tr = frame(["up"] * 40 + ["down"] * 40)
    b = ol.base_rate(tr, ["up", "down"], first="2000-01")
    t = tr.index[40]               # labels for months 0-4 known: all "up"
    assert b.loc[t, "up"] == pytest.approx((5 + 1) / (5 + 2))


def test_log_loss_is_log_four_for_an_even_split_and_counts_baseline_as_a_miss():
    idx = pd.period_range("2000-01", periods=3, freq="M")
    even = pd.DataFrame(0.25, index=idx, columns=["a", "b", "c", "d"])
    labels = pd.Series(["a", "b", "c"], index=idx)
    assert ol.log_loss(even, labels) == pytest.approx(np.log(4))
    with_baseline = pd.DataFrame({"baseline": [1.0], "a": [0.0]}, index=idx[:1])
    assert ol.log_loss(with_baseline, labels.iloc[:1]) == pytest.approx(-np.log(1e-3))


def test_grade_counts_a_hard_landing_runup_as_caught_at_half_its_months():
    tr = frame(["hard_landing"] * 10 + ["goldilocks"] * 10 + ["hard_landing"] * 10)
    calls = pd.Series(["hard_landing"] * 5 + ["goldilocks"] * 5 + ["goldilocks"] * 10
                      + ["goldilocks"] * 6 + ["hard_landing"] * 4, index=tr.index)
    g = ol.grade(calls, tr["label"], tr)
    assert (g["hard_landing_runups"], g["hard_landing_caught"]) == (2, 1)
    assert g["months"] == pytest.approx(19 / 30)


def test_a_signpost_weight_scales_its_squared_distance():
    # Half weight on i: a point 2 away on i counts as sqrt(0.5 * 4) away.
    idx = pd.period_range("2000-01", periods=1, freq="M")
    scores = pd.DataFrame({"g": [0.0], "i": [2.0]}, index=idx)
    pos = pd.DataFrame({"g": [0.0, 1.0], "i": [0.0, 2.0]}, index=["a", "b"])
    even = ol.probabilities(scores, pos, 1.0)
    half = ol.probabilities(scores, pos, 1.0, weights={"i": 0.5})
    da, db = np.sqrt(0.5 * 4), 1.0
    want = np.exp(-da) / (np.exp(-da) + np.exp(-db))
    assert half.iloc[0]["a"] == pytest.approx(want)
    assert half.iloc[0]["a"] > even.iloc[0]["a"]


def test_realigned_positions_shrink_toward_a_scaled_prior():
    """Two goldilocks episodes at g = +0.2 and one hard landing at g = -0.1;
    a prior at +/-0.6. The prior is scaled to the learned spread (0.15 / 0.6),
    then weighted as `strength` episodes against each regime's episodes."""
    labels = (["goldilocks"] * 10 + ["hard_landing"] * 10 + ["goldilocks"] * 10)
    g = [0.2] * 10 + [-0.1] * 10 + [0.2] * 10
    train = frame(labels, g=g)
    prior = pd.DataFrame({"g": [0.6, -0.6, 0.0], "i": [0.0, 0.0, 0.3]},
                         index=["goldilocks", "hard_landing", "baseline"])
    pos = ol.realigned_positions(train, prior, NAMES, strength=0.5)
    scale = 0.075 / 0.3            # mean |learned| over g and i, over mean |prior| on the same cells
    assert pos.at["goldilocks", "g"] == pytest.approx((2 * 0.2 + 0.5 * 0.6 * scale) / 2.5)
    assert pos.at["hard_landing", "g"] == pytest.approx((1 * -0.1 + 0.5 * -0.6 * scale) / 1.5)
    # Baseline has no label: the average month, as one episode.
    assert pos.at["baseline", "g"] == pytest.approx((0.1 + 0) / 1.5)
    assert pos.at["baseline", "i"] == pytest.approx((0 + 0.5 * 0.3 * scale) / 1.5)
    # No weight on the prior: the run-up positions themselves.
    raw = ol.realigned_positions(train, prior, NAMES, strength=0.0)
    assert raw.at["goldilocks", "g"] == pytest.approx(0.2)


def test_realigned_positions_fall_back_to_the_prior_for_an_unseen_regime():
    train = frame(["goldilocks"] * 12, g=[0.3] * 12)
    prior = pd.DataFrame({"g": [0.6, -0.6], "i": [0.0, 0.0]},
                         index=["goldilocks", "stagflation"])
    pos = ol.realigned_positions(train, prior, NAMES)
    scale = 0.15 / 0.3
    assert pos.at["stagflation", "g"] == pytest.approx(-0.6 * scale)


def test_a_purged_hole_keeps_episodes_either_side_apart():
    """Goldilocks / hard landing (held out) / goldilocks: what is left is two
    goldilocks episodes, not one, so the data keeps its full weight against
    the prior."""
    labels = ["goldilocks"] * 100 + ["hard_landing"] * 40 + ["goldilocks"] * 160
    train = frame(labels)
    rest = train[~ol.purged(train.index, train.index[100], train.index[139])]
    sp = ol.spells(rest["label"])
    assert list(sp["label"]) == ["goldilocks", "goldilocks"]
