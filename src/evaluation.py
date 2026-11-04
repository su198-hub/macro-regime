"""How well the three-year call has done: the figures on the methodology page.

Every number is recomputed from the store and the configs, through the same
grading as scripts/realign_positions.py, so the page cannot drift from the
model. The answer key is the three-year forward label (src/truth.py); months
are graded from 1976 to the last month with three full years after it.

  held out    each episode called from positions learned without it or any
              month sharing its outcome window (the headline)
  real time   each month called from positions learned only from outcomes
              known by then
Both read today's revised driver scores, with centers and scales set on the
full history, so neither is a test on data as it was first published. The
live call log is.

Months next to each other share most of their answer, so the uncertainty is
measured by resampling whole episodes, not months.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd

from . import outlook as ol
from . import regimes as R
from . import truth

FIRST = "1976-01"
ODDS_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]


def _calls(probs: pd.DataFrame, fits: pd.Series, reg: dict, window) -> pd.Series:
    gate = reg["settings"].get("fit_gate") == "closer_than_neutral"
    return R.call_regime(probs, reg, fits if gate else None)["called"].reindex(window)


def _without(reg: dict, names: list[str]) -> dict:
    r = copy.deepcopy(reg)
    r["driver_salience"] = {k: v for k, v in reg["driver_salience"].items() if k in names}
    return r


def by_regime(calls: pd.Series, labels: pd.Series, regimes: list[str]) -> pd.DataFrame:
    """Precision, recall and F1 for each regime, and how often it followed.

    Precision: of the months called that regime, the share it followed.
    Recall: of the months it followed, the share called it. A call outside the
    four regimes (transitional, no clear regime) counts against recall only.
    """
    keep = labels.notna()
    c, t = calls[keep], labels[keep]
    rows = {}
    for r in regimes:
        called, true = c == r, t == r
        hit = int((called & true).sum())
        p = hit / called.sum() if called.sum() else np.nan
        rc = hit / true.sum() if true.sum() else np.nan
        rows[r] = {"precision": p, "recall": rc,
                   "f1": 2 * p * rc / (p + rc) if p + rc else np.nan,
                   "called": int(called.sum()), "followed": int(true.sum()),
                   "base_rate": float(true.mean())}
    return pd.DataFrame(rows).T


def axes(calls: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """The same three measures on each axis alone: growth below potential
    (stagflation or hard landing) and inflation above expectations
    (stagflation or high growth, high inflation)."""
    keep = labels.notna() & calls.isin(truth.QUADRANT.values())
    c, t = calls[keep], labels[keep]
    sides = {"growth_below": {r for (up_g, _), r in truth.QUADRANT.items() if not up_g},
             "inflation_above": {r for (_, up_i), r in truth.QUADRANT.items() if up_i}}
    rows = {}
    for name, side in sides.items():
        called, true = c.isin(side), t.isin(side)
        hit = (called & true).sum()
        p, rc = hit / called.sum(), hit / true.sum()
        rows[name] = {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc),
                      "called": int(called.sum()), "followed": int(true.sum()),
                      "base_rate": float(true.mean())}
    return pd.DataFrame(rows).T


def bootstrap(calls: pd.Series, labels: pd.Series, regimes: list[str],
              draws: int = 2000, seed: int = 0) -> dict:
    """5th and 95th percentiles of the balanced score and of each regime's
    precision, resampling whole episodes with replacement."""
    keep = labels.notna()
    c, t = calls[keep], labels[keep]
    sp = ol.spells(t)
    # Per episode: months, hits, and how many of its months were called each regime.
    n = sp["months"].to_numpy(float)
    hits = np.array([(c.loc[s:e] == lab).sum() for s, e, lab in zip(sp.start, sp.end, sp.label)], float)
    called = np.array([[(c.loc[s:e] == r).sum() for r in regimes] for s, e in zip(sp.start, sp.end)], float)
    own = np.array([[lab == r for r in regimes] for lab in sp.label], float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sp), (draws, len(sp)))
    w = np.stack([np.bincount(row, minlength=len(sp)) for row in pick]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = (w @ (own * hits[:, None])) / (w @ (own * n[:, None]))
        balanced = np.nanmean(recall, axis=1)
        precision = (w @ (own * hits[:, None])) / (w @ called)
    with warnings.catch_warnings():
        # A regime never called has no precision in any draw.
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {"balanced": tuple(np.nanpercentile(balanced, [5, 95])), "episodes": len(sp)}
        out["precision"] = {r: tuple(np.nanpercentile(precision[:, i], [5, 95]))
                            for i, r in enumerate(regimes)}
    return out


def calibration(probs: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """For bands of the leading regime's odds: months, the odds stated, and how
    often the leader is what followed."""
    keep = labels.notna() & probs.notna().all(axis=1)
    p, t = probs[keep], labels[keep]
    lead = p.max(axis=1)
    right = p.idxmax(axis=1) == t
    band = pd.cut(lead, ODDS_BINS)
    out = pd.DataFrame({"months": right.groupby(band, observed=True).size(),
                        "stated": lead.groupby(band, observed=True).mean(),
                        "right": right.groupby(band, observed=True).mean()})
    return out[out["months"] > 0]


def longest_misses(calls: pd.Series, labels: pd.Series, k: int = 4) -> pd.DataFrame:
    """The longest unbroken runs of one wrong call against one outcome."""
    keep = labels.notna() & calls.notna()
    c, t = calls[keep], labels[keep]
    wrong = c != t
    key = c.where(wrong) + "|" + t.where(wrong)
    run = ((key != key.shift()) | ~wrong).cumsum()
    rows = [{"start": g.index[0], "end": g.index[-1], "called": c[g.index[0]],
             "followed": t[g.index[0]], "months": len(g)}
            for _, g in key[wrong].groupby(run[wrong])]
    return (pd.DataFrame(rows, columns=["start", "end", "called", "followed", "months"])
            .sort_values(["months", "start"], ascending=[False, True]).head(k)
            .reset_index(drop=True))


def evaluate(train: pd.DataFrame, nowcast: pd.Series, reg: dict, names: list[str],
             first: str = FIRST) -> dict:
    """Everything the methodology page reports about the call's record."""
    from scripts.realign_positions import held_out, priors, real_time

    regimes = list(truth.QUADRANT.values())
    window = train.index[train.index >= pd.Period(first, "M")]
    labels = train["label"].reindex(window)
    prior = priors(reg, names)
    seg = train.loc[window]

    P, F = held_out(train, prior, reg, names)
    held = _calls(P, F, reg, window)
    Prt, Frt = real_time(train, prior, reg, names)
    rt = _calls(Prt, Frt, reg, window)
    base = ol.base_rate(train, regimes, first=first).reindex(window)
    signs = pd.Series([truth.QUADRANT[(g > 0, i > 0)] for g, i in
                       zip(seg["growth"], seg["inflation"])], index=window)
    methods = {
        "held_out": ol.grade(held, labels, seg, P.reindex(window)),
        "real_time": ol.grade(rt, labels, seg, Prt.reindex(window)),
        "continues": ol.grade(ol.persistence(nowcast, window), labels, seg),
        "base_rate": ol.grade(base.idxmax(axis=1), labels, seg, base),
        "signs": ol.grade(signs, labels, seg),
    }

    drop = {}
    for d in names:
        keep = [n for n in names if n != d]
        Pd, Fd = held_out(train, prior[keep], _without(reg, keep), keep)
        drop[d] = ol.score(_calls(Pd, Fd, _without(reg, keep), window), labels)["balanced"]
    gi = [n for n in names if n in ("growth", "inflation")]
    Pd, Fd = held_out(train, prior[gi], _without(reg, gi), gi)
    only_gi = ol.score(_calls(Pd, Fd, _without(reg, gi), window), labels)["balanced"]

    lead = P.reindex(window).max(axis=1)
    return {
        "window": (window[0], window[-1]), "months": int(labels.notna().sum()),
        "methods": methods,
        "by_regime": by_regime(held, labels, regimes),
        "axes": axes(held, labels),
        # The years behind the most confident calls, most months first.
        "confident_years": lead[lead > ODDS_BINS[-2]].index.year.value_counts(),
        "bootstrap": bootstrap(held, labels, regimes),
        "confusion": pd.crosstab(labels, held).reindex(index=regimes),
        "calibration": calibration(P.reindex(window), labels),
        "misses": longest_misses(held, labels),
        "without": drop, "growth_inflation_only": only_gi,
    }
