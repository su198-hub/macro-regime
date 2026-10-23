"""The three-year outlook: which regime the next three years look like.

The answer key is the three-year forward label in src/truth.py: for each month,
the regime the following 36 months turned out to be. Each regime's position on
the signposts is learned from it -- the average signpost scores across
every month whose next three years were that regime, which is what the run-up
to that regime looked like. Today's signposts are then compared with those
positions, nearest first.

The dashboard's archetypes in config/regimes.yml answer a different question:
what each regime looks like while it is happening. Applied to today's readings
they describe the present. They are used here only as the judgment fallback
when grading, and as the comparison the track record is set against.

Grading leaves out one episode at a time. An episode is an unbroken run of one
forward label. Its months are held out, and so is every month whose 36-month
outcome window overlaps any of theirs, because neighbouring months share most
of their answer. A regime with no episode left after that falls back to its
archetype, the judgment position, rather than to nothing.

Three ways of grading, reported side by side so overfitting is visible
rather than hidden (scripts/forecast_scorecard.py):
  - fitted: positions learned from every month, then graded on the same months.
    The most a method can explain, and an upper bound.
  - held out: held_out, below; each episode graded by positions learned without it.
  - real time: real_time, below; each month sees only the labels known by then.
Benchmarks: persistence (the latest confirmed regime continues) and the base
rate of regimes known so far. Neither looks at a signpost.

Choices made on held-out episodes of the six-driver framework, 1983-2023:
  - every signpost counts equally in the distance. The dashboard's salience
    weights did no better (30% vs 31% of months), and equal is one opinion fewer;
  - this month's readings, not a 3- or 6-month average (within a point either way);
  - temperature 0.75, the lowest held-out log loss of 0.2 to 3.0. The
    probabilities are deliberately flat: out of sample they beat an even 25%
    split only narrowly (log loss 1.32 against 1.39).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import truth

HORIZON = 36            # months a forward label describes
# A forward label is known only once the last GDP quarter in its window is
# published: about four months after the window ends.
LABEL_LAG = 4
START = "1968-01"       # first month with every signpost at least partly measured, and with an answer key
MIN_COVERAGE = 0.4      # share of a signpost's indicator weight that must be present
TEMPERATURE = 0.75


def monthly_drivers(drivers: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Signpost scores on a monthly PeriodIndex, keeping only months where every
    signpost is scored and carries at least MIN_COVERAGE of its weight."""
    d = drivers.copy()
    d.index = pd.DatetimeIndex(d.index).to_period("M")
    ok = d[names].notna().all(axis=1)
    for n in names:
        cov = f"{n}__coverage"
        if cov in d:
            ok &= d[cov] >= MIN_COVERAGE
    return d.loc[ok, names].astype(float)


def training_frame(scores: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Months with both signpost scores and a forward label, from START."""
    j = scores.join(forward["label"], how="inner")
    return j[j.index >= pd.Period(START, "M")]


def spells(labels: pd.Series) -> pd.DataFrame:
    """Unbroken runs of one label, oldest first. A gap in the months also ends a
    run, so two episodes either side of a held-out stretch stay two."""
    gap = np.zeros(len(labels), dtype=bool)
    if isinstance(labels.index, pd.PeriodIndex) and len(labels) > 1:
        gap[1:] = np.diff(labels.index.asi8) != 1
    run = ((labels != labels.shift()) | gap).cumsum()
    rows = [{"start": g.index[0], "end": g.index[-1], "label": g.iloc[0], "months": len(g)}
            for _, g in labels.groupby(run)]
    return pd.DataFrame(rows, columns=["start", "end", "label", "months"])


def positions(train: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Each regime's average signpost scores over its months (regimes x signposts)."""
    return train.groupby("label")[names].mean()


CORE_SHARE = 0.25       # the strongest quarter of each regime's months; see core_positions


def core_positions(frame: pd.DataFrame, outcome: pd.DataFrame, names: list[str],
                   share: float = CORE_SHARE) -> pd.DataFrame:
    """Each regime's average driver scores over its strongest months only.

    A month's strength is how far its growth and inflation gaps (`outcome`)
    sat inside its regime's quadrant: the smaller of the two distances past
    zero, so a month deep on one axis but borderline on the other does not
    count as strong. Keeps the top `share` of each regime's months. An
    average over all months sits near neutral; this is the regime at its
    most pronounced, closer in spirit to a judgment of the regime itself.
    """
    sign = {r: (1.0 if up_g else -1.0, 1.0 if up_i else -1.0)
            for (up_g, up_i), r in truth.QUADRANT.items()}
    gaps = outcome[["growth", "inflation"]].rename(columns={"growth": "_g", "inflation": "_i"})
    j = frame[[*names, "label"]].join(gaps, how="inner")
    j = j[j["label"].isin(sign)]
    sg = j["label"].map(lambda r: sign[r][0])
    si = j["label"].map(lambda r: sign[r][1])
    depth = np.minimum(j["_g"] * sg, j["_i"] * si)
    cut = depth.groupby(j["label"]).transform(lambda d: d.quantile(1 - share))
    return j[depth >= cut].groupby("label")[names].mean()


PRIOR_STRENGTH = 0.5    # episodes' worth of weight on the prior; see realigned_positions


def realigned_positions(train: pd.DataFrame, prior: pd.DataFrame, names: list[str],
                        strength: float = PRIOR_STRENGTH) -> pd.DataFrame:
    """Scenario positions for a regime call that describes the next three years.

    Each labeled regime's run-up position (positions, above) is shrunk toward
    a prior, a judgment of where the regime sits, weighting the data by its
    number of episodes E and the prior by `strength`:

        position = (E * learned + strength * prior) / (E + strength)

    The prior is first scaled to the learned positions' spread. Run-ups
    average many months and sit near the middle, about a third as far out as
    a judgment of the regime itself, so an unscaled prior pulls every position
    outward and the call with it. A scenario with no label of its own, such as
    Baseline, takes the average of all months as one episode. A labeled regime
    with no episode yet takes its scaled prior.
    """
    labeled = [r for r in prior.index if r in set(truth.QUADRANT.values())]
    learned = positions(train, names)
    seen = [r for r in labeled if r in learned.index]
    if seen:
        scale = (learned.loc[seen, names].abs().to_numpy().mean()
                 / max(prior.loc[seen, names].abs().to_numpy().mean(), 1e-9))
    else:
        scale = 1.0
    scaled = prior[names] * scale
    episodes = spells(train["label"])["label"].value_counts()
    out = scaled.copy()
    for r in prior.index:
        if r in labeled:
            e = float(episodes.get(r, 0))
            if e:
                out.loc[r] = (e * learned.loc[r, names] + strength * scaled.loc[r]) / (e + strength)
        elif len(train):
            out.loc[r] = (train[names].mean() + strength * scaled.loc[r]) / (1 + strength)
    return out


def support(train: pd.DataFrame) -> pd.DataFrame:
    """How many months and separate episodes stand behind each position."""
    sp = spells(train["label"])
    return pd.DataFrame({"months": train["label"].value_counts(),
                         "episodes": sp["label"].value_counts()}).fillna(0).astype(int)


def probabilities(scores: pd.DataFrame, pos: pd.DataFrame,
                  temperature: float = TEMPERATURE, weights: dict | None = None) -> pd.DataFrame:
    """Softmax of minus the distance to each position. Signposts count equally
    unless `weights` gives some a different weight on their squared distance,
    as the regime settings' salience does on the dashboard."""
    X = scores[pos.columns].to_numpy(dtype=float)
    P = pos.to_numpy(dtype=float)
    w = np.array([float((weights or {}).get(c, 1.0)) for c in pos.columns])
    dist = np.sqrt((((X[:, None, :] - P[None, :, :]) ** 2) * w).sum(axis=2))
    z = -dist / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return pd.DataFrame(e / e.sum(axis=1, keepdims=True), index=scores.index, columns=pos.index)


def purged(index: pd.PeriodIndex, start: pd.Period, end: pd.Period) -> np.ndarray:
    """Months whose outcome window overlaps that of any month from start to end.

    Month t's answer covers t to t+35, so it shares outcomes with the held-out
    months whenever t falls within 35 months either side of them.
    """
    lag = HORIZON - 1
    return (index >= start - lag) & (index <= end + lag)


def held_out(train: pd.DataFrame, judgment: pd.DataFrame, names: list[str],
             temperature: float = TEMPERATURE, weights: dict | None = None) -> pd.DataFrame:
    """Probabilities for every training month from positions learned without
    its episode or anything sharing its outcomes. Columns follow `judgment`."""
    regimes = list(judgment.index)
    out = []
    for sp in spells(train["label"]).itertuples():
        test = train.loc[sp.start:sp.end, names]
        rest = train[~purged(train.index, sp.start, sp.end)]
        pos = positions(rest, names).reindex(regimes)
        pos = pos.fillna(judgment[names])
        out.append(probabilities(test, pos, temperature, weights))
    return pd.concat(out).sort_index()[regimes]


def persistence(nowcast: pd.Series, index: pd.PeriodIndex, lag: int | None = None) -> pd.Series:
    """The baseline call: the next three years look like the latest confirmed
    regime, which in month t is the nowcast label for month t - lag. By default
    lag is how far the nowcast's window reaches past the month."""
    if lag is None:
        now = truth.METHODS["now"]
        lag = truth.months_after(now["years"], now["align"], now["measure"])
    return nowcast.shift(lag).reindex(index)


def score(calls: pd.Series, truth: pd.Series) -> dict:
    """Share of months called right, and the average of each regime's share,
    which is what chance (25%) and an always-the-same call are judged against."""
    t = truth.reindex(calls.index)
    keep = calls.notna() & t.notna()
    hit = calls[keep] == t[keep]
    by_regime = hit.groupby(t[keep]).mean()
    return {"months": float(hit.mean()), "balanced": float(by_regime.mean()),
            "by_regime": by_regime.to_dict(), "n": int(keep.sum())}


def real_time(train: pd.DataFrame, judgment: pd.DataFrame, names: list[str],
              temperature: float = TEMPERATURE, first: str = "1990-01",
              weights: dict | None = None) -> pd.DataFrame:
    """Probabilities each month from positions learned only from labels already
    known then. Month s's label describes s to s+35, so it is known from s+36:
    in month t the model learns from months up to t - HORIZON - LABEL_LAG. A regime not yet
    seen falls back to its judgment position, as in held_out."""
    regimes = list(judgment.index)
    rows = {}
    for t in train.index[train.index >= pd.Period(first, "M")]:
        known = train[train.index <= t - HORIZON - LABEL_LAG]
        if known.empty:
            continue
        pos = positions(known, names).reindex(regimes).fillna(judgment[names])
        rows[t] = probabilities(train.loc[[t], names], pos, temperature, weights).iloc[0]
    return pd.DataFrame(rows).T[regimes] if rows else pd.DataFrame(columns=regimes)


def base_rate(train: pd.DataFrame, regimes: list[str], first: str = "1990-01") -> pd.DataFrame:
    """The benchmark that knows nothing about the signposts: each month, the mix
    of regimes among labels already known, with one pseudo-month for each so no
    regime gets zero. No hindsight: month t sees labels up to t - HORIZON - LABEL_LAG."""
    rows = {}
    for t in train.index[train.index >= pd.Period(first, "M")]:
        known = train.loc[train.index <= t - HORIZON - LABEL_LAG, "label"]
        counts = known.value_counts().reindex(regimes, fill_value=0) + 1.0
        rows[t] = counts / counts.sum()
    return pd.DataFrame(rows).T[regimes] if rows else pd.DataFrame(columns=regimes)


def log_loss(probs: pd.DataFrame, labels: pd.Series, floor: float = 1e-3) -> float:
    """Average of minus the log probability given to what happened. An even
    split over four regimes scores log 4 = 1.39; lower is better. Probability
    on a scenario the answer key has no label for, such as Baseline, counts as
    probability not given to the truth."""
    t = labels.reindex(probs.index)
    keep = t.notna()
    p = [probs.at[i, t[i]] if t[i] in probs.columns else 0.0 for i in probs.index[keep]]
    return float(-np.log(np.clip(p, floor, 1.0)).mean())


def grade(calls: pd.Series, labels: pd.Series, train: pd.DataFrame,
          probs: pd.DataFrame | None = None) -> dict:
    """Everything the scorecard reports for one method, on the months it called."""
    out = score(calls, labels)
    ep = by_episode(train.loc[calls.dropna().index], {"m": calls})
    hl = ep[ep["label"] == "hard_landing"]
    out["hard_landing_runups"] = int(len(hl))
    out["hard_landing_caught"] = int((hl["m"] >= 0.5).sum())
    out["log_loss"] = log_loss(probs, labels) if probs is not None else np.nan
    out["first"], out["last"] = str(calls.dropna().index.min()), str(calls.dropna().index.max())
    return out


def by_episode(train: pd.DataFrame, calls: dict[str, pd.Series]) -> pd.DataFrame:
    """For each episode, the share of its months each method called right."""
    rows = []
    for sp in spells(train["label"]).itertuples():
        row = {"start": sp.start, "end": sp.end, "label": sp.label, "months": sp.months}
        for name, c in calls.items():
            w = c.loc[sp.start:sp.end].dropna()
            row[name] = float((w == sp.label).mean()) if len(w) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
