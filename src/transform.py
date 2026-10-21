"""Transforms and normalisation.

Pure functions on pandas Series. No config, no IO, no state. Everything here
is unit-testable and that matters, because a silent sign error in a transform
produces a dashboard that is confidently wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLIP = 3.0  # normalised scores clip here before driver aggregation


# ---------- transforms ----------

def level(s: pd.Series) -> pd.Series:
    return s


def yoy_pct(s: pd.Series) -> pd.Series:
    periods = _periods_per_year(s)
    return s.pct_change(periods) * 100


def pct_change_3m_ann(s: pd.Series) -> pd.Series:
    ppy = _periods_per_year(s)
    n = max(1, round(ppy / 4))
    return ((s / s.shift(n)) ** (ppy / n) - 1) * 100


def diff_12m(s: pd.Series) -> pd.Series:
    return s - s.shift(_periods_per_year(s))


def sum_12m(s: pd.Series) -> pd.Series:
    """Rolling one-year sum of a flow, such as a monthly budget deficit.

    Needs a full year of periods, so a partial year never passes for an annual
    total. Summing a year also removes the seasonal pattern in unadjusted flows.
    """
    n = _periods_per_year(s)
    return s.rolling(n, min_periods=n).sum()


def pct_change_5y_ann(s: pd.Series) -> pd.Series:
    """Annualised growth over five years: a trend rate rather than a cyclical one.

    Where a year-on-year rate would carry the recession that happens to sit in
    the window, five years annualised carries the trend. Use it where the
    question is about the economy's growth rate rather than this year's: debt
    dynamics turn on the rate the debt compounds against over a decade, not on
    the quarter GDP collapsed.
    """
    n = 5 * _periods_per_year(s)
    return ((s / s.shift(n)) ** (1 / 5) - 1) * 100


def dev_5y_pct(s: pd.Series) -> pd.Series:
    """Percent deviation from the series' own trailing five-year average.

    For stocks that have no meaningful absolute level because they grow with
    the economy — oil inventories being the case in hand, where the question is
    never "how many barrels" but "more or fewer than usual". Five years is the
    convention the EIA publishes against, long enough to average out a cycle,
    short enough to follow structural growth in storage.

    Needs three years of history before it reports, so a new series does not
    read as a huge deviation from its own first months.
    """
    n = 5 * _periods_per_year(s)
    base = s.rolling(n, min_periods=int(n * 0.6)).mean()
    return (s / base - 1.0) * 100


def diff_36m_ann(s: pd.Series) -> pd.Series:
    """Change over three years, per year.

    For stances that are set over a policy cycle rather than a calendar year.
    QE and QT run for years, and a budget takes a year or two to follow
    through; a twelve-month window turns one sustained programme into a run of
    on-and-off pulses. Annualised so its scale reads like the one-year change.
    """
    n = 3 * _periods_per_year(s)
    return (s - s.shift(n)) / 3


def _inflation(s: pd.Series) -> pd.Series:
    return s.pct_change(_periods_per_year(s)) * 100


def off_target_share_36m(s: pd.Series) -> pd.Series:
    """Share of the last three years inflation spent outside 1.5-2.5%, in %.

    Takes a price index and works out the inflation itself, so the indicator
    reads straight off the core PCE series already in the set. A track record:
    a central bank that has missed for three years is believed less than one
    that has not, whatever its stance today. Counts misses in both directions,
    because an undershoot the bank could not fix is also a credibility cost.
    """
    infl = _inflation(s)
    n = 3 * _periods_per_year(s)
    off = ((infl < 1.5) | (infl > 2.5)).astype(float).where(infl.notna())
    return off.rolling(n, min_periods=n).mean() * 100


def vol_36m(s: pd.Series) -> pd.Series:
    """Standard deviation of inflation over the last three years, in points.

    Distinct from the miss rate: inflation through the 2010s was persistently
    below target but very steady, a credible undershoot, where 2021-23 was
    both off target and violent. One statistic cannot tell those apart.
    """
    infl = _inflation(s)
    n = 3 * _periods_per_year(s)
    return infl.rolling(n, min_periods=n).std()


def pct_change_3y_ann(s: pd.Series) -> pd.Series:
    """Annualized growth over three years: the pace over a policy cycle, as for
    revenue against spending, or nominal GDP in r minus g."""
    n = 3 * _periods_per_year(s)
    return ((s / s.shift(n)) ** (1 / 3) - 1) * 100


def mean_5y(s: pd.Series) -> pd.Series:
    """Trailing five-year average of a rate, such as quarterly productivity
    growth: the average pace rather than the latest, noisy reading. Reports
    once four of the five years are in."""
    n = 5 * _periods_per_year(s)
    return s.rolling(n, min_periods=int(n * 0.8)).mean()


def mean_36m(s: pd.Series) -> pd.Series:
    """Trailing three-year average, for a stance judged on its track record
    rather than this month's reading: the Taylor gap averaged over a cycle."""
    n = 3 * _periods_per_year(s)
    return s.rolling(n, min_periods=n).mean()


def dev_10y(s: pd.Series) -> pd.Series:
    """Points above the series' own trailing ten-year average, for a share that
    drifts over decades, such as business investment as a share of GDP. Needs
    six years of history before it reports."""
    n = 10 * _periods_per_year(s)
    return s - s.rolling(n, min_periods=int(n * 0.6)).mean()


def mean_12m(s: pd.Series) -> pd.Series:
    """Trailing one-year average, for a news-based index that spikes month to
    month, such as policy uncertainty."""
    n = _periods_per_year(s)
    return s.rolling(n, min_periods=n).mean()


def diff_5y(s: pd.Series) -> pd.Series:
    """Change over five years, matched by date rather than by position, so a
    gap in the series never stretches the window. An annual or irregular
    series (one published every five years and then yearly) is first put on a
    yearly footing, interpolating only between points both published, so five
    steps are always five years."""
    s = s.dropna()
    if s.empty:
        return s
    days = pd.Series(s.index).diff().dt.days.median() if len(s) > 1 else 365
    if days > 135:
        s = s.groupby(s.index.year).last()
        s = s.reindex(range(s.index.min(), s.index.max() + 1)).interpolate(limit_area="inside")
        s.index = pd.to_datetime([f"{y}-12-31" for y in s.index])
        return s - s.shift(5)
    back = s.reindex(s.index - pd.DateOffset(years=5))
    return s - pd.Series(back.to_numpy(), index=s.index)


def target_distance_change_12m(s: pd.Series) -> pd.Series:
    """Change over a year in how far inflation sits from 2%, in points.

    Takes a price index. Positive when inflation is moving away from target,
    negative when it is converging: the trajectory half of credibility, which a
    miss rate alone cannot see -- a bank three points off and closing is in a
    different position from one three points off and drifting."""
    dist = (_inflation(s) - 2.0).abs()
    return dist - dist.shift(_periods_per_year(s))


TRANSFORMS = {
    "level": level,
    "pct_change_3y_ann": pct_change_3y_ann,
    "mean_5y": mean_5y,
    "mean_36m": mean_36m,
    "dev_10y": dev_10y,
    "target_distance_change_12m": target_distance_change_12m,
    "mean_12m": mean_12m,
    "diff_5y": diff_5y,
    "yoy_pct": yoy_pct,
    "pct_change_3m_ann": pct_change_3m_ann,
    "diff_12m": diff_12m,
    "sum_12m": sum_12m,
    "dev_5y_pct": dev_5y_pct,
    "pct_change_5y_ann": pct_change_5y_ann,
    "diff_36m_ann": diff_36m_ann,
    "off_target_share_36m": off_target_share_36m,
    "vol_36m": vol_36m,
}


def _periods_per_year(s: pd.Series) -> int:
    """Infer frequency from the index rather than trusting config."""
    if len(s.dropna()) < 3:
        return 12
    days = pd.Series(s.dropna().index).diff().dt.days.median()
    if days <= 4:
        return 252
    if days <= 10:
        return 52
    if days <= 45:
        return 12
    if days <= 135:
        return 4
    return 1


# ---------- normalisation ----------

def normalise(s: pd.Series, spec: dict) -> pd.Series:
    method = spec.get("method", "zscore")
    if method == "gap":
        centre = float(spec["center"])
        scale = float(spec["scale"])
        if scale == 0:
            raise ValueError("normalise: scale cannot be zero")
        out = (s - centre) / scale
    elif method == "zscore":
        window = int(spec.get("window", 240))
        mean = s.rolling(window, min_periods=max(24, window // 4)).mean()
        std = s.rolling(window, min_periods=max(24, window // 4)).std()
        out = (s - mean) / std.replace(0, np.nan)
    else:
        raise ValueError(f"unknown normalise method: {method}")
    return out.clip(-CLIP, CLIP)


def to_monthly(s: pd.Series, end: pd.Timestamp | None = None,
               carry_periods: int = 0) -> pd.Series:
    """Resample to month-end, forward-filling lower-frequency series.

    Forward fill is deliberate: a quarterly capex number stays the best
    available estimate until the next one lands, which is how you would
    actually read it. It does mean quarterly drivers move in steps.

    Without `end`, the fill stops at the series' own last observation, so a
    quarterly value dated April drops out in May even though the next release
    is months away. Pass `end` (the latest month any input reaches) and
    `carry_periods` to hold the last value up to that many of the series' own
    periods: two quarters for a quarterly series, two years for an annual one.
    Monthly and faster series are never extended.
    """
    monthly = s.resample("ME").last().ffill()
    ppy = _periods_per_year(s)
    if end is None or carry_periods <= 0 or ppy >= 12 or monthly.empty:
        return monthly
    end = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
    if end <= monthly.index[-1]:
        return monthly
    months = carry_periods * (12 // ppy)
    idx = pd.date_range(monthly.index[0], end, freq="ME")
    return monthly.reindex(idx).ffill(limit=months)
