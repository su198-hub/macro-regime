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


TRANSFORMS = {
    "level": level,
    "yoy_pct": yoy_pct,
    "pct_change_3m_ann": pct_change_3m_ann,
    "diff_12m": diff_12m,
    "sum_12m": sum_12m,
    "dev_5y_pct": dev_5y_pct,
    "pct_change_5y_ann": pct_change_5y_ann,
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
