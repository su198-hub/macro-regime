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


TRANSFORMS = {
    "level": level,
    "yoy_pct": yoy_pct,
    "pct_change_3m_ann": pct_change_3m_ann,
    "diff_12m": diff_12m,
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


def to_monthly(s: pd.Series) -> pd.Series:
    """Resample to month-end, forward-filling lower-frequency series.

    Forward fill is deliberate: a quarterly capex number stays the best
    available estimate until the next one lands, which is how you would
    actually read it. It does mean quarterly drivers move in steps.
    """
    return s.resample("ME").last().ffill()
