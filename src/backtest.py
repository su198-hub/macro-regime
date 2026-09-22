"""Ex-post scoring: what regime the economy was actually in, and who called it.

There is no official record of regimes, so the benchmark is built from what
happened, from data the store already holds:

  hard landing   an NBER recession under way anywhere in the window
  otherwise      growth held      the unemployment gap (UNRATE - NROU) rose by
                                  no more than `growth_cut` across the window
                 inflation high   core PCE averaged above `inflation_cut`
                 held & low  -> goldilocks
                 held & high -> high growth, high inflation
                 weak & high -> stagflation
                 weak & low  -> unclassified: slow growth without a recession,
                                which no archetype describes, so a model that
                                declines to call it is right

The default window runs six months either side of the month, which is the fair
test of a nowcast: it asks what the economy was doing around the month, not
what it did next. `centred=False` scores against the following twelve months.

Both models are scored on today's revised data, so the absolute accuracy
flatters them; the comparison between them is fair because it is like for like.
"""

from __future__ import annotations

import pandas as pd

# NBER business-cycle peaks and troughs, as monthly (peak, trough). Final and
# never revised once announced; add the next one when the committee dates it.
NBER = [("1969-12", "1970-11"), ("1973-11", "1975-03"), ("1980-01", "1980-07"),
        ("1981-07", "1982-11"), ("1990-07", "1991-03"), ("2001-03", "2001-11"),
        ("2007-12", "2009-06"), ("2020-02", "2020-04")]

UNCLASSIFIED = "unclassified"


def recession_months(index: pd.PeriodIndex) -> pd.Series:
    flag = pd.Series(False, index=index)
    for peak, trough in NBER:
        flag.loc[pd.Period(peak, "M"):pd.Period(trough, "M")] = True
    return flag


def realised(wide: pd.DataFrame, centred: bool = True, inflation_cut: float = 2.5,
             growth_cut: float = 0.1) -> pd.Series:
    """The regime each month actually turned out to be in, by the rules above.

    `wide` needs UNRATE, NROU and PCEPILFE on a date index. Returns a series on
    a monthly PeriodIndex, covering only months with a complete window.
    """
    m = wide[["UNRATE", "NROU", "PCEPILFE"]].resample("ME").last().ffill()
    m.index = m.index.to_period("M")
    gap = m["UNRATE"] - m["NROU"]
    core = m["PCEPILFE"].pct_change(12) * 100
    rec = recession_months(m.index)
    out = {}
    months = list(m.index)
    for k, t in enumerate(months):
        # Twelve months either way: t-5..t+6 centred, t+1..t+12 forward.
        lo, hi = (k - 5, k + 6) if centred else (k + 1, k + 12)
        if lo < 0 or hi >= len(months):
            continue
        window = months[lo: hi + 1]
        start = window[0] if centred else t
        if rec.loc[window].any() or rec.loc[t]:
            out[t] = "hard_landing"
            continue
        held = (gap.loc[window[-1]] - gap.loc[start]) <= growth_cut
        high = core.loc[window].mean() > inflation_cut
        out[t] = {(True, False): "goldilocks", (True, True): "high_growth_high_inflation",
                  (False, True): "stagflation", (False, False): UNCLASSIFIED}[(held, high)]
    return pd.Series(out, dtype=object)


def _as_months(calls: pd.Series) -> pd.Series:
    c = calls.copy()
    if not isinstance(c.index, pd.PeriodIndex):
        c.index = pd.DatetimeIndex(c.index).to_period("M")
    # Transitional means the model saw no clear leader: score it as declining
    # to call, the same as unclassified.
    return c.replace({"transitional": UNCLASSIFIED})


def scorecard(calls: dict[str, pd.Series], truth: pd.Series,
              since: str | None = None) -> pd.DataFrame:
    """One row per measure, one column per model, over months all models scored."""
    series = {k: _as_months(v) for k, v in calls.items()}
    common = truth.index
    for c in series.values():
        common = common.intersection(c.dropna().index)
    if since:
        common = common[common >= pd.Period(since, "M")]
    t = truth.loc[common]
    hl_true = t == "hard_landing"
    # A costly error is a recession called a good regime, or a hard landing
    # called in Goldilocks — the two mistakes that would move a portfolio.
    good_true = t == "goldilocks"
    rows = {}
    for name, c in series.items():
        c = c.loc[common]
        hl_call = c == "hard_landing"
        good_call = c.isin(["goldilocks", "high_growth_high_inflation"])
        rows[name] = {
            "Months called correctly": (c == t).mean(),
            "Hard-landing calls, months": int(hl_call.sum()),
            "Hard-landing calls that were right": (hl_call & hl_true).sum() / max(hl_call.sum(), 1),
            "Recession months caught": (hl_call & hl_true).sum() / max(hl_true.sum(), 1),
            "Costly errors, months": int((good_call & hl_true).sum() + (hl_call & good_true).sum()),
        }
    out = pd.DataFrame(rows)
    out.attrs["months"] = len(common)
    out.attrs["first"], out.attrs["last"] = (common.min(), common.max()) if len(common) else (None, None)
    shares = t.value_counts(normalize=True)
    out.attrs["naive"] = (shares.index[0], float(shares.iloc[0])) if len(shares) else (None, float("nan"))
    return out
