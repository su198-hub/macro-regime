"""The regime each month was actually in since 1968, judged with hindsight.

Two numbers per month, both in percentage points:

  growth      real GDP growth less CBO potential growth, over the four quarters
              centred on the month. Above zero, the economy grew faster than
              its capacity.
  inflation   core PCE inflation over the twelve months centred on the month,
              less expected inflation: the SPF ten-year CPI median, less the
              usual gap between CPI and PCE inflation. Before 1991 the Philly
              Fed fills that survey series from Blue Chip and Livingston, and
              before October 1979 it is carried back by the Fed's FRB/US
              measure (PTR, the Kozicki-Tinsley estimate before 1979).

Each axis flips only when it clears its band on the other side of zero, so a
wobble around zero keeps the last call and is marked weak. A quadrant must last
three months; shorter spells take the label before them.

Quadrants use the model's names: up and low is goldilocks, up and high is
high_growth_high_inflation, down and high is stagflation, down and low is
hard_landing. Unlike backtest.realised, a recession is not forced to
hard_landing: 1970, 1974, 1980 and 1990 count as stagflation because inflation ran hot.

Everything is on today's revised data, which is the point: this is what
happened, not what anyone could see at the time. At the end of the sample the
windows run to the latest print, and those months are marked provisional. A
provisional month cannot start a new regime: it keeps the last confirmed one,
and what its own readings point to is kept alongside as provisional_reading.
"""

from __future__ import annotations

import pandas as pd

from .backtest import NBER, recession_months

CODES = {
    "gdp": "usnaac0169",            # real GDP, chained, SAAR
    "potential": "usfcst1985",      # CBO real potential GDP, quarterly
    "core_pce": "uspric0006",       # PCE excluding food and energy
    "pce": "uspric0001",            # PCE, total
    "cpi": "uspric2156",            # CPI all items, SA
    "spf_cpi10": "usfcst0062",      # SPF median CPI over ten years, from 1979
    "ptr": "frbus:PTR",             # FRB/US expected PCE inflation, ten years, from 1968
}

# PTR, the Fed's FRB/US measure of long-run expected inflation, carries the
# survey back before October 1979: surveys from 1979, the Kozicki-Tinsley
# estimate before. Over their first five overlapping years (1979Q4-1984Q4) it
# ran 0.09 points below the survey series, so it is put on the survey's basis
# by adding that back, then treated exactly as the survey is.
PTR_TO_SPF = 0.09

# Names the inputs are stored under, so the hosted app can rebuild the labels
# from the published snapshot. Prefixed so they never collide with indicators.
STORE_IDS = {name: f"TR_{name.upper()}" for name in CODES}

TITLES = {
    "gdp": ("Real GDP", "USD bn, SAAR", "Quarterly"),
    "potential": ("CBO real potential GDP", "USD bn, SAAR", "Quarterly"),
    "core_pce": ("PCE price index excluding food and energy", "index", "Monthly"),
    "pce": ("PCE price index", "index", "Monthly"),
    "cpi": ("Consumer price index, all items", "index", "Monthly"),
    "spf_cpi10": ("SPF median expected CPI inflation, next 10 years", "%", "Quarterly"),
    "ptr": ("FRB/US expected PCE inflation, next 10 years (PTR)", "%", "Quarterly"),
}

QUADRANT = {(True, False): "goldilocks", (True, True): "high_growth_high_inflation",
            (False, True): "stagflation", (False, False): "hard_landing"}

GROWTH_BAND = 0.5       # percentage points, for a one-year window
INFLATION_BAND = 0.5    # percentage points, for a one-year window


def band(years: float, one_year: float = GROWTH_BAND) -> float:
    """The neutral band for a window of `years`. An average over a longer
    window moves less, so a fixed band would rarely be cleared and the first
    reading would stick for a decade; the band shrinks with the square root."""
    return one_year / years ** 0.5
MIN_MONTHS = 3

# What the record says each stretch was, to check the labels against. The
# expected label is the textbook reading; the check reports the share agreeing.
EPISODES = [
    ("1969-70 recession", "1969-12", "1970-11", "stagflation"),
    ("Oil shock recession", "1973-11", "1975-03", "stagflation"),
    ("Volcker recessions", "1980-01", "1982-11", "stagflation"),
    ("Recovery and disinflation", "1983-06", "1986-12", "goldilocks"),
    ("Late-80s boom", "1988-01", "1989-06", "high_growth_high_inflation"),
    ("1990-91 recession", "1990-08", "1991-03", "stagflation"),
    ("Late-90s expansion", "1996-01", "1999-12", "goldilocks"),
    ("2001 recession", "2001-03", "2001-11", "hard_landing"),
    ("Global financial crisis", "2008-06", "2009-06", "hard_landing"),
    ("Covid shutdown", "2020-03", "2020-05", "hard_landing"),
    ("Reopening boom", "2021-03", "2021-12", "high_growth_high_inflation"),
    ("2022 squeeze", "2022-03", "2022-12", "stagflation"),
    ("2023-24 soft landing", "2023-07", "2024-12", "goldilocks"),
]


def fetch(api) -> pd.DataFrame:
    """Every input from Macrobond as one monthly frame, quarterly series on
    the first month of their quarter."""
    out = {}
    for name, code in CODES.items():
        if code.startswith("frbus:"):
            from .sources.frbus import FrbusSource
            src = FrbusSource()
            try:
                out[name] = src.series(code.split(":", 1)[1])
            finally:
                src.close()
            continue
        s = api.get_one_series(code)
        idx = pd.to_datetime(pd.Series(s.dates)).dt.tz_localize(None).dt.to_period("M")
        out[name] = pd.Series(pd.to_numeric(pd.Series(s.values), errors="coerce").values, index=idx)
    return pd.DataFrame(out).sort_index()


def to_observations(m: pd.DataFrame, vintage) -> pd.DataFrame:
    """The inputs as store rows, one vintage: today's revised history."""
    rows = []
    for name, sid in STORE_IDS.items():
        s = m[name].dropna()
        rows.append(pd.DataFrame({"series_id": sid, "observation_date": s.index.to_timestamp().date,
                                  "vintage_date": vintage, "value": s.values}))
    return pd.concat(rows, ignore_index=True)


def from_store(wide: pd.DataFrame) -> pd.DataFrame:
    """The store's wide frame (dates x TR_ series) back into fetch()'s shape."""
    back = {sid: name for name, sid in STORE_IDS.items()}
    w = wide[[c for c in wide.columns if c in back]].rename(columns=back).astype(float)
    w.index = pd.DatetimeIndex(w.index).to_period("M")
    return w.groupby(level=0).last().reindex(columns=list(CODES)).sort_index()


# The labels the /history page offers. The nowcast is the regime at the time.
# The forward labels are the regime of the next three or five years, with
# perfect foresight, for judging what the signposts said about the years ahead;
# they need the whole window, so they stop that long before the latest data.
#
# Measure: "endpoints" is the change from the start of the window to its end.
# "averaged" is the mean of every overlapping small-window change inside the
# window, so one unusual month at either end counts once among many. (Back-to-
# back changes would not help: chained together they collapse to the start and
# end points again.) All three labels average, and AVERAGED gives the small
# and big windows in months.
#
# The nowcast dates each 12-month change by the month it ends in, the way a
# year-over-year rate is quoted, and averages the 7 ending from 3 months before
# the month to 3 after. Together they cover 18 months, the 15 before the month
# and the 3 after, so only the latest 3 months wait for data. Dating a change
# by its middle month instead (the 7 changes inside the 18 months centred on
# the month) matches history better -- 88% of episode months against 79%, 98%
# of recession months against 95% -- because a change ending in a month mostly
# describes the year before it, so these labels run about six months behind
# events. The 2022 squeeze shows it most: stagflation from August 2022.
#
# Every label holds a regime at least a year. Of the year-over-year versions
# tested (1 to 25 changes, minimum regime of 3 to 18 months, bands of 0.30 to
# 0.50; scripts/window_sweep.py), 7 changes with a 12-month minimum and the
# 0.41 band is the steadiest that still matches 10 of 11 episodes. That rests
# on two episodes at exactly half their months, the late-80s boom and the 2022
# squeeze: a 13-month minimum, or a band 0.05 either way, loses the squeeze.
#
# For the forward labels, scored by how often they match the most common
# nowcast regime over the years they describe, 12-month stretches beat 6- and
# 24-month ones over 3 years, 24-month stretches beat shorter ones over 5
# years, and a one-year minimum flips less than a 3-month one.
MEASURES = ("endpoints", "averaged")
AVERAGED = {(1.5, "ending"): (12, 18), (3, "forward"): (12, 36), (5, "forward"): (24, 60)}
METHODS = {
    "now": {"years": 1.5, "align": "ending", "measure": "averaged", "min_months": 12},
    "next3": {"years": 3, "align": "forward", "measure": "averaged", "min_months": 12},
    "next5": {"years": 5, "align": "forward", "measure": "averaged", "min_months": 12},
}
DEFAULT = METHODS["now"]


def stretches(years: float, align: str) -> tuple[int, int]:
    """How many small changes an averaged window holds, and how long each is, in months."""
    small, big = _small_big(years, align)
    return big - small + 1, small


def _centred(level: pd.Series, half: int, per_year: int) -> pd.Series:
    """% a year from `half` periods before to `half` after each period. Near
    the end the window runs to the latest print instead."""
    level = level.dropna()
    out = ((level.shift(-half) / level.shift(half)) ** (per_year / (2 * half)) - 1) * 100
    last = level.index[-1]
    for t in level.index[-half:]:
        n = (last - (t - half)).n
        out[t] = ((level[last] / level[t - half]) ** (per_year / n) - 1) * 100
    return out


def _windowed(level: pd.Series, years: float, per_year: int, align: str) -> pd.Series:
    """% a year over `years`, centred on each period or starting in it
    (forward)."""
    n = int(round(years * per_year))
    if align == "centered":
        return _centred(level, n // 2, per_year)
    level = level.dropna()
    return ((level.shift(-n) / level) ** (per_year / n) - 1) * 100


def _averaged(level: pd.Series, small: int, big: int, align: str, per_year: int) -> pd.Series:
    """Mean of the annualized `small`-period changes whose windows sit inside
    the `big`-period window centred on, or starting in, each period. Near the
    end a centred window averages the changes it has, or the latest one.

    "ending" dates each change by the period it ends in, as a year-over-year
    rate is quoted, and averages the changes ending in the periods balanced
    around each one: with 7 of them, from 3 periods before to 3 after. The
    data they cover still spans `big` periods, ending `(k - 1) / 2` after."""
    level = level.dropna()
    g = ((level / level.shift(small)) ** (per_year / small) - 1) * 100   # change ending at each period
    k = big - small + 1
    if align == "forward":
        return g.rolling(k, min_periods=k).mean().shift(-big)
    if align == "ending":
        if k % 2 == 0:
            raise ValueError(f"an ending window needs an odd number of changes, got {k}")
        after = (k - 1) // 2
        out = g.rolling(k, min_periods=k).mean().shift(-after)
        # Near the end, average the changes that have ended so far.
        for t in (g.index[-after:] if after else []):
            out[t] = g.loc[t - after:].mean()
        return out
    half = big // 2
    out = g.rolling(k, min_periods=k).mean().shift(-half)
    last = g.index[-1]
    for t in g.index[-half:]:
        # Right at the end no change sits wholly inside the window yet; use the latest.
        out[t] = g.loc[min(t - half + small, last):].mean()
    return out


def _small_big(years: float, align: str) -> tuple[int, int]:
    """Small and big windows, in months, for the averaged measure."""
    return AVERAGED.get((years, align), (12, int(round(12 * years))))


def window_months(years: float, align: str, measure: str = "endpoints") -> int:
    """How many months a label's window spans."""
    return _small_big(years, align)[1] if measure == "averaged" else int(round(12 * years))


def months_after(years: float, align: str, measure: str = "endpoints") -> int:
    """How many months after the labeled month its window reaches, which is
    also how many of the latest months wait for later data."""
    if align == "forward":
        return window_months(years, align, measure)
    if align == "ending":
        n, _ = stretches(years, align)
        return (n - 1) // 2
    return window_months(years, align, measure) // 2


def _measure(level: pd.Series, years: float, per_year: int, align: str, measure: str) -> pd.Series:
    if measure == "endpoints":
        if align == "ending":
            raise ValueError("ending dating applies to the averaged measure only")
        return _windowed(level, years, per_year, align)
    small, big = _small_big(years, align)
    return _averaged(level, small * per_year // 12, big * per_year // 12, align, per_year)


def growth_parts(gdp: pd.Series, potential: pd.Series, years: float = 1,
                 align: str = "centered", measure: str = "endpoints") -> pd.DataFrame:
    """Real GDP and CBO potential growth, % a year over the window around (or
    starting in) each quarter, placed on the quarter's middle month and
    interpolated between."""
    def q(s):
        s = s.dropna()
        return s.groupby(s.index.asfreq("Q")).mean()

    g = q(gdp)
    p = q(potential).reindex(g.index)
    parts = pd.DataFrame({"gdp_growth": _measure(g, years, 4, align, measure),
                          "potential_growth": _measure(p, years, 4, align, measure)}).dropna()
    parts.index = parts.index.asfreq("M", how="start") + 1
    full = pd.period_range(parts.index[0] - 1, parts.index[-1] + 1, freq="M")
    out = parts.reindex(full).interpolate(limit_area="inside").ffill().bfill()
    # A window reaching past the month is incomplete for the last quarters it
    # reaches into: half the window centred, (changes - 1) / 2 when ending.
    q_after = months_after(years, align, measure) // 3 if align != "forward" else 0
    out["provisional"] = (out.index >= g.index[-q_after].asfreq("M", how="start")) if q_after else False
    return out


def axes(m: pd.DataFrame, years: float = 1, align: str = "centered",
         measure: str = "endpoints") -> pd.DataFrame:
    """Growth and inflation gap per month, with every input in its own units so
    a reader can check them."""
    parts = growth_parts(m["gdp"], m["potential"], years, align, measure)
    core = m["core_pce"].dropna()
    inflation = _measure(core, years, 12, align, measure)
    # CPI runs above PCE; take the gap over the ten years around each month.
    wedge = (m["cpi"].pct_change(12, fill_method=None) - m["pce"].pct_change(12, fill_method=None)) * 100
    wedge = wedge.rolling(120, center=True, min_periods=60).mean().ffill().bfill()
    spf = m["spf_cpi10"]
    if "ptr" in m and m["ptr"].notna().any() and spf.notna().any():
        early = (m["ptr"] + PTR_TO_SPF).dropna()
        spf = spf.combine_first(early[early.index < spf.first_valid_index()])
    spf = spf.interpolate(limit_area="inside").ffill()
    expected = spf - wedge
    n = window_months(years, align, measure)
    after = months_after(years, align, measure)
    # Compare with the benchmark over the same window as inflation.
    if align == "forward":
        expected = expected.rolling(n, min_periods=1).mean().shift(-(n - 1))
    elif align == "ending":
        # The n months ending `after` past the month; near the end, the
        # window ending at the latest reading.
        expected = expected.rolling(n, min_periods=1).mean().shift(-after).ffill()
    elif n > 12:
        expected = expected.rolling(n, center=True, min_periods=1).mean()
    out = pd.DataFrame({
        "gdp_growth": parts["gdp_growth"], "potential_growth": parts["potential_growth"],
        "core_pce": inflation, "spf_cpi10": spf, "wedge": wedge, "expected": expected})
    out["growth"] = out["gdp_growth"] - out["potential_growth"]
    out["inflation"] = out["core_pce"] - out["expected"]
    out = out.dropna(subset=["growth", "inflation"])
    late = (out.index > core.index[-after - 1]) if align != "forward" and after else False
    out["provisional"] = parts["provisional"].reindex(out.index, fill_value=False) | late
    return out


def _hysteresis(x: pd.Series, band: float) -> pd.DataFrame:
    """Up or down per month, flipping only when x clears `band` the other way."""
    state, rows = None, {}
    for t, v in x.dropna().items():
        clear = abs(v) >= band
        if clear or state is None:
            state = v >= 0
        rows[t] = (state, clear)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["up", "clear"])


def _persist(labels: pd.Series, n: int) -> pd.Series:
    """Spells shorter than `n` months take the label before them."""
    out = labels.copy()
    run = (out != out.shift()).cumsum()
    lengths = out.groupby(run).transform("size")
    short = (lengths < n) & (run != run.iloc[0])
    out[short] = None
    return out.ffill()


def primary(ax: pd.DataFrame, growth_band: float = GROWTH_BAND,
            inflation_band: float = INFLATION_BAND, min_months: int = MIN_MONTHS) -> pd.DataFrame:
    g = _hysteresis(ax["growth"], growth_band)
    i = _hysteresis(ax["inflation"], inflation_band)
    both = g.join(i, lsuffix="_g", rsuffix="_i", how="inner")
    raw = pd.Series([QUADRANT[(a, b)] for a, b in zip(both["up_g"], both["up_i"])], index=both.index)
    return pd.DataFrame({
        "label": _persist(raw, min_months),
        "raw": raw,
        "strength": (both["clear_g"] & both["clear_i"]).map({True: "clear", False: "weak"}),
    })


def build(m: pd.DataFrame, start: str = "1968-01", years: float = 1, align: str = "centered",
          measure: str = "endpoints", min_months: int = MIN_MONTHS) -> pd.DataFrame:
    """One row per month from `start`: the inputs, both gaps and the label."""
    ax = axes(m, years, align, measure)
    span = window_months(years, align, measure) / 12
    out = ax.join(primary(ax, band(span, GROWTH_BAND), band(span, INFLATION_BAND), min_months), how="inner")
    out = out[out.index >= pd.Period(start, "M")]
    out["recession"] = recession_months(out.index)
    # Half-complete windows are not enough to call a turn: one weak quarter at
    # the edge once flipped the latest months to stagflation and back.
    out["provisional_reading"] = out["label"]
    confirmed = out.loc[~out["provisional"], "label"]
    if len(confirmed):
        out.loc[out["provisional"], "label"] = confirmed.iloc[-1]
    return out


def what_followed(fwd: pd.DataFrame, now: pd.Series, years: int) -> pd.DataFrame:
    """For each month with a forward label, the most common current-regime
    label over the window it describes, and whether the two agree."""
    n = 12 * years
    rows = {}
    for t, label in fwd["label"].items():
        w = now.loc[t:t + (n - 1)]
        if len(w) < n:
            continue
        mix = w.value_counts(normalize=True)
        rows[t] = {"label": label, "most_common": mix.index[0], "share": mix.iloc[0], "mix": mix.to_dict(),
                   "agree": label == mix.index[0]}
    return pd.DataFrame.from_dict(rows, orient="index")


def episode_check(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, a, b, expect in EPISODES:
        w = lab.loc[pd.Period(a, "M"):pd.Period(b, "M")]
        if w.empty:
            continue
        rows.append({"episode": name, "from": a, "to": b, "expected": expect,
                     "labeled": w["label"].value_counts().idxmax(),
                     "share": (w["label"] == expect).mean()})
    return pd.DataFrame(rows)


def spells(lab: pd.DataFrame) -> pd.DataFrame:
    """One row per unbroken run of a label, oldest first, with the inputs
    averaged over it in their own units."""
    run = (lab["label"] != lab["label"].shift()).cumsum()
    rows = []
    for _, g in lab.groupby(run):
        rows.append({
            "start": g.index[0], "end": g.index[-1], "months": len(g), "label": g["label"].iloc[0],
            "gdp_growth": g["gdp_growth"].mean(), "potential_growth": g["potential_growth"].mean(),
            "growth": g["growth"].mean(),
            "core_pce": g["core_pce"].mean(), "expected": g["expected"].mean(),
            "inflation": g["inflation"].mean(),
            "clear": (g["strength"] == "clear").mean(),
            "recession_months": int(g["recession"].sum()), "provisional": bool(g["provisional"].any()),
        })
    return pd.DataFrame(rows)


def summary(lab: pd.DataFrame) -> dict:
    rec = lab["recession"]
    down = lab["label"].isin(["stagflation", "hard_landing"])
    return {
        "months": len(lab),
        "clear share": (lab["strength"] == "clear").mean(),
        "recession months growth down": down[rec].mean(),
        "expansion months growth down": down[~rec].mean(),
        "mix": lab["label"].value_counts(normalize=True).round(3).to_dict(),
    }


__all__ = ["CODES", "STORE_IDS", "NBER", "axes", "primary", "build", "spells", "episode_check", "what_followed",
           "summary", "fetch", "from_store", "to_observations", "growth_parts", "band"]
