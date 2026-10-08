"""The regime each month was actually in since 1980, judged with hindsight.

Two numbers per month, both in percentage points:

  growth      real GDP growth less CBO potential growth, over the four quarters
              centred on the month. Above zero, the economy grew faster than
              its capacity.
  inflation   core PCE inflation over the twelve months centred on the month,
              less expected inflation: the SPF ten-year CPI median, less the
              usual gap between CPI and PCE inflation. Before 1991 the Philly
              Fed fills that survey series from Blue Chip and Livingston.

Each axis flips only when it clears its band on the other side of zero, so a
wobble around zero keeps the last call and is marked weak. A quadrant must last
three months; shorter spells take the label before them.

Quadrants use the model's names: up and low is goldilocks, up and high is
high_growth_high_inflation, down and high is stagflation, down and low is
hard_landing. Unlike backtest.realised, a recession is not forced to
hard_landing: 1980 and 1990 count as stagflation because inflation ran hot.

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
}

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
}

QUADRANT = {(True, False): "goldilocks", (True, True): "high_growth_high_inflation",
            (False, True): "stagflation", (False, False): "hard_landing"}

GROWTH_BAND = 0.5       # percentage points
INFLATION_BAND = 0.5    # percentage points
MIN_MONTHS = 3

# What the record says each stretch was, to check the labels against. The
# expected label is the textbook reading; the check reports the share agreeing.
EPISODES = [
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


# The adopted method and the alternatives the /history page lets reviewers
# compare. Window is in years; alignment is centred on the month or trailing
# (ending in it); the inflation benchmark is the survey or a stepped line.
DEFAULT = {"years": 1, "align": "centered", "anchor": "survey"}
WINDOWS = (1, 3, 5)
ALIGNS = ("centered", "trailing")
ANCHORS = ("survey", "stepped")
# A stepped inflation line, for those who read the 1980s as high inflation:
# (applies before this month, level in %), last entry for everything after.
STEPS = [("1990-01", 4.0), ("1996-01", 3.0), (None, 2.0)]


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


def _windowed(level: pd.Series, years: int, per_year: int, align: str) -> pd.Series:
    """% a year over `years`, centred on each period or ending in it."""
    n = years * per_year
    if align == "centered":
        return _centred(level, n // 2, per_year)
    level = level.dropna()
    return ((level / level.shift(n)) ** (per_year / n) - 1) * 100


def stepped_anchor(index: pd.PeriodIndex) -> pd.Series:
    out = pd.Series(STEPS[-1][1], index=index, dtype=float)
    for until, level in reversed(STEPS[:-1]):
        out[index < pd.Period(until, "M")] = level
    return out


def growth_parts(gdp: pd.Series, potential: pd.Series, years: int = 1,
                 align: str = "centered") -> pd.DataFrame:
    """Real GDP and CBO potential growth, % a year over the window around (or
    ending in) each quarter, placed on the quarter's middle month and
    interpolated between."""
    def q(s):
        s = s.dropna()
        return s.groupby(s.index.asfreq("Q")).mean()

    g = q(gdp)
    p = q(potential).reindex(g.index)
    parts = pd.DataFrame({"gdp_growth": _windowed(g, years, 4, align),
                          "potential_growth": _windowed(p, years, 4, align)}).dropna()
    parts.index = parts.index.asfreq("M", how="start") + 1
    full = pd.period_range(parts.index[0] - 1, parts.index[-1] + 1, freq="M")
    out = parts.reindex(full).interpolate(limit_area="inside").ffill().bfill()
    # A centred window is incomplete for the last half-window of quarters.
    half = 2 * years
    out["provisional"] = (out.index >= g.index[-half].asfreq("M", how="start")) if align == "centered" else False
    return out


def axes(m: pd.DataFrame, years: int = 1, align: str = "centered", anchor: str = "survey") -> pd.DataFrame:
    """Growth and inflation gap per month, with every input in its own units so
    a reader can check them."""
    parts = growth_parts(m["gdp"], m["potential"], years, align)
    core = m["core_pce"].dropna()
    inflation = _windowed(core, years, 12, align)
    # CPI runs above PCE; take the gap over the ten years around each month.
    wedge = (m["cpi"].pct_change(12, fill_method=None) - m["pce"].pct_change(12, fill_method=None)) * 100
    wedge = wedge.rolling(120, center=True, min_periods=60).mean().ffill().bfill()
    spf = m["spf_cpi10"].interpolate(limit_area="inside").ffill()
    expected = spf - wedge
    if years > 1:
        # Over a multi-year window, compare with what was expected over it.
        expected = expected.rolling(12 * years, center=align == "centered", min_periods=1).mean()
    out = pd.DataFrame({
        "gdp_growth": parts["gdp_growth"], "potential_growth": parts["potential_growth"],
        "core_pce": inflation, "spf_cpi10": spf, "wedge": wedge, "expected": expected})
    if anchor == "stepped":
        out["expected"] = stepped_anchor(out.index)
    out["growth"] = out["gdp_growth"] - out["potential_growth"]
    out["inflation"] = out["core_pce"] - out["expected"]
    out = out.dropna(subset=["growth", "inflation"])
    late = (out.index > core.index[-(6 * years) - 1]) if align == "centered" else False
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


def build(m: pd.DataFrame, start: str = "1980-01", years: int = 1, align: str = "centered",
          anchor: str = "survey") -> pd.DataFrame:
    """One row per month from `start`: the inputs, both gaps and the label."""
    ax = axes(m, years, align, anchor)
    out = ax.join(primary(ax), how="inner")
    out = out[out.index >= pd.Period(start, "M")]
    out["recession"] = recession_months(out.index)
    # Half-complete windows are not enough to call a turn: one weak quarter at
    # the edge once flipped the latest months to stagflation and back.
    out["provisional_reading"] = out["label"]
    confirmed = out.loc[~out["provisional"], "label"]
    if len(confirmed):
        out.loc[out["provisional"], "label"] = confirmed.iloc[-1]
    return out


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


__all__ = ["CODES", "STORE_IDS", "NBER", "axes", "primary", "build", "spells", "episode_check",
           "summary", "fetch", "from_store", "to_observations", "growth_parts"]
