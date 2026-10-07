"""The regime each month was actually in since 1980, judged with hindsight.

Two independent recipes label every month on a growth x inflation grid, and
the months where they agree are the truth set the models are scored against.

Primary (anchored to trend and to expectations):
  growth      two measures, each scaled by its typical move, then averaged:
              the Chicago Fed National Activity Index over the month and its
              neighbours, against its median over the ten years around it
              (its zero is the 1967-onward pace, above the trend since 2000);
              and real GDP growth less CBO potential over the half-year around
              it, which carries the services side the CFNAI underweights.
  inflation   core PCE annualized over the twelve months centred on the month,
              minus what was expected: the SPF ten-year CPI median, less the
              usual gap between CPI and PCE inflation. Before 1991 the Philly
              Fed fills that series from Blue Chip and Livingston surveys.
  Each axis flips only when it clears a band on the other side of zero, so a
  wobble around zero keeps the last call and is flagged as weak. A quadrant
  must then last three months; shorter spells take the label before them.

Cross-check (AQR's recipe, Ilmanen, Maloney and Ross, 2014):
  growth      z(CFNAI) + z(IP growth over the year less what the SPF expected)
  inflation   z(CPI inflation over the year) + z(that less what the SPF expected)
  Each axis is up when above its full-sample median. This is relative, not
  anchored: half of all months are high inflation by construction.

Quadrants use the model's names: up and low is goldilocks, up and high is
high_growth_high_inflation, down and high is stagflation, down and low is
hard_landing. Unlike backtest.realised, a recession is not forced to
hard_landing: 1980 and 1990 count as stagflation if inflation ran hot.

Everything is on today's revised data, which is the point: this is what
happened, not what anyone could see at the time.
"""

from __future__ import annotations

import pandas as pd

from .backtest import NBER, recession_months

CODES = {
    "cfnai": "ussurv1198",          # Chicago Fed National Activity Index
    "ip": "usprod1022",             # industrial production, SA
    "cpi": "uspric2156",            # CPI all items, SA
    "core_pce": "uspric0006",       # PCE excluding food and energy
    "pce": "uspric0001",            # PCE, total
    "spf_cpi10": "usfcst0062",      # SPF median CPI over ten years, from 1979
    "spf_ip_q0": "usfcst0896",      # SPF IP level, current quarter
    "spf_ip_q4": "usfcst0900",      # SPF IP level, four quarters ahead
    "spf_cpi1y": "usfcst0666",      # SPF median CPI over the next year, from 1981
    "gdp": "usnaac0169",            # real GDP, chained, SAAR
    "potential": "usfcst1985",      # CBO real potential GDP, quarterly
}

QUADRANT = {(True, False): "goldilocks", (True, True): "high_growth_high_inflation",
            (False, True): "stagflation", (False, False): "hard_landing"}

GROWTH_BAND = 0.25      # in typical spreads of the growth measures
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


# Names the inputs are stored under, so the hosted app can rebuild the labels
# from the published snapshot. Prefixed so they never collide with indicators.
STORE_IDS = {name: f"TR_{name.upper()}" for name in CODES}

TITLES = {
    "cfnai": ("Chicago Fed National Activity Index", "index", "Monthly"),
    "ip": ("Industrial production", "index", "Monthly"),
    "cpi": ("Consumer price index, all items", "index", "Monthly"),
    "core_pce": ("PCE price index excluding food and energy", "index", "Monthly"),
    "pce": ("PCE price index", "index", "Monthly"),
    "spf_cpi10": ("SPF median expected CPI inflation, next 10 years", "%", "Quarterly"),
    "spf_ip_q0": ("SPF median industrial production, current quarter", "index", "Quarterly"),
    "spf_ip_q4": ("SPF median industrial production, four quarters ahead", "index", "Quarterly"),
    "spf_cpi1y": ("SPF median expected CPI inflation, next year", "%", "Quarterly"),
    "gdp": ("Real GDP", "USD bn, SAAR", "Quarterly"),
    "potential": ("CBO real potential GDP", "USD bn, SAAR", "Quarterly"),
}


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


def _ann(level: pd.Series, months: int, shift: int = 0) -> pd.Series:
    """Annualized % change over `months`, ending `shift` months after each month."""
    return ((level.shift(-shift) / level.shift(months - shift)) ** (12 / months) - 1) * 100


def _spread(s: pd.Series) -> float:
    """Typical size of a move, from 1980-2019 so Covid does not set it, and by
    median absolute deviation so recessions do not either."""
    base = s.loc[pd.Period("1980-01", "M"):pd.Period("2019-12", "M")].dropna()
    return 1.4826 * (base - base.median()).abs().median()


def gdp_growth_parts(gdp: pd.Series, potential: pd.Series) -> pd.DataFrame:
    """Real GDP and CBO potential growth, % annualized over the half-year
    centred on each quarter (trailing for the latest), placed on the middle
    month of the quarter and interpolated."""
    def q(s):
        s = s.dropna()
        return s.groupby(s.index.asfreq("Q")).mean()

    def growth(x):
        centred = ((x.shift(-1) / x.shift(1)) ** 2 - 1) * 100
        return centred.combine_first(((x / x.shift(1)) ** 4 - 1) * 100)

    g = q(gdp)
    parts = pd.DataFrame({"gdp_growth": growth(g), "potential_growth": growth(q(potential).reindex(g.index))}).dropna()
    parts.index = parts.index.asfreq("M", how="start") + 1
    full = pd.period_range(parts.index[0] - 1, parts.index[-1] + 1, freq="M")
    return parts.reindex(full).interpolate(limit_area="inside").ffill().bfill()


def gdp_gap_growth(gdp: pd.Series, potential: pd.Series) -> pd.Series:
    parts = gdp_growth_parts(gdp, potential)
    return parts["gdp_growth"] - parts["potential_growth"]


def axes(m: pd.DataFrame) -> pd.DataFrame:
    """Growth and inflation gap per month, with every input in its own units so
    a reader can check them. Near the end the inflation window runs to the
    latest print instead, and those months are marked provisional."""
    cfnai = m["cfnai"].dropna()
    # The index's zero is the average pace since 1967, above the trend of the
    # 2000s; measure it against the median of the ten years around it.
    cfnai_ma3 = cfnai.rolling(3, center=True, min_periods=2).mean()
    cfnai_trend = cfnai.rolling(120, center=True, min_periods=60).median()
    cf = cfnai_ma3 - cfnai_trend
    parts = gdp_growth_parts(m["gdp"], m["potential"])
    gdp = parts["gdp_growth"] - parts["potential_growth"]
    growth = pd.concat([cf / _spread(cf), gdp / _spread(gdp)], axis=1).mean(axis=1)
    core = m["core_pce"].dropna()
    centred = _ann(core, 12, shift=6)
    last = core.index[-1]
    edge = pd.Series({t: ((core[last] / core[t - 6]) ** (12 / (last - (t - 6)).n) - 1) * 100
                      for t in core.index[-6:]}, dtype=float)
    inflation = centred.combine_first(edge)
    # CPI runs above PCE; take the gap over the ten years around each month.
    wedge = (m["cpi"].pct_change(12, fill_method=None) - m["pce"].pct_change(12, fill_method=None)) * 100
    wedge = wedge.rolling(120, center=True, min_periods=60).mean().ffill().bfill()
    spf = m["spf_cpi10"].interpolate(limit_area="inside").ffill()
    out = pd.DataFrame({
        "growth": growth, "growth_cfnai": cf, "growth_gdp": gdp,
        "cfnai_ma3": cfnai_ma3, "cfnai_trend": cfnai_trend,
        "gdp_growth": parts["gdp_growth"], "potential_growth": parts["potential_growth"],
        "core_pce": inflation, "spf_cpi10": spf, "wedge": wedge, "expected": spf - wedge})
    out["inflation"] = out["core_pce"] - out["expected"]
    out["provisional"] = out.index > core.index[-7]
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


def _quarterly_surprise(level: pd.Series, q0: pd.Series, q4: pd.Series) -> pd.Series:
    """Realized growth over four quarters less the SPF forecast made at the
    start, dated at the quarter it ends, held for its three months."""
    qa = level.groupby(level.index.asfreq("Q")).mean()
    real = (qa / qa.shift(4) - 1) * 100
    fc = q4.copy()
    fc.index = fc.index.asfreq("Q") + 4
    if q0 is not None:
        start = q0.copy()
        start.index = start.index.asfreq("Q") + 4
        fc = (fc / start - 1) * 100
    s = (real - fc).dropna()
    monthly = s.copy()
    monthly.index = s.index.asfreq("M", how="start")
    return monthly.reindex(pd.period_range(monthly.index[0], s.index[-1].asfreq("M", how="end"))).ffill()


def aqr(m: pd.DataFrame) -> pd.DataFrame:
    def z(s):
        s = s.dropna()
        return (s - s.mean()) / s.std()

    ip_surprise = _quarterly_surprise(m["ip"].dropna(), m["spf_ip_q0"].dropna(), m["spf_ip_q4"].dropna())
    cpi_yoy = m["cpi"].pct_change(12, fill_method=None) * 100
    cpi_surprise = _quarterly_surprise(m["cpi"].dropna(), None, m["spf_cpi1y"].dropna())
    growth = pd.concat([z(m["cfnai"]), z(ip_surprise)], axis=1).mean(axis=1)
    inflation = pd.concat([z(cpi_yoy), z(cpi_surprise)], axis=1).mean(axis=1)
    both = pd.DataFrame({"growth": growth, "inflation": inflation}).dropna()
    up = both["growth"] > both["growth"].median()
    high = both["inflation"] > both["inflation"].median()
    both["label"] = [QUADRANT[(a, b)] for a, b in zip(up, high)]
    return both


def build(m: pd.DataFrame, start: str = "1980-01") -> pd.DataFrame:
    """One row per month from `start`: both recipes, whether they agree, and
    the truth label (blank where they disagree)."""
    ax = axes(m)
    p = primary(ax)
    a = aqr(m)
    out = ax.join(p, how="inner").join(a.add_prefix("aqr_"), how="left")
    out = out[out.index >= pd.Period(start, "M")]
    out["recession"] = recession_months(out.index)
    out["agree"] = out["label"] == out["aqr_label"]
    out["truth"] = out["label"].where(out["agree"])
    return out


def episode_check(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, a, b, expect in EPISODES:
        w = lab.loc[pd.Period(a, "M"):pd.Period(b, "M")]
        if w.empty:
            continue
        rows.append({"episode": name, "from": a, "to": b, "expected": expect,
                     "primary": w["label"].value_counts().idxmax(),
                     "primary share": (w["label"] == expect).mean(),
                     "aqr share": (w["aqr_label"] == expect).mean()})
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
            "cfnai_ma3": g["cfnai_ma3"].mean(), "cfnai_trend": g["cfnai_trend"].mean(),
            "growth": g["growth"].mean(),
            "core_pce": g["core_pce"].mean(), "expected": g["expected"].mean(),
            "inflation": g["inflation"].mean(),
            "agree": g["agree"].mean(), "weak": (g["strength"] == "weak").mean(),
            "recession_months": int(g["recession"].sum()), "provisional": bool(g["provisional"].any()),
        })
    return pd.DataFrame(rows)


def summary(lab: pd.DataFrame) -> dict:
    rec = lab["recession"]
    down = lab["label"].isin(["stagflation", "hard_landing"])
    return {
        "months": len(lab),
        "agreement": lab["agree"].mean(),
        "weak share": (lab["strength"] == "weak").mean(),
        "recession months growth down": down[rec].mean(),
        "expansion months growth down": down[~rec].mean(),
        "primary mix": lab["label"].value_counts(normalize=True).round(3).to_dict(),
        "aqr mix": lab["aqr_label"].value_counts(normalize=True).round(3).to_dict(),
    }


__all__ = ["CODES", "STORE_IDS", "NBER", "axes", "primary", "aqr", "build", "spells",
           "episode_check", "summary", "fetch", "from_store", "to_observations"]
