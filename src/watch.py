"""Turning points: what would end the current regime, and is it happening?

An add-on beside the three-year call, not part of it. The model says which
regime the next three years most resemble; this asks a shorter question about
the regime we are in now: whether the pressures that ended it before are
building, and toward which exit.

Each gauge in config/watch.yml is the average of its components' scores. A
component is scored against its own full history (a z-score, capped at ±3)
and signed so that positive leans toward the way high-growth, high-inflation
spells have ended in stagflation. That is descriptive: it uses today's data
and all of history, so it shows what the pattern looked like, not what a
reader would have computed then. `track_record` reports how the pattern did
at past exits, and `realtime=True` repeats it with each score using only
earlier data.

Three gauges are tested against past exits and make up the stagflation alarm.
The rest are watch items: productivity, which decides whether inflation can
fall without growth slowing, and fiscal pressure, which has no precedent in
the sample.

Pure functions over a wide frame of stored series; the page and the tests both
call them, and nothing here touches the store or the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.transform import CLIP

HGHI, STAGFLATION, GOLDILOCKS = "high_growth_high_inflation", "stagflation", "goldilocks"


def load_config(path: str | Path = "config/watch.yml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def components(cfg: dict):
    for gauge_id, gauge in cfg["gauges"].items():
        for comp in gauge["components"]:
            yield gauge_id, comp


def required_series(cfg: dict) -> list[str]:
    out = [sid for _, comp in components(cfg) for sid in comp["series"]]
    out += [sid for panel in cfg.get("panels", {}).values() for sid in panel.get("series", {}).values()]
    out += list(cfg.get("chain", {}).get("series", {}).values())
    out += [sid for row in cfg.get("then_now", {}).get("rows", []) for sid in row.get("series", [])]
    return sorted(set(out))


def monthly(wide: pd.DataFrame, sid: str, carry: int = 5) -> pd.Series:
    """One reading a month, indexed by month.

    Daily series are averaged over the month. A quarterly reading is dated to
    the last month of its quarter, the period it describes, and held until the
    next one, for up to `carry` months.
    """
    if sid not in wide.columns:
        return pd.Series(dtype=float)
    s = wide[sid].dropna().astype(float)
    if s.empty:
        return s
    s.index = pd.DatetimeIndex(s.index)
    gap = (float(np.median(np.diff(s.index.values).astype("timedelta64[D]").astype(int)))
           if len(s) > 2 else 30.0)
    if gap > 80:   # quarterly
        s.index = s.index.to_period("Q").asfreq("M", how="end")
        s = s.groupby(level=0).last()
    else:
        s = s.groupby(s.index.to_period("M")).mean()
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    s = s.reindex(full)
    return s.ffill(limit=carry) if carry > 0 else s


def _ann(s: pd.Series, k: int) -> pd.Series:
    return ((s / s.shift(k)) ** (12 / k) - 1) * 100


def _sum12(s: pd.Series) -> pd.Series:
    return s.rolling(12, min_periods=12).sum()


def component_value(comp: dict, wide: pd.DataFrame) -> pd.Series:
    """The component in its own units, monthly, before scoring."""
    series = [monthly(wide, sid) for sid in comp["series"]]
    if any(s.empty for s in series):
        return pd.Series(dtype=float)
    a = series[0]
    b = series[1].reindex(a.index) if len(series) > 1 else None
    t = comp["transform"]
    if t == "level":
        v = a
    elif t == "yoy":
        v = _ann(a, 12)
    elif t.startswith("ann_"):          # ann_3m, ann_6m, ann_24m: pace over that many months, a year
        v = _ann(a, int(t[4:-1]))
    elif t == "chg_12m":
        v = a.diff(12)
    elif t == "spread":
        v = a - b
    elif t == "ratio":
        v = a / b * 100
    elif t == "sum12_share":            # a flow over twelve months against another flow
        v = _sum12(a) / _sum12(b) * 100
    elif t == "sum12_over_level":       # a flow over twelve months against an annual-rate level
        v = _sum12(a) / b.ffill(limit=6) * 100
    elif t == "chg_12m_gap":            # how much more one rate moved than another over a year
        v = a.diff(12) - b.diff(12)
    elif t == "diff":                   # change from the month before, such as jobs added
        v = a.diff()
    elif t == "share_of_share":         # a against a share of c: duties over goods imports (b, % of GDP c)
        v = a / (b / 100 * series[2].reindex(a.index)) * 100
    else:
        raise ValueError(f"unknown transform {t!r}")
    v = v * float(comp.get("factor", 1))
    return v.replace([np.inf, -np.inf], np.nan).dropna()


def score(values: pd.Series, sign: float = 1, realtime: bool = False,
          min_months: int = 120) -> tuple[pd.Series, dict]:
    """z-score against the series' own history, signed and capped at ±3.

    By default the average and spread come from all of history. With
    `realtime` each month uses only the months up to it (at least
    `min_months`), which is what a reader could have computed at the time.
    """
    if realtime:
        mean, std = values.expanding(min_months).mean(), values.expanding(min_months).std()
    else:
        mean, std = float(values.mean()), float(values.std())
    z = ((values - mean) / std).clip(-CLIP, CLIP) * float(sign)
    norm = {"center": float(values.mean()), "scale": float(values.std()),
            "from": str(values.index.min()) if len(values) else ""}
    return z, norm


def compute(cfg: dict, wide: pd.DataFrame, realtime: bool = False) -> dict:
    """Values, scores, gauges and the alarm, monthly."""
    meta = cfg.get("meta", {})
    carry = int(meta.get("carry_months", 6))
    values, scores, norms = {}, {}, {}
    for gauge_id, comp in components(cfg):
        key = f"{gauge_id}::{comp['id']}"
        v = component_value(comp, wide)
        if v.empty:
            continue
        z, norm = score(v, comp.get("sign", 1), realtime)
        values[key], scores[key], norms[key] = v, z, norm
    scores = pd.DataFrame(scores).sort_index()
    if not scores.empty:
        # Ragged edge: releases land on different days, so a score holds its
        # last reading while the next is due, and a late release does not move
        # a gauge by its absence.
        scores = scores.reindex(pd.period_range(scores.index.min(), scores.index.max(), freq="M"))
        scores = scores.ffill(limit=carry)
    gauges = {}
    for gauge_id, gauge in cfg["gauges"].items():
        cols = [c for c in scores.columns if c.startswith(gauge_id + "::")]
        if not cols:
            continue
        part = scores[cols]
        need = max(1, int(np.ceil(len(gauge["components"]) / 2)))   # half its components reporting
        gauges[gauge_id] = part.mean(axis=1).where(part.notna().sum(axis=1) >= need)
    gauges = pd.DataFrame(gauges)
    tested = [g for g, gauge in cfg["gauges"].items() if gauge.get("role") == "alarm" and g in gauges]
    if tested:
        alarm = gauges[tested].mean(axis=1).where(gauges[tested].notna().all(axis=1))
    else:
        alarm = pd.Series(dtype=float)
    smooth = int(meta.get("smooth_months", 3))
    return {"values": values, "scores": scores, "norms": norms, "gauges": gauges,
            "alarm": alarm.rolling(smooth, min_periods=smooth).mean(), "alarm_raw": alarm}


# ---------- the regime's past exits ----------

def spells(labels: pd.Series) -> pd.DataFrame:
    """Unbroken runs of a regime label: start, end, months, and what came next."""
    lab = labels.dropna()
    if lab.empty:
        return pd.DataFrame(columns=["label", "start", "end", "months", "next"])
    run = (lab != lab.shift()).cumsum()
    rows = [{"label": g.iloc[0], "start": g.index[0], "end": g.index[-1], "months": len(g)}
            for _, g in lab.groupby(run)]
    out = pd.DataFrame(rows)
    out["next"] = out["label"].shift(-1)
    return out


def exits(labels: pd.Series, regime: str = HGHI) -> pd.DataFrame:
    """Each finished spell of `regime` and the regime that followed it."""
    sp = spells(labels)
    return sp[(sp["label"] == regime) & sp["next"].notna()].reset_index(drop=True)


def before_exits(frame: pd.DataFrame, ex: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    """Each column of `frame` averaged over the `months` before each exit."""
    rows = []
    for r in ex.itertuples():
        window = pd.period_range(r.end - months, r.end - 1, freq="M")
        rows.append({"end": r.end, "next": r.next, **frame.reindex(window).mean().to_dict()})
    return pd.DataFrame(rows)


def track_record(alarm: pd.Series, labels: pd.Series, warn: float, regime: str = HGHI,
                 lookback: int = 24, horizon: int = 12) -> dict:
    """How the alarm did at the regime's past exits.

    For each exit: the first month the alarm was above `warn`, looking back up
    to `lookback` months within the spell, and how many months ahead that was.
    Across all finished spells: of the months above `warn`, the share followed
    within `horizon` months by a stagflation exit, a goldilocks exit, or none.
    The current spell is left out, since how it ends is not known.
    """
    ex = exits(labels, regime)
    rows = []
    for r in ex.itertuples():
        window = alarm.reindex(pd.period_range(max(r.start, r.end - lookback), r.end, freq="M"))
        hit = window[window > warn]
        last6 = alarm.reindex(pd.period_range(r.end - 6, r.end - 1, freq="M")).mean()
        rows.append({"start": r.start, "end": r.end, "next": r.next, "months": r.months,
                     "first": hit.index[0] if len(hit) else None,
                     "lead": (r.end - hit.index[0]).n if len(hit) else None,
                     "last6": float(last6)})
    per_exit = pd.DataFrame(rows)
    sp = spells(labels)
    open_spell = sp.iloc[-1] if len(sp) and sp.iloc[-1]["label"] == regime else None
    done = [p for p in labels.index[labels == regime]
            if open_spell is None or p < open_spell["start"]]
    follows = pd.Series("none", index=pd.PeriodIndex(done, freq="M"), dtype=object)
    for r in ex.itertuples():
        near = [p for p in follows.index if 0 <= (r.end - p).n <= horizon]
        follows[near] = r.next
    a = alarm.reindex(follows.index)
    flagged = follows[a > warn]
    far = [all(abs((p - e).n) > horizon for e in ex["end"]) for p in a.index]
    quiet = a[far]

    def then(what: str) -> float:
        return float((flagged == what).mean()) if len(flagged) else float("nan")

    return {"per_exit": per_exit, "flagged": int(len(flagged)), "months": int(a.notna().sum()),
            "then_stagflation": then(STAGFLATION), "then_goldilocks": then(GOLDILOCKS),
            "then_none": then("none"),
            "quiet_mean": float(quiet.mean()) if len(quiet) else float("nan")}


# ---------- the productivity check ----------

def greenspan(wide: pd.DataFrame, ids: dict, tol: float = 0.25) -> pd.DataFrame:
    """Wages, margins and inflation, each against a year earlier.

    In 1996 Greenspan read faster productivity from three facts together,
    before the official numbers showed it: pay was rising, profit margins were
    holding, and inflation was not picking up. That combination needs output
    per hour to be growing faster than reported. One row a month: the three
    readings, their change over a year, whether each condition holds (within
    `tol` percentage points), and how many do.
    """
    wages = _ann(monthly(wide, ids["wages"]), 12)
    core = _ann(monthly(wide, ids["core"]), 12)
    profit, price = monthly(wide, ids["unit_profit"]), monthly(wide, ids["unit_price"])
    if wages.empty or core.empty or profit.empty or price.empty:
        return pd.DataFrame()
    margin = profit / price.reindex(profit.index) * 100
    out = pd.DataFrame({"wages": wages, "margin": margin, "core": core})
    # Margins are quarterly and arrive a quarter late, so the newest months
    # carry the last reading and the check still reports.
    out["margin"] = out["margin"].ffill(limit=6)
    out = out.dropna()
    for k in ("wages", "margin", "core"):
        out[f"d_{k}"] = out[k].diff(12)
    out = out.dropna()
    out["ok_wages"] = out["d_wages"] >= -tol
    out["ok_margin"] = out["d_margin"] >= -tol
    out["ok_core"] = out["d_core"] <= tol
    out["met"] = out[["ok_wages", "ok_margin", "ok_core"]].sum(axis=1)
    return out


def share(num: pd.Series, den: pd.Series, factor: float = 100) -> pd.Series:
    """A spending flow against GDP, monthly."""
    if num.empty or den.empty:
        return pd.Series(dtype=float)
    return (num / den.reindex(num.index).ffill(limit=6) * factor).dropna()


def reading(x: float, warn: float) -> str:
    if x is None or pd.isna(x):
        return "not scored"
    if x >= warn:
        return "warning"
    return "building" if x >= 0 else "quiet"


# ---------- the chain from AI to a goldilocks exit ----------
#
# Four links, each judged on its own reading: "seen", "partly", "not yet", or
# "not scored" when the data is missing. The thresholds are judgements set out
# in config/watch.yml; none is tested, since the AI series start in 2023.

SEEN, PARTLY, NOT_YET, NOT_SCORED = "seen", "partly", "not yet", "not scored"


def _missing(*xs) -> bool:
    return any(x is None or pd.isna(x) for x in xs)


def year_change(s: pd.Series) -> float:
    """The latest value less the value twelve months before it."""
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1] - s.get(s.index[-1] - 12, np.nan))


def link_adoption(change: float, rise: float) -> str:
    """Firms paying for AI, change in the share over a year (pp)."""
    if _missing(change):
        return NOT_SCORED
    return SEEN if change >= rise else NOT_YET


def link_output(pace: float, fast: float, normal: float) -> str:
    """Output per hour against the IT-payoff pace (`fast`) and the 2005-19 pace (`normal`)."""
    if _missing(pace, fast, normal):
        return NOT_SCORED
    return SEEN if pace >= fast else PARTLY if pace > normal else NOT_YET


def link_costs(d_ulc: float, d_wages: float, tol: float) -> str:
    """Unit labor costs slowing while pay holds is the productivity gain reaching
    costs. If pay slowed too, only the part of the slowdown beyond pay's counts:
    partly when costs slowed by more than pay, not yet when pay explains it all."""
    if _missing(d_ulc, d_wages):
        return NOT_SCORED
    if d_ulc <= -tol and d_wages >= -tol:
        return SEEN
    return PARTLY if d_ulc <= -tol and d_ulc < d_wages - tol else NOT_YET


def link_inflation(gap: float, d_gap: float, growth: float, band: float, tol: float) -> str:
    """Seen once core inflation is inside the regime line with growth above
    potential; partly while the gap is narrowing."""
    if _missing(gap, growth):
        return NOT_SCORED
    if gap < band and growth > 0:
        return SEEN
    return PARTLY if not _missing(d_gap) and d_gap <= -tol else NOT_YET


# ---------- the IT boom against today ----------
#
# One reading per row for a past period and for the last few months. "growth"
# is the annual pace of a level across the period, "mean" the average, "start"
# the value in the period's first month against the latest.

BETTER, WORSE, SIMILAR = "better", "worse", "similar"


def period_stat(v: pd.Series, start: str, end: str, kind: str) -> float:
    v = v.dropna()
    a, b = pd.Period(start, "M"), pd.Period(end, "M")
    if kind == "growth":
        x0, x1 = v.get(a - 1, np.nan), v.get(b, np.nan)
        return float(((x1 / x0) ** (12 / ((b - a).n + 1)) - 1) * 100) if not _missing(x0, x1) else np.nan
    if kind == "start":
        return float(v.get(a, np.nan))
    part = v[a:b]
    return float(part.mean()) if len(part) else np.nan


def recent_stat(v: pd.Series, months: int, kind: str) -> float:
    v = v.dropna()
    if v.empty:
        return np.nan
    if kind == "growth":
        x0 = v.get(v.index[-1] - months, np.nan)
        return float(((v.iloc[-1] / x0) ** (12 / months) - 1) * 100) if not _missing(x0) else np.nan
    if kind == "start":
        return float(v.iloc[-1])
    return float(v[v.index > v.index[-1] - months].mean())


def compare(then: float, now: float, helps: float, similar: float) -> str:
    """Whether today's reading is better or worse for a goldilocks exit than the
    past one. `helps` is +1 when a higher reading helps; a gap under `similar`
    reads as about the same."""
    if _missing(then, now):
        return NOT_SCORED
    d = (now - then) * helps
    return BETTER if d > similar else WORSE if d < -similar else SIMILAR
