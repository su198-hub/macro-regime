"""Driver aggregation.

Reads the indicator config, pulls the series it names as of a stated vintage,
transforms, normalises, applies direction, and takes a weighted mean per
driver. Output is a monthly frame with one column per driver in roughly -1 to +1.

Indicator-level scores are kept alongside the driver scores. When someone
asks why a driver moved, you need the answer in one click, not a rerun.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .transform import TRANSFORMS, _periods_per_year, normalise, to_monthly

DRIVER_SCALE = 2.0  # weighted clipped scores divided by this to land near ±1


def load_config(path: str | Path = "config/indicators.yml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Stamp every regime call with the config that produced it.

    Without this you cannot answer 'why did March change' six months later,
    because you will have edited the weights twice since.
    """
    blob = yaml.safe_dump(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def required_series(cfg: dict) -> list[str]:
    ids = set()
    for driver in cfg["drivers"].values():
        for ind in driver["indicators"]:
            src = ind["source"]
            fred = src.get("fred")
            if isinstance(fred, str):
                ids.add(fred)
            elif isinstance(fred, list):
                ids.update(fred)
            for d in (src.get("derived") or {}).values():
                if isinstance(d.get("fred"), str):
                    ids.add(d["fred"])
    return sorted(ids)


def _build_input(ind: dict, wide: pd.DataFrame, carry: int = 0) -> pd.Series | None:
    """One indicator's input, transformed, monthly.

    Lower-frequency inputs are carried forward to the latest month any series
    reaches, for up to `carry` of their own periods (see to_monthly).

    A year-on-year change is taken at the series' own frequency *before* it is
    carried, which matters at the ragged edge. Carry a quarterly series first
    and a twelve-month change compares the last quarter published against
    whatever the month a year ago held, which in the months after a release is
    three quarters back, not four: wage growth of 3.1% printed as 2.3%. The
    error appears only in the newest months, which is where the dashboard
    looks.
    """
    src = ind["source"]
    expr = src.get("expr")
    end = wide.index.max() if not wide.empty else None
    transform = TRANSFORMS[ind.get("transform", "level")]

    if not expr:
        col = src["fred"]
        if col not in wide.columns:
            return None
        return to_monthly(transform(wide[col].dropna()), end, carry)

    # An expression over series that all share one frequency can also be built
    # at that frequency, transformed there, and carried afterwards. Mixed
    # frequencies cannot: the monthly side has to drive the index, so those
    # combine first and transform after, as before.
    natives = {_periods_per_year(wide[c].dropna()) for c in src.get("fred", [])
               if c in wide.columns}
    native_first = len(natives) == 1 and natives.pop() < 12 and not src.get("derived")

    env: dict[str, pd.Series] = {}
    native: dict[str, pd.Series] = {}
    for col in src.get("fred", []):
        if col not in wide.columns:
            return None
        native[col] = wide[col].dropna()
        env[col] = native[col] if native_first else to_monthly(native[col], end, carry)

    # Derived inputs are built in order, so each can use the ones before it:
    #   {fred: X, transform: t}      a transform of one source series
    #   {expr: "...", floor: f}      a formula over inputs so far, optionally floored
    #   {first_of: [a, b]}           a where available, otherwise b
    for alias, spec in (src.get("derived") or {}).items():
        if "first_of" in spec:
            parts = [env.get(name) for name in spec["first_of"]]
            if any(p is None for p in parts):
                return None
            combined = parts[0]
            for p in parts[1:]:
                combined = combined.combine_first(p)
            env[alias] = combined
        elif "expr" in spec:
            frame = pd.DataFrame(env)
            try:
                value = frame.eval(spec["expr"])
            except Exception as exc:
                raise ValueError(f"bad derived expr {alias} for {ind['id']}: {spec['expr']}") from exc
            if "floor" in spec:
                value = value.where(value.isna() | (value >= float(spec["floor"])),
                                    float(spec["floor"]))
            env[alias] = value
        else:
            base = env.get(spec["fred"])
            if base is None:
                return None
            fn = TRANSFORMS[spec.get("transform", "level")]
            # Same rule as the indicator's own transform: take the change at
            # the series' own frequency, then carry. A quarterly series carried
            # first and differenced after loses a quarter at the ragged edge.
            raw = native.get(spec["fred"])
            if raw is not None and not native_first and _periods_per_year(raw) < 12:
                env[alias] = to_monthly(fn(raw), end, carry)
            else:
                env[alias] = fn(base)

    frame = pd.DataFrame(env).dropna(how="all")
    if frame.empty:
        return None
    try:
        combined = frame.eval(expr)
    except Exception as exc:
        raise ValueError(f"bad expr for {ind['id']}: {expr}") from exc
    if native_first:
        return to_monthly(transform(combined.dropna()), end, carry)
    return transform(combined)


def indicator_frames(cfg: dict, wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(transformed input before normalisation, normalised direction-adjusted score).

    Both monthly, columns "driver::indicator". The first is the number an
    analyst would recognise, such as consumption up 2.9% on a year earlier; the
    second is what enters the driver.
    """
    inputs: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    carry = int((cfg.get("meta") or {}).get("carry_forward_periods", 0))
    for driver_name, driver in cfg["drivers"].items():
        for ind in driver["indicators"]:
            transformed = _build_input(ind, wide, int(ind.get("carry_forward_periods", carry)))
            if transformed is None or transformed.dropna().empty:
                continue
            key = f"{driver_name}::{ind['id']}"
            inputs[key] = transformed
            scores[key] = normalise(transformed, ind["normalize"]) * int(ind["direction"])
    if not scores:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(inputs).sort_index(), pd.DataFrame(scores).sort_index()


def indicator_scores(cfg: dict, wide: pd.DataFrame) -> pd.DataFrame:
    """Normalised, direction-adjusted score per indicator, monthly."""
    return indicator_frames(cfg, wide)[1]


def driver_scores(cfg: dict, wide: pd.DataFrame, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weighted mean of available indicators, renormalised for coverage.

    Weights are renormalised over whatever is actually present each month, so
    a driver does not silently collapse toward zero in early history when only
    two of its four indicators exist. It does mean early driver scores rest on
    fewer inputs, which the coverage column reports.

    These are confirmed scores. The latest months, where releases are still
    arriving, are left unscored and read by provisional_reading instead.
    """
    if scores is None:
        scores = indicator_scores(cfg, wide)
    if scores.empty:
        return pd.DataFrame()
    # Ragged edge: in the latest months some inputs have not been released yet.
    # A month where less than this share of the weight of indicators that had
    # already started is present is incomplete, and gets no score rather than
    # one resting on whichever release came first. Indicators that did not
    # exist yet do not count against a month, so early history survives.
    min_reported = float((cfg.get("meta") or {}).get("min_reported_share", 0))

    frames = {}
    for driver_name, driver in cfg["drivers"].items():
        cols, weights, anchors = [], [], []
        for ind in driver["indicators"]:
            col = f"{driver_name}::{ind['id']}"
            if col in scores.columns:
                cols.append(col)
                weights.append(float(ind["weight"]))
                if ind.get("anchor"):
                    anchors.append(col)
        if not cols:
            continue

        block = scores[cols]
        w = np.array(weights)
        present = block.notna().to_numpy().astype(float)
        denom = present @ w
        numer = np.nansum(block.to_numpy() * w, axis=1)
        started = block.notna().cummax().to_numpy().astype(float) @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(denom > 0, numer / denom, np.nan)
            reported = np.where(started > 0, denom / started, 0.0)
        value = np.where(reported >= min_reported, value, np.nan)
        # An anchor that has started but not reported this month holds the
        # month back from confirmation, however much else has arrived.
        for col in anchors:
            missing = (block[col].notna().cummax() & block[col].isna()).to_numpy()
            value = np.where(missing, np.nan, value)

        frames[driver_name] = pd.Series(
            np.clip(value / DRIVER_SCALE, -1, 1), index=block.index
        )
        frames[f"{driver_name}__coverage"] = pd.Series(
            denom / w.sum(), index=block.index
        )

    return pd.DataFrame(frames).sort_index()


def driver_breakdown(cfg: dict, driver_name: str, inputs: pd.DataFrame,
                     scores: pd.DataFrame, as_of) -> pd.DataFrame:
    """Each indicator's part in one driver's score for one month.

    contribution = weight share among available indicators × score / DRIVER_SCALE,
    so the column sums to the driver score before it is clipped to ±1.
    """
    rows = []
    for ind in cfg["drivers"][driver_name]["indicators"]:
        key = f"{driver_name}::{ind['id']}"
        has = key in scores.columns and as_of in scores.index
        rows.append({
            "id": ind["id"],
            "value": inputs.at[as_of, key] if has else np.nan,
            "score": scores.at[as_of, key] if has else np.nan,
            "weight": float(ind["weight"]),
        })
    out = pd.DataFrame(rows)
    present = out["score"].notna()
    total = out.loc[present, "weight"].sum()
    out["share"] = np.where(present & (total > 0), out["weight"] / (total or 1), np.nan)
    out["contribution"] = out["share"] * out["score"] / DRIVER_SCALE
    return out


# ---------- provisional reading ----------

def indicator_inputs(cfg: dict) -> dict[str, list[str]]:
    """Source series behind each indicator, keyed "driver::indicator"."""
    out = {}
    for driver_name, driver in cfg["drivers"].items():
        for ind in driver["indicators"]:
            fred = ind["source"].get("fred")
            out[f"{driver_name}::{ind['id']}"] = [fred] if isinstance(fred, str) else list(fred or [])
    return out


def series_frequency(wide: pd.DataFrame) -> dict[str, int]:
    """Native periods per year of each raw series, inferred from its dates."""
    return {c: _periods_per_year(wide[c].dropna()) for c in wide.columns}


def provisional_reading(cfg: dict, scores: pd.DataFrame, month: pd.Timestamp) -> dict:
    """Read one not-yet-confirmed month from what has been released so far.

    An indicator released for the month uses its own score. One not released
    yet carries its latest score, for at most `provisional.fill_months`, and is
    marked as carried. Weights are renormalised over released and carried
    indicators, as for confirmed scores.

    Returns drivers (score per driver), reported (share of each driver's
    started weight released for the month), share (mean of those) and status,
    one row per indicator: reported, carried or pending.
    """
    prov = (cfg.get("meta") or {}).get("provisional") or {}
    fill = int(prov.get("fill_months", 2))
    history = scores.loc[:month]
    rows, drivers, reported = [], {}, {}
    for driver_name, driver in cfg["drivers"].items():
        num = den = rep = started = 0.0
        for ind in driver["indicators"]:
            key = f"{driver_name}::{ind['id']}"
            w = float(ind["weight"])
            s = history[key].dropna() if key in history.columns else pd.Series(dtype=float)
            row = {"driver": driver_name, "key": key, "id": ind["id"], "weight": w,
                   "anchor": bool(ind.get("anchor")), "score": np.nan, "last_month": pd.NaT}
            if s.empty:
                row["status"] = "not_started"
            else:
                started += w
                row["last_month"] = s.index[-1]
                lag = (month.year - s.index[-1].year) * 12 + month.month - s.index[-1].month
                if lag == 0:
                    row.update(status="reported", score=s.iloc[-1])
                    rep += w
                elif lag <= fill:
                    row.update(status="carried", score=s.iloc[-1])
                else:
                    row["status"] = "pending"
                if not np.isnan(row["score"]):
                    num += w * row["score"]
                    den += w
            rows.append(row)
        drivers[driver_name] = np.clip(num / den / DRIVER_SCALE, -1, 1) if den else np.nan
        reported[driver_name] = rep / started if started else np.nan
    reported = pd.Series(reported)
    return {"month": month, "drivers": pd.Series(drivers), "reported": reported,
            "share": float(reported.mean()), "status": pd.DataFrame(rows)}


def provisional_months(cfg: dict, drivers: pd.DataFrame, scores: pd.DataFrame) -> list[dict]:
    """Provisional readings for months after the latest fully confirmed one."""
    names = [n for n in cfg["drivers"] if n in drivers.columns]
    if not names or scores.empty:
        return []
    confirmed = drivers[names].dropna()
    after = confirmed.index[-1] if len(confirmed) else scores.index[0]
    threshold = float(((cfg.get("meta") or {}).get("provisional") or {}).get("min_reported_share", 0.5))
    out = []
    for month in scores.index[scores.index > after]:
        reading = provisional_reading(cfg, scores, month)
        if reading["share"] >= threshold and reading["drivers"].notna().all():
            out.append(reading)
    return out


def expected_release(month: pd.Timestamp, periods_per_year: int, lag_days: float) -> pd.Timestamp:
    """When a series' value covering `month` should appear, from its usual lag.

    Monthly values are complete at month end, quarterly at quarter end; weekly
    and daily series count as complete at month end too, because the monthly
    reading takes the month's last value.
    """
    end = month + pd.offsets.MonthEnd(0)
    if periods_per_year == 4:
        end = month + pd.offsets.QuarterEnd(0)
    elif periods_per_year == 1:
        end = month + pd.offsets.YearEnd(0)
    return (end + pd.Timedelta(days=float(lag_days))).normalize()


def median_release_lag(first_published: pd.DataFrame, periods_per_year: int) -> float:
    """Median days from period end to first publication.

    first_published: columns observation_date, published, one row per recent
    observation. Observation dates are period starts.
    """
    if first_published.empty:
        return np.nan
    obs = pd.to_datetime(first_published["observation_date"])
    if periods_per_year >= 12:
        ends = obs + pd.offsets.MonthEnd(0) if periods_per_year == 12 else obs
    elif periods_per_year == 4:
        ends = obs + pd.offsets.QuarterEnd(0)
    else:
        ends = obs + pd.offsets.YearEnd(0)
    lags = (pd.to_datetime(first_published["published"]) - ends).dt.days
    return float(lags.median())


def compute(store, cfg: dict, vintage: dt.date | None = None) -> dict:
    """Everything the dashboard needs for one point in time."""
    vintage = vintage or dt.date.today()
    series = required_series(cfg)
    wide = store.as_of(series, vintage)
    if wide.empty:
        return {"drivers": pd.DataFrame(), "indicators": pd.DataFrame(),
                "inputs": pd.DataFrame(), "provisional": [], "vintage": vintage}
    inputs, scores = indicator_frames(cfg, wide)
    drivers = driver_scores(cfg, wide, scores)
    freq = series_frequency(wide)
    lags = {sid: median_release_lag(store.first_published(sid, vintage), freq.get(sid, 12))
            for sid in freq}
    return {
        "drivers": drivers,
        "indicators": scores,
        "inputs": inputs,
        "provisional": provisional_months(cfg, drivers, scores),
        "frequency": freq,
        "release_lags": lags,
        "last_obs": {c: wide[c].dropna().index.max() for c in wide.columns},
        "vintage": vintage,
        "config_hash": config_hash(cfg),
    }
