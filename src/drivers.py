"""Driver aggregation.

Reads the indicator config, pulls the series it names as of a stated vintage,
transforms, normalises, applies direction, and takes a weighted mean per
driver. Output is a monthly frame of five columns in roughly -1 to +1.

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

from .transform import TRANSFORMS, normalise, to_monthly

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
                ids.add(d["fred"])
    return sorted(ids)


def _build_input(ind: dict, wide: pd.DataFrame) -> pd.Series | None:
    """Resolve one indicator's raw input, handling derived expressions."""
    src = ind["source"]
    expr = src.get("expr")

    if not expr:
        col = src["fred"]
        if col not in wide.columns:
            return None
        return to_monthly(wide[col].dropna())

    env: dict[str, pd.Series] = {}
    for col in src.get("fred", []):
        if col not in wide.columns:
            return None
        env[col] = to_monthly(wide[col].dropna())

    for alias, spec in (src.get("derived") or {}).items():
        base = env.get(spec["fred"])
        if base is None:
            return None
        env[alias] = TRANSFORMS[spec.get("transform", "level")](base)

    frame = pd.DataFrame(env).dropna(how="all")
    if frame.empty:
        return None
    try:
        return frame.eval(expr)
    except Exception as exc:
        raise ValueError(f"bad expr for {ind['id']}: {expr}") from exc


def indicator_frames(cfg: dict, wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(transformed input before normalisation, normalised direction-adjusted score).

    Both monthly, columns "driver::indicator". The first is the number an
    analyst would recognise, such as consumption up 2.9% on a year earlier; the
    second is what enters the driver.
    """
    inputs: dict[str, pd.Series] = {}
    scores: dict[str, pd.Series] = {}
    for driver_name, driver in cfg["drivers"].items():
        for ind in driver["indicators"]:
            raw = _build_input(ind, wide)
            if raw is None or raw.dropna().empty:
                continue
            transformed = TRANSFORMS[ind.get("transform", "level")](raw)
            key = f"{driver_name}::{ind['id']}"
            inputs[key] = transformed
            scores[key] = normalise(transformed, ind["normalize"]) * int(ind["direction"])
    if not scores:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(inputs).sort_index(), pd.DataFrame(scores).sort_index()


def indicator_scores(cfg: dict, wide: pd.DataFrame) -> pd.DataFrame:
    """Normalised, direction-adjusted score per indicator, monthly."""
    return indicator_frames(cfg, wide)[1]


def driver_scores(cfg: dict, wide: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean of available indicators, renormalised for coverage.

    Weights are renormalised over whatever is actually present each month, so
    a driver does not silently collapse toward zero in early history when only
    two of its four indicators exist. It does mean early driver scores rest on
    fewer inputs, which the coverage column reports.
    """
    scores = indicator_scores(cfg, wide)
    if scores.empty:
        return pd.DataFrame()

    frames = {}
    for driver_name, driver in cfg["drivers"].items():
        cols, weights = [], []
        for ind in driver["indicators"]:
            col = f"{driver_name}::{ind['id']}"
            if col in scores.columns:
                cols.append(col)
                weights.append(float(ind["weight"]))
        if not cols:
            continue

        block = scores[cols]
        w = np.array(weights)
        present = block.notna().to_numpy().astype(float)
        denom = present @ w
        numer = np.nansum(block.to_numpy() * w, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(denom > 0, numer / denom, np.nan)

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


def compute(store, cfg: dict, vintage: dt.date | None = None) -> dict:
    """Everything the dashboard needs for one point in time."""
    vintage = vintage or dt.date.today()
    wide = store.as_of(required_series(cfg), vintage)
    if wide.empty:
        return {"drivers": pd.DataFrame(), "indicators": pd.DataFrame(),
                "inputs": pd.DataFrame(), "vintage": vintage}
    inputs, scores = indicator_frames(cfg, wide)
    return {
        "drivers": driver_scores(cfg, wide),
        "indicators": scores,
        "inputs": inputs,
        "vintage": vintage,
        "config_hash": config_hash(cfg),
    }
