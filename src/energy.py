"""Energy tightness: is energy demand outrunning supply?

Scores the indicators in config/energy.yml much as the main model scores its
own: each is centered on its average to 2019, signed so positive means tighter,
and clipped at ±3. Two departures, both because the years to 2019 were
unusually calm for energy. The spread is measured over the full history, so a
series that barely moved before 2020 does not read every later move as an
extreme; and a block maps its weighted mean through tanh(x / 2) rather than
dividing by 2 and clipping, so it approaches ±1 without sticking there. It then
combines the blocks into power tightness, fuels tightness and one energy
tightness number. Also places each month on ASR's two
markers -- demand surging, supply expanding -- to say which of their energy
scenarios it looks like.

Pure functions over a wide frame of stored series; the page and the tests both
call them, and nothing here touches the store or the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.transform import CLIP, TRANSFORMS, to_monthly
from src.drivers import DRIVER_SCALE


def load_config(path: str | Path = "config/energy.yml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def indicators(cfg: dict):
    for block_id, block in cfg["blocks"].items():
        for ind in block["indicators"]:
            yield block_id, ind


def required_series(cfg: dict) -> list[str]:
    out = []
    for _, ind in indicators(cfg):
        out += list(ind["series"])
        out += [ind[k] for k in ("deflator", "denominator") if ind.get(k)]
    out += [c["series"] for c in cfg.get("context", [])]
    return sorted(set(out))


def _monthly(wide: pd.DataFrame, sid: str, end) -> pd.Series:
    if sid not in wide.columns:
        return pd.Series(dtype=float)
    s = wide[sid].dropna()
    # Quarterly samples (860M) hold for one quarter past their last reading.
    return to_monthly(s, end=end, carry_periods=1) if not s.empty else s


def rel_36m_ann(series: list[pd.Series], deflator: pd.Series) -> pd.Series:
    """Three-year change a year in each price relative to the deflator, averaged."""
    parts = []
    for s in series:
        rel = np.log(s / deflator.reindex(s.index))
        parts.append((rel - rel.shift(36)) / 3 * 100)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False) if parts else pd.Series(dtype=float)


def indicator_value(ind: dict, wide: pd.DataFrame, end) -> pd.Series:
    """The indicator in its own units, monthly, before scoring.

    `smooth_months` averages the result over that many months, for series that
    are a fresh forecast each month and swing between releases.
    """
    v = _indicator_value(ind, wide, end)
    n = int(ind.get("smooth_months", 1))
    return v.rolling(n, min_periods=n).mean().dropna() if n > 1 and not v.empty else v


def _indicator_value(ind: dict, wide: pd.DataFrame, end) -> pd.Series:
    series = [_monthly(wide, sid, end) for sid in ind["series"]]
    series = [s for s in series if not s.empty]
    if not series:
        return pd.Series(dtype=float)
    if ind["transform"] == "rel_36m_ann":
        deflator = _monthly(wide, ind["deflator"], end)
        if deflator.empty:
            return pd.Series(dtype=float)
        return rel_36m_ann(series, deflator).dropna()
    s = pd.concat(series, axis=1).mean(axis=1)
    if ind["transform"] == "share":
        # A spending flow against GDP, so it reads the same in 2015 dollars and
        # 2026 dollars. `factor` reconciles units ($ millions over $ billions).
        denom = _monthly(wide, ind["denominator"], end)
        return (s / denom.reindex(s.index) * float(ind.get("factor", 100))).dropna()
    if ind["transform"] == "pct_36m_ann":
        if ind.get("sum_months"):
            n = int(ind["sum_months"])
            s = s.rolling(n, min_periods=n).sum()
        return (((s / s.shift(36)) ** (1 / 3) - 1) * 100).dropna()
    if ind["transform"] == "mean_12m":
        return s.rolling(12, min_periods=12).mean().dropna()
    if ind["transform"] == "dev_5y_same_month_pct":
        # Against the same month in the five years before, as EIA reports
        # inventories, so the seasonal build and draw cancel.
        base = pd.concat([s.shift(12 * k) for k in range(1, 6)], axis=1).mean(axis=1, skipna=False)
        return ((s / base - 1) * 100).dropna()
    return TRANSFORMS[ind["transform"]](s).dropna()


def score(values: pd.Series, center_to: str, min_months: int,
          full_spread: bool = True) -> tuple[pd.Series, dict]:
    """z-score centered on history to `center_to`, spread from all of it.

    The center is what normal looked like before the boom. The spread comes
    from the whole history, so the scale reflects how far the series has
    actually ranged; with `full_spread` off it comes from the same years as
    the center, as in the main model.
    """
    base = values[values.index <= pd.Timestamp(center_to)]
    basis = "average to 2019"
    if len(base) < min_months:
        base, basis = values, "all history (too little before 2020)"
    center = float(base.mean())
    scale = float((values if full_spread else base).std())
    if not scale or np.isnan(scale):
        return pd.Series(np.nan, index=values.index), {"center": center, "scale": scale, "basis": basis}
    z = ((values - center) / scale).clip(-CLIP, CLIP)
    return z, {"center": center, "scale": scale, "basis": basis}


def compute(cfg: dict, wide: pd.DataFrame) -> dict:
    """Everything the page needs: values, scores, blocks, composites, scenario."""
    meta = cfg.get("meta", {})
    # The latest month is set by the scored series alone: context series (the
    # Epoch path) run years into the planned future and would drag it forward.
    scored = [sid for _, ind in indicators(cfg) for sid in ind["series"] if sid in wide]
    end = wide[scored].dropna(how="all").index.max() if scored else None
    values, scores, norms = {}, {}, {}
    for block_id, ind in indicators(cfg):
        key = f"{block_id}::{ind['id']}"
        v = indicator_value(ind, wide, end)
        if v.empty:
            continue
        z, norm = score(v, meta.get("center_to", "2019-12-31"), int(meta.get("min_history_months", 36)),
                        bool(meta.get("full_spread", True)))
        values[key], scores[key], norms[key] = v, z * float(ind.get("sign", 1)), norm
    values = pd.DataFrame(values).sort_index()
    scores = pd.DataFrame(scores).sort_index()
    # Ragged edge: releases land on different days, so the newest month lacks
    # whatever is published later (FRED prices trail the EIA outlook by a
    # month). Each score holds its last reading for up to `carry_months`, so a
    # late release does not move the composite by its absence.
    carry = int(meta.get("carry_months", 2))
    if not scores.empty and end is not None:
        idx = pd.date_range(scores.index.min(), pd.Timestamp(end) + pd.offsets.MonthEnd(0), freq="ME")
        scores = scores.reindex(idx).ffill(limit=carry)

    blocks = {}
    w_all = {b: sum(float(i["weight"]) for i in blk["indicators"]) for b, blk in cfg["blocks"].items()}
    for block_id, block in cfg["blocks"].items():
        cols = [f"{block_id}::{i['id']}" for i in block["indicators"] if f"{block_id}::{i['id']}" in scores]
        if not cols:
            continue
        w = np.array([float(i["weight"]) for i in block["indicators"]
                      if f"{block_id}::{i['id']}" in scores])
        part = scores[cols]
        present = part.notna().to_numpy().astype(float)
        denom = present @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            num = np.nansum(part.to_numpy() * w, axis=1)
            val = np.where(denom > 0, num / denom, np.nan)
        # A block needs at least `min_coverage` of its weight reporting, so
        # early history resting on one long-running indicator is left blank
        # rather than presented as the whole block.
        val = np.where(denom >= float(meta.get("min_coverage", 0.5)) * w_all[block_id], val, np.nan)
        blocks[block_id] = pd.Series(np.tanh(val / DRIVER_SCALE), index=part.index)
    blocks = pd.DataFrame(blocks).sort_index()

    def combine(weights: dict, frame: pd.DataFrame) -> pd.Series:
        cols = [k for k in weights if k in frame]
        if not cols:
            return pd.Series(dtype=float)
        w = pd.Series({k: weights[k] for k in cols}, dtype=float)
        part = frame[cols] * w.apply(np.sign)
        mag = w.abs()
        present = part.notna().mul(mag, axis=1).sum(axis=1)
        out = part.mul(mag, axis=1).sum(axis=1, min_count=1) / present.replace(0, np.nan)
        # Every component must report: power without supply, or energy
        # without fuels, would be a different measure under the same name.
        return out.where(part.notna().all(axis=1)).clip(-1, 1)

    composite = pd.DataFrame(index=blocks.index)
    composite["power"] = combine(cfg["combine"]["power"], blocks)
    if "fuels" in blocks:
        composite["fuels"] = blocks["fuels"]
    composite["energy"] = combine(cfg["combine"]["energy"], composite)

    scenario = pd.Series(
        [scenario_for(d, s, float(meta.get("mid_band", 0.15)))
         for d, s in zip(blocks.get("power_demand", pd.Series(np.nan, blocks.index)),
                         blocks.get("power_supply", pd.Series(np.nan, blocks.index)))],
        index=blocks.index)
    return {"values": values, "scores": scores, "norms": norms, "blocks": blocks,
            "composite": composite, "scenario": scenario}


SCENARIOS = {
    "ai_boom": ("AI Boom", "Demand is surging and supply is expanding to meet it."),
    "energy_first": ("Energy First", "Supply is not keeping up, so energy constrains growth."),
    "current_policies": ("Current Policies", "Neither demand nor supply is far from normal."),
}


def scenario_for(demand: float, supply: float, mid: float = 0.15) -> str | None:
    """Which of ASR's scenarios the two markers point to.

    When demand is surging, the question is whether supply keeps pace with it,
    not merely whether supply is above its own history: within `mid` of demand
    is AI Boom, further behind is Energy First. Without a surge, supply
    shrinking is Energy First too -- in ASR's words, supply constrained, fossil
    fuels critical -- and anything else is Current Policies. Net Zero and High
    Damage are not read from these two markers.
    """
    if demand is None or supply is None or np.isnan(demand) or np.isnan(supply):
        return None
    if demand > mid:
        return "ai_boom" if supply >= demand - mid else "energy_first"
    if supply < -mid:
        return "energy_first"
    return "current_policies"


def split(parts: dict[str, float], threshold: float) -> tuple[str, str] | None:
    """The pair pulling hardest in opposite directions, if both pass `threshold`.

    A score near zero can mean nothing is happening, or two big forces
    canceling. This tells them apart: it returns (tightest, loosest) when one
    part reads at least `threshold` tight and another at least as loose.
    """
    parts = {k: v for k, v in parts.items() if v is not None and not np.isnan(v)}
    if len(parts) < 2:
        return None
    hi, lo = max(parts, key=parts.get), min(parts, key=parts.get)
    return (hi, lo) if parts[hi] >= threshold and parts[lo] <= -threshold else None


def latest_indicator_scores(cfg: dict, scores: pd.DataFrame, block_id: str,
                            min_weight: float = 0.1) -> dict[str, float]:
    """Each indicator's latest score in a block, by its short name."""
    out = {}
    for ind in cfg["blocks"][block_id]["indicators"]:
        key = f"{block_id}::{ind['id']}"
        if float(ind["weight"]) >= min_weight and key in scores and scores[key].notna().any():
            out[ind.get("short", ind["label"])] = float(scores[key].dropna().iloc[-1])
    return out
