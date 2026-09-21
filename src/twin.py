"""The twin: the live model with the bench's proposed indicators swapped in.

Two jobs.

build_config turns the live indicator config plus config/twin.yml into the twin
config at run time. The twin is never a second copy of indicators.yml, so the
two can only differ where twin.yml says they do, and a change to the live model
reaches the twin without anyone remembering to copy it.

build_store makes the twin's own data store: a copy of the main store with the
twin-only series added. It is a separate file on purpose. Two of those series
come from the Bloomberg terminal, and `publish` refuses to run while the store
it pushes holds any licensed series; putting them in the main store would block
every presentation update. The twin store is never published.
"""

from __future__ import annotations

import copy
import datetime as dt
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OVERLAY = "config/twin.yml"


def default_db() -> str:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return os.environ.get("MACRO_REGIME_TWIN_DB",
                          str(Path(base) / "macro-regime" / "twin.duckdb"))


def load_overlay(path: str = OVERLAY) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------- the config ----------

def build_config(live: dict, overlay: dict) -> dict:
    """Live config plus the swaps and changes in the overlay.

    Each incoming indicator takes the weight of the one it replaces unless the
    overlay sets one, so every driver's weights still sum to what they did.
    """
    cfg = copy.deepcopy(live)
    horizons = cfg.setdefault("meta", {}).setdefault("horizons", {})

    def retag(old_id: str | None, new_id: str, horizon: str | None) -> None:
        for ids in horizons.values():
            if ids and old_id in ids:
                ids.remove(old_id)
        if horizon:
            horizons.setdefault(horizon, []).append(new_id)

    for driver, swaps in (overlay.get("swaps") or {}).items():
        inds = cfg["drivers"][driver]["indicators"]
        for swap in swaps:
            pos = next((k for k, i in enumerate(inds) if i["id"] == swap["out"]), None)
            if pos is None:
                raise KeyError(f"twin.yml swaps out {swap['out']}, which {driver} does not have")
            incoming = copy.deepcopy(swap["in"])
            horizon = incoming.pop("horizon", None)
            incoming.setdefault("weight", inds[pos]["weight"])
            inds[pos] = incoming
            retag(swap["out"], incoming["id"], horizon)

    for ind_id, patch in (overlay.get("changes") or {}).items():
        target = next((i for d in cfg["drivers"].values() for i in d["indicators"]
                       if i["id"] == ind_id), None)
        if target is None:
            raise KeyError(f"twin.yml changes {ind_id}, which the live model does not have")
        target.update(copy.deepcopy(patch))
    return cfg


def swapped_ids(overlay: dict) -> tuple[set[str], set[str], set[str]]:
    """(removed, added, changed) indicator ids, for labelling the comparison."""
    out, into = set(), set()
    for swaps in (overlay.get("swaps") or {}).values():
        for s in swaps:
            out.add(s["out"])
            into.add(s["in"]["id"])
    return out, into, set((overlay.get("changes") or {}))


# ---------- series built from vintages rather than fetched ----------

def _vintages(api, code: str):
    return [v for v in api.get_all_vintage_series(code).series if v.revision_time_stamp]


def _annual(v) -> pd.Series:
    idx = pd.to_datetime(pd.Series(v.dates)).dt.tz_localize(None).values
    s = pd.Series(v.values, index=idx).dropna()
    return s.groupby(s.index.year).mean()


def _frame(series_id: str, rows: list[tuple[dt.date, float]]) -> pd.DataFrame:
    """A revision is known on the day it is published and never revised itself."""
    days = [d for d, _ in rows]
    return pd.DataFrame({"series_id": series_id, "observation_date": days,
                         "vintage_date": days, "value": [x for _, x in rows]})


def growth_revision(api, code: str, series_id: str, ahead: int = 5) -> pd.DataFrame:
    """Change in the projected average growth rate over the next `ahead` years.

    Growth, not level: a rebasing rescales every year of a projection by the
    same factor, which moves the level by 10-14% and the growth rate not at all.
    """
    rows, prev = [], None
    for v in _vintages(api, code):
        stamp, ann = v.revision_time_stamp, _annual(v)
        y0, y1 = stamp.year, stamp.year + ahead
        if prev is not None and all(y in ann.index and y in prev.index for y in (y0, y1)):
            now = ((ann[y1] / ann[y0]) ** (1 / ahead) - 1) * 100
            then = ((prev[y1] / prev[y0]) ** (1 / ahead) - 1) * 100
            rows.append((stamp.date(), now - then))
        prev = ann
    return _frame(series_id, rows)


def level_revision(api, code: str, series_id: str, ahead: int = 5) -> pd.DataFrame:
    """Change in the projected level `ahead` years out, in log percent."""
    rows, prev = [], None
    for v in _vintages(api, code):
        stamp, ann = v.revision_time_stamp, _annual(v)
        y1 = stamp.year + ahead
        if prev is not None and y1 in ann.index and y1 in prev.index:
            rows.append((stamp.date(), 100 * np.log(ann[y1] / prev[y1])))
        prev = ann
    return _frame(series_id, rows)


DERIVED = {
    "CBO_POTGROWTH_REV": lambda api: growth_revision(api, "usfcst1985", "CBO_POTGROWTH_REV"),
    "CBO_LABFORCE_REV": lambda api: level_revision(api, "usfcst0572", "CBO_LABFORCE_REV"),
}


# ---------- the store ----------

def build_store(main_db: str, twin_db: str, live: dict, twin: dict, sources: dict,
                route, open_vendor, log=print) -> None:
    """Copy the main store and add every series the twin needs that it lacks."""
    from src.drivers import required_series
    from src.store import Store

    Path(twin_db).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(main_db, twin_db)
    log(f"Copied {main_db} -> {twin_db}")

    needed = sorted(set(required_series(twin)) - set(required_series(live)))
    store = Store(twin_db)
    vendors: dict = {}
    failed = []
    try:
        for sid in needed:
            try:
                entry = (sources.get("series") or {}).get(sid) or {}
                if entry.get("use") == "derived":
                    if "macrobond" not in vendors:
                        vendors["macrobond"] = open_vendor("macrobond")
                    df = DERIVED[sid](vendors["macrobond"].api)
                    vendor, code = "macrobond", entry.get("derived_from", sid)
                    meta = {"title": entry.get("note", sid).strip()[:120], "units": "",
                            "frequency": "irregular", "has_vintages": True}
                else:
                    vendor, code = route(sid, sources, None)
                    if vendor not in vendors:
                        vendors[vendor] = open_vendor(vendor)
                    df = vendors[vendor].fetch_with_vintages(code).assign(series_id=sid)
                    meta = vendors[vendor].describe(code)
                n = store.upsert_observations(df)
                store.record_meta(sid, vendor, source_code=code, **meta)
                log(f"  {sid:<18} {vendor}:{code:<30} {n:>6} rows")
            except Exception as exc:
                failed.append(sid)
                log(f"  {sid:<18} FAILED: {exc}")
    finally:
        for v in vendors.values():
            if hasattr(v, "close"):
                v.close()
        store.close()
    if failed:
        raise RuntimeError(f"twin store is missing {', '.join(failed)}")
