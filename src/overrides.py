"""Session-local overrides of the model's judgment calls.

The control room lets a reader change the assumptions and watch the call move.
Nothing here touches the files in config/: an override set is a small dict held
in one browser session, applied to a copy of the config on the way into the
engine. Two people can hold different views of the same published data at the
same time, and closing the tab restores the defaults.

What can be changed, and why:

  Indicator weight    How much an indicator counts inside its driver. The most
                      arguable number in the whole config and the cheapest to
                      reason about: weights are relative within a driver.
  Center and scale    Where "normal" sits for an indicator and how many points
                      of score a unit of deviation buys. Only for gap-normalized
                      indicators; a z-score has no center to argue about, it is
                      defined by its own window.
  Driver salience     How much a driver counts when measuring distance to an
                      archetype.
  Engine settings     Temperature, persistence, the confidence floor, and the
                      fit gate with its tolerance.

What cannot, and why:

  Sources, transforms, direction, anchors, carry-forward
                      These define what a number *is*, not how much it counts.
                      Change them and the labels on the dashboard stop being
                      true, so they stay in config where they are reviewable.
  Archetypes          The coordinates of each regime are what the words
                      "goldilocks" and "stagflation" mean here. Editing them
                      privately would leave two readers using the same name for
                      different things.
  The regime set      Adding or removing a regime changes every probability on
                      the page and needs a methodology change to go with it.

Values are clamped rather than rejected: a weight cannot go negative, a scale
cannot reach zero (it divides), and a center cannot wander so far that the
indicator stops measuring anything.
"""

from __future__ import annotations

import copy
import json

WEIGHTS = "weights"
CENTERS = "centers"
SCALES = "scales"
SALIENCE = "salience"
SETTINGS = "settings"
GROUPS = (WEIGHTS, CENTERS, SCALES, SALIENCE, SETTINGS)

WEIGHT_LIMITS = (0.0, 1.0)
SALIENCE_LIMITS = (0.0, 3.0)
SETTING_LIMITS = {
    "temperature": (0.05, 1.00),
    "persistence_months": (1, 12),
    "min_confidence": (0.00, 0.90),
    "fit_tolerance": (1.00, 2.00),
}
GATE_VALUES = ("closer_than_neutral", "none")
GATE_KEY = "fit_gate"


def key(driver: str, indicator_id: str) -> str:
    return f"{driver}::{indicator_id}"


def empty() -> dict:
    return {g: {} for g in GROUPS}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def center_limits(center: float, scale: float) -> tuple[float, float]:
    """Five scale units either side. Beyond that every month clips to ±3."""
    reach = 5.0 * abs(scale)
    return center - reach, center + reach


def scale_limits(scale: float) -> tuple[float, float]:
    """A tenth of the default to five times it. Zero would divide by zero."""
    return 0.1 * abs(scale), 5.0 * abs(scale)


def indicator_defaults(cfg: dict) -> dict:
    """Every indicator's editable defaults, keyed driver::id, in config order."""
    out = {}
    for driver, spec in cfg["drivers"].items():
        total = sum(float(i.get("weight", 0.0)) for i in spec["indicators"]) or 1.0
        for ind in spec["indicators"]:
            norm = ind.get("normalize") or ind.get("normalise") or {}
            weight = float(ind.get("weight", 0.0))
            out[key(driver, ind["id"])] = {
                "driver": driver,
                "driver_label": spec.get("label", driver),
                "id": ind["id"],
                "label": ind.get("label") or ind["id"],
                "weight": weight,
                "share": weight / total,
                "method": norm.get("method", "zscore"),
                "center": float(norm["center"]) if "center" in norm else None,
                "scale": float(norm["scale"]) if "scale" in norm else None,
                "window": norm.get("window"),
                "why": ind.get("why", ""),
                "anchor": bool(ind.get("anchor")),
            }
    return out


def default_value(group: str, name: str, cfg: dict, reg: dict, inds: dict | None = None):
    inds = inds if inds is not None else indicator_defaults(cfg)
    if group == WEIGHTS:
        return inds[name]["weight"]
    if group == CENTERS:
        return inds[name]["center"]
    if group == SCALES:
        return inds[name]["scale"]
    if group == SALIENCE:
        return float(reg["driver_salience"].get(name, 1.0))
    if group == SETTINGS:
        if name == GATE_KEY:
            return str(reg["settings"].get(GATE_KEY, "none"))
        return reg["settings"].get(name)
    raise KeyError(group)


def clean(ov: dict | None, cfg: dict, reg: dict) -> dict:
    """Drop unknown and no-op entries, coerce types, clamp to sane ranges.

    Anything that survives is a real change from the default, so the count of
    entries is what the dashboard reports as "custom settings active".
    """
    out = empty()
    if not ov:
        return out
    inds = indicator_defaults(cfg)

    for name, raw in (ov.get(WEIGHTS) or {}).items():
        if name not in inds:
            continue
        value = _clamp(float(raw), *WEIGHT_LIMITS)
        if abs(value - inds[name]["weight"]) > 1e-9:
            out[WEIGHTS][name] = value

    for group, field, limits in ((CENTERS, "center", center_limits),
                                 (SCALES, "scale", scale_limits)):
        for name, raw in (ov.get(group) or {}).items():
            spec = inds.get(name)
            # Only gap normalization has a center and a scale to argue about.
            if not spec or spec["method"] != "gap" or spec[field] is None:
                continue
            lo, hi = (limits(spec["center"], spec["scale"]) if group == CENTERS
                      else limits(spec["scale"]))
            value = _clamp(float(raw), lo, hi)
            if abs(value - spec[field]) > 1e-9:
                out[group][name] = value

    for name, raw in (ov.get(SALIENCE) or {}).items():
        if name not in reg["driver_salience"]:
            continue
        value = _clamp(float(raw), *SALIENCE_LIMITS)
        if abs(value - float(reg["driver_salience"][name])) > 1e-9:
            out[SALIENCE][name] = value

    for name, raw in (ov.get(SETTINGS) or {}).items():
        if name == GATE_KEY:
            value = str(raw)
            if value in GATE_VALUES and value != str(reg["settings"].get(GATE_KEY, "none")):
                out[SETTINGS][name] = value
            continue
        if name not in SETTING_LIMITS:
            continue
        lo, hi = SETTING_LIMITS[name]
        value = _clamp(float(raw), lo, hi)
        if name == "persistence_months":
            value = int(round(value))
        current = reg["settings"].get(name)
        if current is None or abs(value - float(current)) > 1e-9:
            out[SETTINGS][name] = value

    return out


def apply(cfg: dict, reg: dict, ov: dict | None) -> tuple[dict, dict]:
    """Config and regimes with the overrides applied, as fresh copies."""
    ov = clean(ov, cfg, reg)
    if not count(ov):
        return cfg, reg
    cfg, reg = copy.deepcopy(cfg), copy.deepcopy(reg)

    for driver, spec in cfg["drivers"].items():
        for ind in spec["indicators"]:
            name = key(driver, ind["id"])
            if name in ov[WEIGHTS]:
                ind["weight"] = ov[WEIGHTS][name]
            if name in ov[CENTERS] or name in ov[SCALES]:
                # Config spells it "normalize"; tolerate either spelling.
                field = "normalize" if "normalize" in ind else "normalise"
                norm = ind.setdefault(field, {})
                if name in ov[CENTERS]:
                    norm["center"] = ov[CENTERS][name]
                if name in ov[SCALES]:
                    norm["scale"] = ov[SCALES][name]

    reg["driver_salience"].update(ov[SALIENCE])
    reg["settings"].update(ov[SETTINGS])
    return cfg, reg


def count(ov: dict | None) -> int:
    return sum(len(ov.get(g) or {}) for g in GROUPS) if ov else 0


def signature(ov: dict | None) -> str:
    """Stable string for cache keys: same overrides, same results."""
    if not count(ov):
        return "{}"
    packed = {g: {k: (round(v, 6) if isinstance(v, (int, float)) else v)
                  for k, v in sorted((ov.get(g) or {}).items())}
              for g in GROUPS if ov.get(g)}
    return json.dumps(packed, sort_keys=True, separators=(",", ":"))


def parse(text: str) -> dict:
    """Read a signature back, for sharing a setup as a string."""
    data = json.loads(text) if text else {}
    out = empty()
    for group in GROUPS:
        out[group] = dict(data.get(group) or {})
    return out


SETTING_LABELS = {
    "temperature": "Temperature",
    "persistence_months": "Persistence, months",
    "min_confidence": "Confidence floor",
    "fit_tolerance": "Fit tolerance",
    GATE_KEY: "Fit gate",
}


def summary(ov: dict | None, cfg: dict, reg: dict) -> list[dict]:
    """Rows of what has been changed, for the banner and the control room."""
    ov = clean(ov, cfg, reg)
    inds = indicator_defaults(cfg)
    rows = []
    for group in GROUPS:
        for name, value in (ov.get(group) or {}).items():
            if group in (WEIGHTS, CENTERS, SCALES):
                what = {WEIGHTS: "weight", CENTERS: "center", SCALES: "scale"}[group]
                label = f"{inds[name]['label']} · {what}"
            elif group == SALIENCE:
                label = f"{cfg['drivers'].get(name, {}).get('label', name)} · salience"
            else:
                label = SETTING_LABELS.get(name, name)
            rows.append({"group": group, "name": name, "label": label,
                         "default": default_value(group, name, cfg, reg, inds),
                         "custom": value})
    return rows


def influence(cfg: dict, reg: dict) -> list[dict]:
    """Every indicator ranked by how much of the model it moves.

    Within a driver, weights are renormalized over what reported, so what
    matters is an indicator's share of its driver, not its raw weight. Across
    drivers, salience sets how much that driver counts in the distance. The
    product is a fair first-order ranking; it is only first-order, because
    distance squares the gap, so a driver far from an archetype counts for more
    than this suggests.
    """
    inds = indicator_defaults(cfg)
    total_salience = sum(float(v) for v in reg["driver_salience"].values()) or 1.0
    rows = []
    for name, spec in inds.items():
        sal = float(reg["driver_salience"].get(spec["driver"], 1.0))
        rows.append({**spec, "key": name, "salience": sal,
                     "influence": spec["share"] * sal / total_salience})
    rows.sort(key=lambda r: r["influence"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows
