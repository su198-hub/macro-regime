"""Tests for session-local overrides.

The failure that matters here is silent leakage: an override that mutates the
shared config dict would change what every other reader sees, and nothing on
the page would say so.
"""

import copy

import pytest

from src import overrides as ovr
from src.drivers import load_config
from src.regimes import load_regimes

CFG = load_config("config/indicators.yml")
REG = load_regimes("config/regimes.yml")
FIRST_DRIVER = next(iter(CFG["drivers"]))
FIRST_IND = CFG["drivers"][FIRST_DRIVER]["indicators"][0]
KEY = ovr.key(FIRST_DRIVER, FIRST_IND["id"])


def test_applying_overrides_never_touches_the_shared_config():
    before = copy.deepcopy(CFG), copy.deepcopy(REG)
    cfg, reg = ovr.apply(CFG, REG, {ovr.WEIGHTS: {KEY: 0.99},
                                    ovr.SALIENCE: {FIRST_DRIVER: 2.0}})
    assert (CFG, REG) == before, "overrides mutated the module-level config"
    assert cfg["drivers"][FIRST_DRIVER]["indicators"][0]["weight"] == 0.99
    assert reg["driver_salience"][FIRST_DRIVER] == 2.0


def test_no_overrides_returns_the_same_objects_untouched():
    cfg, reg = ovr.apply(CFG, REG, None)
    assert cfg is CFG and reg is REG
    assert ovr.count(ovr.clean({}, CFG, REG)) == 0


def test_setting_a_value_back_to_its_default_clears_it():
    default = float(FIRST_IND["weight"])
    cleaned = ovr.clean({ovr.WEIGHTS: {KEY: default}}, CFG, REG)
    assert ovr.count(cleaned) == 0, "an unchanged value counted as an override"


def test_out_of_range_values_are_clamped_not_rejected():
    cleaned = ovr.clean({ovr.WEIGHTS: {KEY: 99.0},
                         ovr.SALIENCE: {FIRST_DRIVER: -5.0},
                         ovr.SETTINGS: {"temperature": 0.0, "persistence_months": 99}},
                        CFG, REG)
    assert cleaned[ovr.WEIGHTS][KEY] == ovr.WEIGHT_LIMITS[1]
    assert cleaned[ovr.SALIENCE][FIRST_DRIVER] == ovr.SALIENCE_LIMITS[0]
    assert cleaned[ovr.SETTINGS]["temperature"] == ovr.SETTING_LIMITS["temperature"][0]
    assert cleaned[ovr.SETTINGS]["persistence_months"] == ovr.SETTING_LIMITS["persistence_months"][1]


def test_a_scale_can_never_reach_zero_because_it_divides():
    key = next(k for k, v in ovr.indicator_defaults(CFG).items() if v["method"] == "gap")
    cleaned = ovr.clean({ovr.SCALES: {key: 0.0}}, CFG, REG)
    assert cleaned[ovr.SCALES][key] > 0


def test_centers_are_ignored_for_zscore_indicators():
    zkeys = [k for k, v in ovr.indicator_defaults(CFG).items() if v["method"] != "gap"]
    if not zkeys:
        pytest.skip("no z-scored indicators in config")
    cleaned = ovr.clean({ovr.CENTERS: {zkeys[0]: 5.0}}, CFG, REG)
    assert ovr.count(cleaned) == 0


def test_unknown_keys_are_dropped():
    cleaned = ovr.clean({ovr.WEIGHTS: {"nope::nope": 0.5},
                         ovr.SALIENCE: {"not_a_driver": 1.0},
                         ovr.SETTINGS: {"not_a_setting": 3}}, CFG, REG)
    assert ovr.count(cleaned) == 0


def test_the_gate_can_be_switched_off_but_not_to_nonsense():
    off = ovr.clean({ovr.SETTINGS: {ovr.GATE_KEY: "none"}}, CFG, REG)
    assert off[ovr.SETTINGS][ovr.GATE_KEY] == "none"
    junk = ovr.clean({ovr.SETTINGS: {ovr.GATE_KEY: "always"}}, CFG, REG)
    assert ovr.count(junk) == 0


def test_signature_round_trips_and_is_stable_across_key_order():
    a = ovr.clean({ovr.WEIGHTS: {KEY: 0.31}, ovr.SALIENCE: {FIRST_DRIVER: 1.4}}, CFG, REG)
    b = ovr.clean({ovr.SALIENCE: {FIRST_DRIVER: 1.4}, ovr.WEIGHTS: {KEY: 0.31}}, CFG, REG)
    assert ovr.signature(a) == ovr.signature(b)
    assert ovr.signature(ovr.parse(ovr.signature(a))) == ovr.signature(a)
    assert ovr.signature(None) == "{}"


def test_changed_center_reaches_the_indicator_spec():
    key = next(k for k, v in ovr.indicator_defaults(CFG).items() if v["method"] == "gap")
    driver, ind_id = key.split("::")
    default = ovr.indicator_defaults(CFG)[key]
    new = default["center"] + default["scale"]
    cfg, _ = ovr.apply(CFG, REG, {ovr.CENTERS: {key: new}})
    spec = next(i for i in cfg["drivers"][driver]["indicators"] if i["id"] == ind_id)
    assert spec["normalize"]["center"] == pytest.approx(new)


def test_every_default_has_a_written_justification():
    rationale = REG.get("rationale") or {}
    assert set(rationale.get("driver_salience", {})) == set(REG["driver_salience"])
    for name in ("temperature", "persistence_months", "min_confidence", "fit_gate",
                 "fit_tolerance"):
        assert rationale.get("settings", {}).get(name), f"no rationale for {name}"
    missing = [k for k, v in ovr.indicator_defaults(CFG).items() if not v["why"]]
    assert not missing, f"indicators with no why: {missing}"
