"""Checks on the shipped configuration itself.

These are cheap and catch the edits that break quietly: an indicator added
without a source, a horizon left untagged, a driver that grew past the point
where a reader can hold it in their head.
"""

import pytest
import yaml

from src import ui
from src.drivers import load_config, required_series
from src.transform import TRANSFORMS

CFG = load_config("config/indicators.yml")
SOURCES = yaml.safe_load(open("config/sources.yml"))
DRIVERS = CFG["drivers"]
ALL_IDS = [i["id"] for d in DRIVERS.values() for i in d["indicators"]]


def test_every_series_has_a_vendor_mapping():
    missing = [s for s in required_series(CFG) if s not in SOURCES["series"]]
    assert not missing, f"no sources.yml entry for {missing}"


def test_every_indicator_names_a_known_transform():
    for d in DRIVERS.values():
        for i in d["indicators"]:
            assert i.get("transform", "level") in TRANSFORMS, i["id"]
            for spec in (i["source"].get("derived") or {}).values():
                assert spec.get("transform", "level") in TRANSFORMS, i["id"]


def test_each_driver_keeps_between_three_and_six_indicators():
    """A reader has to be able to hold the list in their head."""
    for name, d in DRIVERS.items():
        assert 3 <= len(d["indicators"]) <= 6, f"{name} has {len(d['indicators'])}"


def test_weights_within_a_driver_sum_to_one():
    """Only ratios matter to the engine, but the file's convention is to sum to
    one so a reader can read a weight as a share."""
    for name, d in DRIVERS.items():
        total = sum(float(i["weight"]) for i in d["indicators"])
        assert total == pytest.approx(1.0), f"{name} weights sum to {total}"


def test_every_indicator_is_tagged_with_a_horizon_exactly_once():
    listed = (CFG["meta"].get("horizons") or {})
    tagged = [i for ids in listed.values() for i in (ids or [])]
    assert set(listed) <= {"short", "medium", "long"}, f"unknown horizon: {set(listed)}"
    assert sorted(tagged) == sorted(set(tagged)), "an indicator is tagged twice"
    assert set(tagged) == set(ALL_IDS), (
        f"untagged: {sorted(set(ALL_IDS) - set(tagged))}, "
        f"unknown: {sorted(set(tagged) - set(ALL_IDS))}")


def test_every_driver_carries_something_slower_than_the_news():
    """A monthly monitor made only of fast data reports what just happened.

    Each driver keeps at least a third of its weight beyond the short horizon.
    """
    for name in DRIVERS:
        mix = ui.horizon_mix(CFG, name)
        assert mix["medium"] + mix["long"] >= 0.33, f"{name} is {mix['short']:.0%} short-horizon"


def test_long_horizon_inputs_are_never_raw_levels_of_a_trending_series():
    """A level against a fixed center scores which decade it is.

    Long-horizon inputs must be a change, a deviation from their own trend, or
    a spread: an expression that differences two series.
    """
    horizon = ui.horizons(CFG)
    changes = {"yoy_pct", "diff_12m", "dev_5y_pct", "pct_change_5y_ann", "pct_change_3m_ann"}
    for d in DRIVERS.values():
        for i in d["indicators"]:
            if horizon.get(i["id"]) != "long":
                continue
            transform = i.get("transform", "level")
            expr = i["source"].get("expr", "")
            is_spread = "-" in expr
            # A rate or ratio published as a level is already a spread-like
            # quantity; anything else has to be differenced.
            assert transform in changes or is_spread or i["id"] in {
                "breakeven_5y5y", "breakeven_10y", "cleveland_10y", "interest_burden_gdp"
            }, f"{i['id']} is long-horizon but enters as a raw level"
