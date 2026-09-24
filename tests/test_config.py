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


def test_every_transform_has_wording_for_the_dashboard():
    """Without this the table prints the function name, such as dev_5y_pct."""
    missing = [t for t in TRANSFORMS if t not in ui.TRANSFORM_TEXT]
    assert not missing, f"no TRANSFORM_TEXT for {missing}"


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


def test_long_horizon_inputs_are_never_raw_levels_by_accident():
    """A level against a fixed center can end up scoring which decade it is.

    A long-horizon input must therefore be a change, a deviation from its own
    trend, or a spread — or else say in config that its level is the point, as
    an inflation expectation measured against the target is. The flag is there
    to make that a decision rather than an oversight.

    A DERIVED series is the third case. Every one of them is built as a
    difference — a revision between consecutive projections, or a projected
    stock against the same month a year earlier — so reading it at `level`
    already reads a change. It is stationary by construction, and differencing
    it again would score the change in the change. Which series those are is
    read from sources.yml rather than guessed from the name.
    """
    import yaml
    derived = {k for k, v in (yaml.safe_load(open("config/sources.yml"))["series"] or {}).items()
               if (v or {}).get("use") == "derived"}
    horizon = ui.horizons(CFG)
    # Dispersion and share-of-time statistics belong here too: a rolling
    # standard deviation, or a count of months spent off target, is already
    # detrended and cannot score which decade it is the way a raw level can.
    changes = {"yoy_pct", "diff_12m", "dev_5y_pct", "pct_change_5y_ann",
               "pct_change_3m_ann", "diff_36m_ann", "vol_36m", "off_target_share_36m"}
    for d in DRIVERS.values():
        for i in d["indicators"]:
            if horizon.get(i["id"]) != "long":
                continue
            transform = i.get("transform", "level")
            is_spread = "-" in i["source"].get("expr", "")
            already_a_revision = any(
                s in derived for s in i["source"].values() if isinstance(s, str))
            assert (transform in changes or is_spread or i.get("absolute_level")
                    or already_a_revision), (
                f"{i['id']} is long-horizon and enters as a raw level: difference it, "
                f"or set absolute_level: true and say why in its reason")
