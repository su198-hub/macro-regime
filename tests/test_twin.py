"""The twin: a config built from the live one, and the pieces it needs.

The comparison on /twin only means something if the twin differs from the live
model exactly where config/twin.yml says and nowhere else. Most of these tests
guard that.
"""

import datetime as dt
import types

import numpy as np
import pandas as pd
import pytest
import yaml

from src import twin as tw
from src import ui
from src.drivers import load_config, required_series
from src.transform import TRANSFORMS, diff_36m_ann, off_target_share_36m, vol_36m

LIVE = load_config("config/indicators.yml")
OVERLAY = tw.load_overlay()
TWIN = tw.build_config(LIVE, OVERLAY)
SOURCES = yaml.safe_load(open("config/sources.yml"))


def ids(cfg):
    return {i["id"] for d in cfg["drivers"].values() for i in d["indicators"]}


def test_building_the_twin_leaves_the_live_config_untouched():
    before = yaml.safe_dump(load_config("config/indicators.yml"), sort_keys=True)
    tw.build_config(LIVE, OVERLAY)
    assert yaml.safe_dump(LIVE, sort_keys=True) == before


def test_the_twin_swaps_exactly_what_twin_yml_lists():
    out, into, changed = tw.swapped_ids(OVERLAY)
    assert ids(TWIN) == (ids(LIVE) - out) | into
    assert changed <= ids(TWIN)


def test_a_new_indicator_takes_the_weight_of_the_one_it_replaces():
    live_w = {i["id"]: i["weight"] for d in LIVE["drivers"].values() for i in d["indicators"]}
    twin_w = {i["id"]: i["weight"] for d in TWIN["drivers"].values() for i in d["indicators"]}
    for swaps in OVERLAY["swaps"].values():
        for s in swaps:
            assert twin_w[s["in"]["id"]] == live_w[s["out"]], s["in"]["id"]


def test_everything_not_swapped_is_identical():
    """Otherwise a gap between the two calls could come from anywhere."""
    out, into, changed = tw.swapped_ids(OVERLAY)
    live = {i["id"]: i for d in LIVE["drivers"].values() for i in d["indicators"]}
    for d in TWIN["drivers"].values():
        for i in d["indicators"]:
            if i["id"] in into or i["id"] in changed:
                continue
            assert i == live[i["id"]], i["id"]
    assert {k: v for k, v in TWIN.items() if k not in ("drivers", "meta")} == \
           {k: v for k, v in LIVE.items() if k not in ("drivers", "meta")}


@pytest.mark.parametrize("name", list(LIVE["drivers"]))
def test_twin_drivers_keep_the_live_rules(name):
    inds = TWIN["drivers"][name]["indicators"]
    assert 3 <= len(inds) <= 6
    assert sum(float(i["weight"]) for i in inds) == pytest.approx(1.0)


def test_every_twin_indicator_is_tagged_with_one_horizon():
    tagged = [i for ids_ in TWIN["meta"]["horizons"].values() for i in (ids_ or [])]
    assert sorted(tagged) == sorted(set(tagged))
    assert set(tagged) == ids(TWIN)


def test_every_twin_series_has_a_source_and_every_transform_exists():
    missing = [s for s in required_series(TWIN) if s not in SOURCES["series"]]
    assert not missing, missing
    for d in TWIN["drivers"].values():
        for i in d["indicators"]:
            assert i.get("transform", "level") in TRANSFORMS, i["id"]
            assert i.get("transform", "level") in ui.TRANSFORM_TEXT, i["id"]


def test_derived_series_have_a_builder():
    for sid, entry in SOURCES["series"].items():
        if entry.get("use") == "derived":
            assert sid in tw.DERIVED, sid


def test_a_bad_overlay_names_the_problem():
    with pytest.raises(KeyError, match="does not have"):
        tw.build_config(LIVE, {"swaps": {"demand": [{"out": "nonexistent", "in": {"id": "x"}}]}})


# ---------- transforms ----------

def monthly(values, start="2000-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"),
                     dtype=float)


def test_three_year_change_is_per_year():
    s = monthly(np.arange(60.0))           # rises 1 a month, 36 over three years
    assert diff_36m_ann(s).dropna().iloc[0] == pytest.approx(12.0)


def test_off_target_share_counts_both_directions():
    # 2% inflation for 4 years, then 5% for 3 years: the last 36 months are all off.
    idx = np.concatenate([100 * 1.02 ** (np.arange(48) / 12),
                          100 * 1.02 ** 4 * 1.05 ** (np.arange(1, 49) / 12)])
    share = off_target_share_36m(monthly(idx))
    assert share.iloc[-1] == pytest.approx(100.0)
    assert share.dropna().iloc[0] == pytest.approx(0.0)


def test_inflation_volatility_is_zero_when_inflation_is_steady():
    idx = 100 * 1.02 ** (np.arange(72) / 12)
    assert vol_36m(monthly(idx)).dropna().abs().max() < 1e-9


# ---------- revisions ----------

def vintage(stamp, values_by_year):
    dates = [dt.datetime(y, 6, 30) for y in values_by_year]
    return types.SimpleNamespace(revision_time_stamp=stamp, dates=dates,
                                 values=list(values_by_year.values()))


class FakeApi:
    def __init__(self, vintages):
        self._v = vintages

    def get_all_vintage_series(self, code):
        return types.SimpleNamespace(series=self._v)


def test_growth_revision_ignores_a_rebasing():
    """A rebasing scales every year by the same factor. The level jumps; the
    growth rate, which is what the twin scores, must not move at all."""
    base = {y: 100 * 1.02 ** (y - 2020) for y in range(2020, 2031)}
    rebased = {y: v * 1.12 for y, v in base.items()}
    api = FakeApi([vintage(dt.datetime(2020, 1, 1), base),
                   vintage(dt.datetime(2020, 8, 1), rebased)])
    df = tw.growth_revision(api, "x", "REV")
    assert len(df) == 1
    assert df["value"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_growth_revision_sees_a_real_upgrade():
    slow = {y: 100 * 1.02 ** (y - 2020) for y in range(2020, 2031)}
    fast = {y: 100 * 1.025 ** (y - 2020) for y in range(2020, 2031)}
    api = FakeApi([vintage(dt.datetime(2020, 1, 1), slow),
                   vintage(dt.datetime(2020, 8, 1), fast)])
    assert tw.growth_revision(api, "x", "REV")["value"].iloc[0] == pytest.approx(0.5, abs=1e-6)


def test_a_revision_is_dated_the_day_it_was_published():
    base = {y: 100.0 for y in range(2020, 2031)}
    up = {y: 101.0 for y in range(2020, 2031)}
    api = FakeApi([vintage(dt.datetime(2020, 1, 1), base),
                   vintage(dt.datetime(2020, 8, 15), up)])
    df = tw.level_revision(api, "x", "REV")
    assert df["observation_date"].iloc[0] == df["vintage_date"].iloc[0] == dt.date(2020, 8, 15)
    assert df["value"].iloc[0] == pytest.approx(100 * np.log(1.01))


def test_the_twin_page_explains_itself_when_there_is_no_store(tmp_path, monkeypatch):
    """CI and the hosted app have no twin store; the page must say how to build
    one rather than fail."""
    import os

    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("MACRO_REGIME_TWIN_DB", str(tmp_path / "absent.duckdb"))
    page = os.path.join(os.path.dirname(os.path.dirname(__file__)), "views", "twin.py")
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert "twin-build" in "".join(h.proto.body for h in at.get("html"))


def test_the_twin_page_is_hidden_from_the_nav():
    """Reachable by its link only, so the presentation never shows it."""
    src = open("app.py", encoding="utf-8").read()
    line = next(l for l in src.splitlines() if "views/twin.py" in l)
    assert 'visibility="hidden"' in line


def test_undated_vintages_are_skipped():
    """The oldest Macrobond copy has no timestamp; a revision needs two dates."""
    api = FakeApi([vintage(None, {2020: 1.0, 2025: 1.0}),
                   vintage(dt.datetime(2020, 1, 1), {2020: 1.0, 2025: 1.0})])
    assert tw.level_revision(api, "x", "REV").empty
