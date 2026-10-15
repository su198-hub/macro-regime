"""Tests for the country registry.

The failure to guard against is a menu that offers a country the app cannot
score: the page would render empty or, worse, show another country's numbers
under the wrong name.
"""

import pytest
import yaml

from src import countries as ctry
from src.drivers import load_config, required_series
from src.regimes import load_regimes

REG = ctry.load_registry()


def test_the_default_country_is_live_and_resolvable():
    code = ctry.default_code(REG)
    assert code in ctry.live_codes(REG)
    here = ctry.resolve(REG, code)
    assert here["code"] == code and here["label"]


def test_every_live_country_has_loadable_config():
    for code in ctry.live_codes(REG):
        here = ctry.resolve(REG, code)
        cfg, reg = load_config(here["indicators"]), load_regimes(here["regimes"])
        assert cfg["drivers"], f"{code}: no drivers"
        assert reg["regimes"], f"{code}: no regimes"
        # Every driver a regime scores on must exist in the indicator set.
        needed = {d for spec in reg["regimes"].values() for d in spec["archetype"]}
        assert needed <= set(cfg["drivers"]), f"{code}: archetypes name unknown drivers"
        with open(here["sources"]) as f:
            sources = yaml.safe_load(f) or {}
        assert sources, f"{code}: empty sources file"


def test_planned_countries_are_not_selectable():
    for spec in ctry.planned(REG):
        assert spec["code"] not in ctry.live_codes(REG)
        assert spec.get("label"), f"{spec['code']}: planned countries still need a label"


def test_series_names_are_unique_across_countries():
    # One store holds every country, so two countries must not use one name for
    # two different series.
    seen = {}
    for code in ctry.live_codes(REG):
        for sid in required_series(load_config(ctry.resolve(REG, code)["indicators"])):
            assert seen.setdefault(sid, code) == code, \
                f"{sid} is claimed by both {seen[sid]} and {code}"


def test_asking_for_a_planned_country_falls_back_in_the_app_and_raises_in_a_script():
    planned = [p["code"] for p in ctry.planned(REG)]
    if not planned:
        pytest.skip("no planned countries in the registry")
    assert ctry.resolve(REG, planned[0])["code"] == ctry.default_code(REG)
    with pytest.raises(ValueError, match="not live"):
        ctry.resolve(REG, planned[0], strict=True)


def test_an_unknown_country_is_an_error_not_a_silent_default():
    with pytest.raises(KeyError):
        ctry.resolve(REG, "atlantis", strict=True)


def test_a_registry_with_no_countries_is_rejected(tmp_path):
    path = tmp_path / "countries.yml"
    path.write_text("default: us\ncountries: {}\n")
    with pytest.raises(ValueError):
        ctry.load_registry(path)


def test_frameworks_default_to_the_first_listed_and_fall_back_when_unknown():
    here = ctry.resolve(REG, "us")
    assert here["framework"] == here["default_framework"] == "v2"
    assert (here["indicators"], here["regimes"]) == ("config/indicators_v2.yml", "config/regimes_v2.yml")
    v1 = ctry.resolve(REG, "us", framework="v1")
    assert (v1["indicators"], v1["regimes"]) == ("config/indicators.yml", "config/regimes.yml")
    assert ctry.resolve(REG, "us", framework="nonsense")["framework"] == "v2"
    assert set(here["all_indicators"]) == {"config/indicators.yml", "config/indicators_v2.yml"}


def test_every_framework_loads_and_its_scenarios_cover_its_signposts():
    for code in ctry.live_codes(REG):
        for fw in ctry.frameworks(REG, code):
            here = ctry.resolve(REG, code, framework=fw)
            cfg, reg = load_config(here["indicators"]), load_regimes(here["regimes"])
            for name, spec in reg["regimes"].items():
                assert set(spec["archetype"]) == set(cfg["drivers"]), f"{code}/{fw}/{name}"
            assert set(reg["driver_salience"]) == set(cfg["drivers"]), f"{code}/{fw}"


def test_a_country_whose_frameworks_are_all_missing_falls_back_to_its_own_files(tmp_path):
    for name in ("ind.yml", "reg.yml", "src.yml"):
        (tmp_path / name).write_text("{}\n")
    reg = {"default": "xx", "countries": {"xx": {
        "label": "X", "status": "live", "indicators": "ind.yml", "regimes": "reg.yml", "sources": "src.yml",
        "frameworks": {"v9": {"label": "gone", "indicators": "missing.yml", "regimes": "missing.yml"}}}}}
    here = ctry.resolve(reg, "xx", root=tmp_path)
    assert here["indicators"].endswith("ind.yml") and here["regimes"].endswith("reg.yml")
    assert len(here["frameworks"]) == 1
