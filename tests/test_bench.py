"""Checks on the indicator bench.

The bench claims to show the set we actually score alongside the candidates. If
it drifts from config/indicators.yml it stops being evidence and becomes a
second opinion, so the first test here is that the two agree.
"""

import pytest

from src import bench as bn
from src.drivers import load_config

BENCH = bn.load("config/bench.yml")
CFG = load_config("config/indicators.yml")
ROWS = bn.rows(BENCH)
LIVE = {i["id"] for d in CFG["drivers"].values() for i in d["indicators"]}


def test_bench_ids_are_unique():
    ids = [r["id"] for r in ROWS]
    assert sorted(ids) == sorted(set(ids)), "an id appears twice in bench.yml"


def test_every_scored_indicator_appears_on_the_bench_as_in_set():
    marked = set(bn.in_set_ids(BENCH))
    assert marked == LIVE, (
        f"on the bench but not scored: {sorted(marked - LIVE)}; "
        f"scored but missing from the bench: {sorted(LIVE - marked)}")


def test_in_set_rows_sit_under_the_driver_that_scores_them():
    home = {i["id"]: d for d, block in CFG["drivers"].items() for i in block["indicators"]}
    for r in ROWS:
        if r.get("status") == "in_set":
            assert r["driver"] == home[r["id"]], f"{r['id']} is filed under the wrong driver"


def test_every_row_says_what_it_is_and_how_it_is_measured():
    """The page exists to answer both questions; a blank one is worse than no row."""
    for r in ROWS:
        for field in ("name", "refers", "measured"):
            assert str(r.get(field, "")).strip(), f"{r['id']} has no {field}"


def test_horizons_and_sources_are_known_values():
    for r in ROWS:
        assert r.get("horizon") in bn.HORIZON_ORDER, f"{r['id']} has horizon {r.get('horizon')!r}"
        assert r.get("where") in bn.SOURCE_LABEL, f"{r['id']} has where {r.get('where')!r}"


def test_external_rows_name_a_vendor():
    """'Needs a new source' is only actionable if it says which."""
    for r in ROWS:
        if r.get("where") == "external":
            assert str(r.get("vendor", "")).strip(), f"{r['id']} is external but names no vendor"


def test_the_shipped_set_passes_its_own_size_rule():
    """The bench opens on the live set, which must not open in a red state."""
    for block in bn.tally(BENCH, set(bn.in_set_ids(BENCH))):
        assert block["ok"], f"{block['label']} starts with {block['n']} indicators"
        assert block["n"] == block["was"]


def test_every_driver_offers_something_to_choose_from():
    for key, block in bn.drivers(BENCH).items():
        items = block.get("items") or []
        candidates = [i for i in items if i.get("status") == "candidate"]
        assert len(candidates) >= 5, f"{key} offers only {len(candidates)} candidates"


def test_diff_and_summary_report_a_swap():
    live = set(bn.in_set_ids(BENCH))
    candidate = next(r["id"] for r in ROWS if r.get("status") == "candidate")
    dropped = next(iter(live))
    picked = (live - {dropped}) | {candidate}

    d = bn.diff(BENCH, picked)
    assert [r["id"] for r in d["added"]] == [candidate]
    assert [r["id"] for r in d["dropped"]] == [dropped]

    text = bn.summary(BENCH, picked)
    assert "1 added, 1 dropped" in text
    assert bn.by_driver(BENCH, ROWS[0]["driver"])  # driver lookup works


def test_horizon_mix_sums_to_one_for_a_non_empty_selection():
    mix = bn.horizon_mix([r for r in ROWS if r.get("status") == "in_set"])
    assert sum(mix.values()) == pytest.approx(1.0)


def test_empty_selection_does_not_divide_by_zero():
    assert sum(bn.horizon_mix([]).values()) == 0
    assert "0 added" in bn.summary(BENCH, set()) or "CHANGES" in bn.summary(BENCH, set())
