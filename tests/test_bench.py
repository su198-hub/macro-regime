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


def test_horizons_natures_and_sources_are_known_values():
    for r in ROWS:
        assert r.get("horizon") in bn.HORIZON_ORDER, f"{r['id']} has horizon {r.get('horizon')!r}"
        assert r.get("nature") in bn.NATURE_ORDER, f"{r['id']} has nature {r.get('nature')!r}"
        assert r.get("where") in bn.SOURCE_LABEL, f"{r['id']} has where {r.get('where')!r}"


def test_nature_is_not_just_horizon_under_another_name():
    """If the two axes always agreed, one of them would be dead weight.

    They are meant to come apart: a valuation can look a decade ahead and still
    mean-revert inside a cycle. Those disagreements are the argument for
    carrying both, so at least one has to exist among the live rows.
    """
    live = [r for r in ROWS if r.get("status") == "in_set"]
    assert bn.flattered(live), "no live row is long-horizon yet cyclical — check the tagging"


def test_every_driver_offers_a_structural_candidate():
    """The set is short of structural content in four drivers out of six.

    A bench that offered none would leave the discussion no way out of it.
    """
    for key, block in bn.drivers(BENCH).items():
        structural = [i for i in (block.get("items") or [])
                      if i.get("status") == "candidate" and i.get("nature") == "structural"]
        assert structural, f"{key} offers no structural candidate"


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


def test_each_driver_offers_a_short_readable_list():
    """Enough to choose from, few enough to read aloud in a meeting."""
    for key, block in bn.drivers(BENCH).items():
        candidates = [i for i in (block.get("items") or []) if i.get("status") == "candidate"]
        assert 5 <= len(candidates) <= 8, f"{key} offers {len(candidates)} candidates"


def test_summary_reports_the_structural_share_and_the_empty_drivers():
    text = bn.summary(BENCH, set(bn.in_set_ids(BENCH)))
    assert "structural" in text
    assert "no structural content at all in:" in text, \
        "the live set has drivers with no structural content; the summary should say so"


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
