"""Checks on the indicator bench.

The bench claims to show the set we actually score alongside the candidates. If
it drifts from config/indicators_v2.yml, the presented model, it stops being
evidence and becomes a second opinion, so the first test here is that the two
agree.
"""

import pytest

from src import bench as bn
from src.drivers import load_config

BENCH = bn.load("config/bench.yml")
CFG = load_config("config/indicators_v2.yml")
ROWS = bn.rows(BENCH)
LIVE = {i["id"] for d in CFG["drivers"].values() for i in d["indicators"]}
V1 = {i["id"] for d in load_config("config/indicators.yml")["drivers"].values()
      for i in d["indicators"]}


def test_bench_ids_are_unique():
    ids = [r["id"] for r in ROWS]
    assert sorted(ids) == sorted(set(ids)), "an id appears twice in bench.yml"


def test_every_scored_indicator_appears_on_the_bench_as_in_set():
    marked = set(bn.in_set_ids(BENCH))
    assert marked == LIVE, (
        f"on the bench but not scored: {sorted(marked - LIVE)}; "
        f"scored but missing from the bench: {sorted(LIVE - marked)}")


def test_v1_rows_are_scored_in_v1_and_not_in_v2():
    """Filed as v1 means exactly that, or the Parked filter misleads."""
    marked = {r["id"] for r in ROWS if r.get("status") == "v1"}
    assert marked <= V1 - LIVE, sorted(marked - (V1 - LIVE))


def test_nothing_v1_scored_went_missing():
    """Two v1 rows are the same measure as a v2 row, so they merged into it."""
    merged = {"locked_in_refinancing", "capacity_utilisation"}
    assert V1 - {r["id"] for r in ROWS} == merged


def test_the_drivers_are_the_presented_models():
    assert list(bn.drivers(BENCH)) == list(CFG["drivers"])
    for key, block in bn.drivers(BENCH).items():
        assert block["label"] == CFG["drivers"][key]["label"]


def test_horizons_agree_with_the_presented_model():
    h = CFG["meta"]["horizons"]
    for r in ROWS:
        if r.get("status") == "in_set":
            want = next(k for k in h if r["id"] in (h[k] or []))
            assert r["horizon"] == want, r["id"]


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


def test_horizons_natures_kinds_and_sources_are_known_values():
    for r in ROWS:
        assert r.get("horizon") in bn.HORIZON_ORDER, f"{r['id']} has horizon {r.get('horizon')!r}"
        assert r.get("nature") in bn.NATURE_ORDER, f"{r['id']} has nature {r.get('nature')!r}"
        assert r.get("kind") in bn.KIND_ORDER, f"{r['id']} has kind {r.get('kind')!r}"
        assert r.get("where") in bn.SOURCE_LABEL, f"{r['id']} has where {r.get('where')!r}"


def test_every_row_says_how_it_would_be_scored():
    for r in ROWS:
        assert r.get("enters") in bn.ENTERS_ORDER, f"{r['id']} has enters {r.get('enters')!r}"


def test_the_scored_set_now_uses_revisions():
    """The bench was built around an objection: a long-run forecast is dead
    weight at its level. v2's potential block answers it, so the guard fails if
    those rows are dropped without the wording that claims the set reads
    revisions being revisited.
    """
    live = [r for r in ROWS if r.get("status") == "in_set"]
    ids = {r["id"] for r in bn.revisions(live)}
    assert {"cbo_potential_growth_revision", "spf_productivity_revision",
            "capex_revision", "cbo_labour_force_revision"} == ids, sorted(ids)


def test_growth_revisions_say_so_in_their_name():
    """The specific objection: a long-run forecast is dead weight at its level."""
    for r in bn.revisions(bn.by_driver(BENCH, "growth")):
        assert "evision" in r["name"], \
            f"{r['id']} is scored as a revision but its name does not say so"


def test_growth_candidates_answer_the_horizon_and_source_criticism():
    """The two things the demand driver was sent back for.

    What growth is offered has to look past what already happened, and it has
    to come from somewhere the set could disagree with, otherwise the bench
    reproduces the problem. Structural is the rule but not an absolute: a single
    non-structural candidate is allowed, and it must be market-priced.
    """
    cands = [i for i in bn.by_driver(BENCH, "growth") if i.get("status") == "candidate"]
    offered = [c for c in cands if not bn.parked(c)]
    soft = [c for c in offered if c["nature"] != "structural"]
    assert len(soft) <= 1, [c["id"] for c in soft]
    assert all(c["kind"] == "market" for c in soft), [c["id"] for c in soft]
    assert len(bn.kinds_present(cands)) >= 3, \
        f"growth candidates draw on only {bn.kinds_present(cands)}"


def test_the_live_growth_driver_does_not_read_one_source_only():
    """The criticism the bench was built on, and the evidence it was answered."""
    live = [i for i in bn.by_driver(BENCH, "growth") if i.get("status") == "in_set"]
    kinds = bn.kinds_present(live)
    assert "official" in kinds and len(kinds) > 1, kinds


def test_nature_is_not_just_horizon_under_another_name():
    """If the two axes always agreed, one of them would be dead weight.

    They are meant to come apart: a valuation can look a decade ahead and still
    mean-revert inside a cycle. Those disagreements are the argument for
    carrying both, so at least one has to exist among the live rows.
    """
    live = [r for r in ROWS if r.get("status") == "in_set"]
    assert bn.flattered(live), "no live row is long-horizon yet cyclical — check the tagging"


def test_every_driver_offers_a_structural_candidate():
    """A bench that offered none would leave the discussion no way past the cycle."""
    for key, block in bn.drivers(BENCH).items():
        structural = [i for i in (block.get("items") or [])
                      if i.get("status") == "candidate" and not bn.parked(i)
                      and i.get("nature") == "structural"]
        assert structural, f"{key} offers no structural candidate"


def test_external_rows_name_a_vendor():
    """'Needs a new source' is only actionable if it says which."""
    for r in ROWS:
        if r.get("where") == "external":
            assert str(r.get("vendor", "")).strip(), f"{r['id']} is external but names no vendor"


def test_the_shipped_set_passes_its_own_size_rule():
    for block in bn.tally(BENCH, set(bn.in_set_ids(BENCH))):
        assert block["ok"], f"{block['label']} ships with {block['n']} indicators"
        assert block["n"] == block["was"]


def test_the_page_opens_on_a_proposal_that_could_itself_ship():
    """Preselected candidates put a proposal on the table, not an illegal one."""
    for block in bn.tally(BENCH, set(bn.opening_ids(BENCH))):
        assert block["ok"], f"{block['label']} opens with {block['n']} indicators"


def test_the_opening_proposal_reads_as_a_swap_against_what_ships():
    """Nothing is preselected now, and that is the proposal having been adopted.

    Every candidate that opened ticked was taken into the set in September 2026,
    so the page opens on what ships. The invariant still worth holding is that
    if something is put forward again, it reads as a coherent swap rather than
    as a row that quietly disagrees with the set.
    """
    pre, drop = set(bn.preselect_ids(BENCH)), set(bn.predrop_ids(BENCH))
    assert not (pre & set(bn.in_set_ids(BENCH))), "a scored row is marked preselect: true"
    d = bn.diff(BENCH, set(bn.opening_ids(BENCH)))
    assert {r["id"] for r in d["added"]} == pre
    assert {r["id"] for r in d["dropped"]} == drop


def test_what_opens_ticked_is_a_swap_not_a_wishlist():
    """Additions have to be paid for, but not one for one.

    What matters is that no driver leaves the three-to-six rule and that the
    proposal stays recognisable as the same set rather than becoming a wish
    list — so a driver may gain or lose one, and no more.
    """
    for block in bn.tally(BENCH, set(bn.opening_ids(BENCH))):
        assert abs(block["n"] - block["was"]) <= 1, (
            f"{block['label']} opens at {block['n']} against {block['was']} as it "
            f"ships; that is a redesign, not a proposal")


def test_each_driver_offers_a_short_readable_list():
    """Enough to choose from, few enough to read aloud in a meeting.

    Parked rows sit after the candidates and do not count: they are kept so they
    can be put back, not put forward.
    """
    for key, block in bn.drivers(BENCH).items():
        offered = [i for i in (block.get("items") or [])
                   if i.get("status") == "candidate" and not bn.parked(i)]
        assert 2 <= len(offered) <= 8, f"{key} offers {len(offered)} candidates"


def test_only_stance_lacks_structural_content():
    """This test used to assert the gap; it now guards the fix.

    The bench existed because four drivers out of six had no structural content.
    In v2 only monetary stance has none, and that is what it is for: how tight
    policy is right now. The page says so in amber rather than dressing it up,
    and this fails if another driver joins it.
    """
    text = bn.summary(BENCH, set(bn.in_set_ids(BENCH)))
    assert "no structural content at all in: Monetary stance\n" in text, text


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


def test_rows_show_the_scored_set_then_the_proposal_then_the_rest():
    """Among candidates, the ones put forward come first; the live set leads,
    and parked rows come last."""
    for key in bn.drivers(BENCH):
        ranks = [0 if i.get("status") == "in_set" else 3 if bn.parked(i)
                 else 1 if i.get("preselect") else 2
                 for i in bn.display_order(bn.by_driver(BENCH, key))]
        assert ranks == sorted(ranks), f"{key} is out of order: {ranks}"
