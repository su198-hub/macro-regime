"""Signpost geometry. A flipped sign here draws a confident, wrong chart."""

import pandas as pd

PROSE_CFG = {"drivers": {
    "demand": {"label": "Demand", "signpost": {
        "low": {"label": "Weak", "phrase": "demand is running below trend"},
        "high": {"label": "Strong", "phrase": "demand is running above trend"},
        "mid": {"phrase": "demand is close to trend"}}},
    "investment": {"label": "Investment spending", "signpost": {
        "low": {"label": "Low", "phrase": "capex is contracting"},
        "high": {"label": "High", "phrase": "capex is expanding"},
        "mid": {"phrase": "capex is flat"}}},
}}
PROSE_REG = {"driver_salience": {"demand": 1.0, "investment": 1.0},
             "regimes": {"boom": {"label": "Boom",
                                  "archetype": {"demand": 0.6, "investment": 0.6}}}}
import pytest

from src import ui

from src.ui import (cluster, driver_phrase, month_mid, month_start, regime_runs, run_length,
                    objection, track_position, what_moved, why_called,
                    why_not_called)


def test_months_plot_mid_month_not_on_the_next_months_line():
    idx = pd.to_datetime(["2016-12-31", "2017-01-31"])
    assert list(month_mid(idx)) == [pd.Timestamp("2016-12-15"), pd.Timestamp("2017-01-15")]
    assert list(month_start(idx)) == [pd.Timestamp("2016-12-01"), pd.Timestamp("2017-01-01")]


def test_track_position_orders_low_to_high_and_clips():
    assert track_position(-1) < track_position(0) < track_position(1)
    assert track_position(0) == pytest.approx(50.0)
    assert track_position(5) == track_position(1)


def test_reverse_mirrors_about_the_centre():
    assert track_position(0.6, reverse=True) == pytest.approx(100 - track_position(0.6))
    assert track_position(0.6, reverse=True) < 50


def test_cluster_merges_near_neighbours_only():
    groups = cluster([(50.0, "a"), (55.0, "b"), (90.0, "c")], gap=10)
    assert [keys for _, keys in groups] == [["a", "b"], ["c"]]
    assert groups[0][0] == pytest.approx(52.5)


def test_run_length_counts_trailing_repeats():
    assert run_length(pd.Series(["x", "y", "y", "y"])) == 3
    assert run_length(pd.Series([], dtype=object)) == 0


def test_regime_runs_collapse_contiguous_months():
    idx = pd.date_range("2024-01-31", periods=5, freq="ME")
    runs = regime_runs(pd.Series(["a", "a", "b", "b", "a"], index=idx))
    assert runs["regime"].tolist() == ["a", "b", "a"]
    assert runs["start"].iloc[1] == idx[2]
    assert runs["end"].iloc[1] == idx[4]


# ---------- the sentences on the dashboard ----------

def test_driver_phrase_follows_the_score_into_the_right_band():
    assert driver_phrase("demand", 0.8, PROSE_CFG) == "demand is running above trend"
    assert driver_phrase("demand", -0.8, PROSE_CFG) == "demand is running below trend"
    assert driver_phrase("demand", 0.02, PROSE_CFG) == "demand is close to trend"


def test_why_called_names_what_matches_and_what_argues_against():
    row = pd.Series({"demand": 0.05, "investment": 0.62})
    text = why_called(row, PROSE_CFG, PROSE_REG, "boom")
    assert text == "Boom because capex is expanding."
    # Demand is furthest from the archetype, so it is the objection, and it is
    # not listed as a reason for the call.
    assert "demand" not in text
    assert objection(row, PROSE_CFG, PROSE_REG, "boom") == \
        "The main argument against: demand is close to trend."


def test_why_called_marks_a_loose_match_without_arithmetic():
    row = pd.Series({"demand": 0.05, "investment": 0.62})
    text = why_called(row, PROSE_CFG, PROSE_REG, "boom", weak=True)
    assert text.endswith("Overall the match is loose.")
    assert "distance" not in text.lower()


def test_why_not_called_names_the_two_worst_gaps():
    row = pd.Series({"demand": -0.7, "investment": -0.6})
    text = why_not_called(row, PROSE_CFG, PROSE_REG, "boom")
    assert text.startswith("Nearest is Boom, but")
    assert "demand is running below trend" in text and "capex is contracting" in text


def test_what_moved_reports_movers_and_holds_without_repeating_the_driver():
    before = pd.Series({"demand": 0.0, "investment": 0.30})
    now = pd.Series({"demand": 0.40, "investment": 0.31})
    text = what_moved(now, before, PROSE_CFG)
    assert text.startswith("demand is running above trend after rising")
    assert "demand has risen, so demand" not in text
    assert "investment spending are little changed" in text


def test_what_moved_says_what_held_when_nothing_moved():
    before = pd.Series({"demand": 0.20, "investment": 0.30})
    now = pd.Series({"demand": 0.21, "investment": 0.31})
    text = what_moved(now, before, PROSE_CFG)
    assert text.startswith("little has moved")
    assert "demand" in text and "investment spending" in text



def test_source_lines_need_nothing_from_the_store_but_its_connection():
    """The store is a cached resource: a hosted session that began before a
    deploy holds an instance of the old Store class. Page code that called a
    method added since raised a TypeError on the live app, so the vendor lookup
    asks only the connection, which every version has."""
    import duckdb
    from views.common import model_vendors

    class OldStore:
        def __init__(self):
            self.con = duckdb.connect()
            self.con.execute("CREATE TABLE series_meta (series_id VARCHAR, source VARCHAR)")
            self.con.execute("INSERT INTO series_meta VALUES ('A','macrobond'),('B','bloomberg'),('C','energy-eia')")

    assert model_vendors(OldStore(), ["A", "B"]) == {"macrobond", "bloomberg"}


def test_rich_bolds_marked_text_and_escapes_the_rest():
    from src.ui import rich
    assert rich("**Changes in potential growth**: a <b>raw</b> tag") == \
        "<b>Changes in potential growth</b>: a &lt;b&gt;raw&lt;/b&gt; tag"
    assert rich("no markers") == "no markers"


def test_cached_results_are_keyed_on_what_the_config_says(tmp_path):
    """Renaming a signpost must change the cache key, or results cached under
    the old config are served beside the new one after a deploy."""
    from views.common import config_stamp
    ind, reg = tmp_path / "ind.yml", tmp_path / "reg.yml"
    ind.write_text("drivers: {fiscal_sustainability: {}}\n")
    reg.write_text("regimes: {}\n")
    before = config_stamp(str(ind), str(reg))
    ind.write_text("drivers: {fiscal: {}}\n")
    assert config_stamp(str(ind), str(reg)) != before


def test_breakdown_table_heads_each_block_with_its_share_and_part():
    import pandas as pd
    from src import ui
    from src.drivers import driver_groups
    cfg = {"drivers": {"growth": {
        "groups": {"cycle": {"label": "Cycle"}, "potential": {"label": "Potential growth"}},
        "indicators": [
            {"id": i, "group": g, "label": f"Indicator {i}", "source": {"series": i},
             "transform": "level", "normalize": {"method": "gap", "center": 0.0, "scale": 1.0},
             "direction": 1, "weight": w}
            for i, g, w in (("a", "cycle", 0.75), ("c", "potential", 0.25))]}}}
    parts = pd.DataFrame({"id": ["a", "c"], "group": ["cycle", "potential"],
                          "value": [1.0, 2.0], "score": [1.0, 2.0], "weight": [0.75, 0.25],
                          "share": [0.75, 0.25], "contribution": [0.375, 0.25]})
    out = ui.breakdown_table(parts, cfg["drivers"]["growth"]["indicators"], 0.625,
                             groups=driver_groups(cfg, "growth"))
    assert (out.index("Cycle") < out.index("Indicator a") < out.index("Potential growth")
            < out.index("Indicator c"))
    assert "<b>75%</b>" in out and "<b>25%</b>" in out


def test_a_scenario_can_be_left_off_the_signpost_chart():
    import pandas as pd
    from src import ui
    reg = {"regimes": {
        "up": {"label": "Up", "short": "UPX", "color": "#008300", "archetype": {"d": 0.5}},
        "mid": {"label": "Middle", "short": "MIDX", "color": "#4a5a7a", "archetype": {"d": 0.0},
                "on_signpost_chart": False}}}
    cfg = {"drivers": {"d": {"label": "D", "indicators": []}}}
    idx = pd.to_datetime(["2026-07-31"])
    html = ui.signpost_html(pd.DataFrame({"d": [0.2]}, index=idx), cfg, reg, idx[0], None)
    assert "UPX" in html and "MIDX" not in html


def test_ordinal_slots_keep_neutral_central_and_order_each_side_evenly():
    from src import ui
    slots, place = ui.ordinal_slots({"hl": -0.13, "stf": -0.29, "gld": 0.18, "hghi": 0.09})
    assert [k[0] for _, k in slots] == ["stf", "hl", "hghi", "gld"]
    assert place(0.0) == 50.0                                   # neutral in the middle
    assert all(p < 50 for p, k in slots[:2]) and all(p > 50 for p, k in slots[2:])
    left = [p for p, _ in slots[:2]]; right = [p for p, _ in slots[2:]]
    assert abs((left[1] - left[0]) - (50 - left[1])) < 1e-9    # evenly spaced within a side
    assert abs((right[1] - right[0]) - (right[0] - 50)) < 1e-9
    assert place(0.09) == slots[2][0]                           # on a scenario: at its slot
    assert slots[2][0] < place(0.12) < slots[3][0]             # between two: between their slots
    assert place(-1.0) == ui.TRACK_PAD and place(1.0) == 100 - ui.TRACK_PAD
    # A row with every scenario below zero still has neutral in the middle.
    slots_n, place_n = ui.ordinal_slots({"a": -0.4, "b": -0.2})
    assert place_n(0.0) == 50.0 and all(p < 50 for p, _ in slots_n)
    # Reversed rows mirror: the higher score on the left, neutral still central.
    slots_r, place_r = ui.ordinal_slots({"a": 0.10, "b": 0.30, "c": -0.4}, reverse=True)
    assert [k for _, k in slots_r] == [["b"], ["a"], ["c"]] and place_r(0.0) == 50.0


def test_why_called_can_say_the_signposts_lean_toward_a_challenger():
    from src import ui
    import pandas as pd
    reg = {"regimes": {"hot": {"label": "Hot", "archetype": {"d": 0.6}},
                       "cold": {"label": "Cold", "archetype": {"d": -0.6}}}}
    cfg = {"drivers": {"d": {"label": "Demand", "indicators": [],
                             "signpost": {"low": {"phrase": "demand is weak"},
                                          "high": {"phrase": "demand is strong"}}}}}
    row = pd.Series({"d": 0.55})
    assert ui.why_called(row, cfg, reg, "hot", leaning=True).startswith("Leaning toward hot")
    assert ui.why_called(row, cfg, reg, "hot").startswith("Hot")


def test_radar_point_runs_from_centre_to_outer_ring():
    """-1 sits at the centre, 0 on the middle ring, +1 on the outer one, and
    spoke 0 points straight up: a flipped sign would draw a regime inside out."""
    cx, cy = 150.0, 126.0
    assert ui.radar_point(-1.0, 0, 5, cx, cy) == pytest.approx((cx, cy))
    x, y = ui.radar_point(0.0, 0, 5, cx, cy)
    assert x == pytest.approx(cx) and y == pytest.approx(cy - ui.RADAR_R / 2)
    x, y = ui.radar_point(1.0, 0, 5, cx, cy)
    assert y == pytest.approx(cy - ui.RADAR_R)
    # Scores past the cap stay on the outer ring; spoke 1 is to the right.
    assert ui.radar_point(3.0, 1, 5, cx, cy) == pytest.approx(ui.radar_point(1.0, 1, 5, cx, cy))
    assert ui.radar_point(1.0, 1, 5, cx, cy)[0] > cx


def test_radar_skips_a_series_with_a_missing_driver():
    labels = {"a": ("A", "High"), "b": ("B", "High"), "c": ("C", "High")}
    full = {"name": "Full", "values": {"a": 0.1, "b": -0.2, "c": 0.3}, "color": "#123456"}
    gap = {"name": "Gap", "values": {"a": 0.1, "b": float("nan"), "c": 0.3}, "color": "#654321"}
    svg = ui.radar_svg(["a", "b", "c"], labels, [full, gap], title="T")
    assert "#123456" in svg and "#654321" not in svg
    assert "Full · B: −0.20" in svg
