"""The turning points page: scoring, the regime's past exits, and the productivity check.

No store and no network: each test builds the few series it needs by hand.
"""

import numpy as np
import pandas as pd
import pytest

from src import watch as wt
from src.sources.public import parse_btos

H, S, G = wt.HGHI, wt.STAGFLATION, wt.GOLDILOCKS


def months(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="MS")


def labels(runs: list[tuple[str, int]], start: str = "2000-01") -> pd.Series:
    out = [lab for lab, n in runs for _ in range(n)]
    return pd.Series(out, index=pd.period_range(start, periods=len(out), freq="M"))


def test_the_config_loads_and_every_transform_is_known():
    cfg = wt.load_config()
    wide = pd.DataFrame({sid: np.linspace(100, 200, 400) for sid in wt.required_series(cfg)},
                        index=months("1990-01-01", 400))
    for _, comp in wt.components(cfg):
        assert not wt.component_value(comp, wide).empty, comp["id"]
    roles = {g["role"] for g in cfg["gauges"].values()}
    assert roles == {"alarm", "watch"}
    assert {g["exit"] for g in cfg["gauges"].values()} <= set(cfg["exits"])
    assert {s["exit"] for s in cfg["scenarios"]} <= set(cfg["exits"])


def test_a_quarterly_series_is_dated_to_the_end_of_its_quarter_and_held():
    wide = pd.DataFrame({"Q": [1.0, 2.0, 3.0, 4.0]},
                        index=pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]))
    m = wt.monthly(wide, "Q")
    assert m.index[0] == pd.Period("2025-03", "M")          # Q1 describes January to March
    assert m[pd.Period("2025-05", "M")] == 1.0              # held until Q2 lands in June
    assert m[pd.Period("2025-06", "M")] == 2.0


def test_a_daily_series_is_averaged_over_the_month():
    days = pd.date_range("2025-01-01", "2025-02-28", freq="D")
    wide = pd.DataFrame({"D": np.where(days.month == 1, 2.0, 4.0)}, index=days)
    m = wt.monthly(wide, "D")
    assert list(m) == [2.0, 4.0]


def test_sign_flips_a_score_and_scores_are_capped():
    v = pd.Series(np.r_[np.zeros(200), 100.0], index=pd.period_range("2000-01", periods=201, freq="M"))
    up, _ = wt.score(v, 1)
    down, _ = wt.score(v, -1)
    assert up.iloc[-1] == 3 and down.iloc[-1] == -3


def test_a_realtime_score_uses_only_earlier_months():
    idx = pd.period_range("2000-01", periods=240, freq="M")
    v = pd.Series(np.r_[np.tile([0.0, 1.0], 60), np.tile([10.0, 11.0], 60)], index=idx)
    full, _ = wt.score(v)
    rt, _ = wt.score(v, realtime=True, min_months=24)
    early = idx[100]
    assert abs(rt[early]) < 1.1          # judged against the calm first half only
    assert full[early] < -0.8            # with hindsight the first half reads low


def test_the_alarm_averages_the_tested_gauges_only():
    cfg = {"meta": {"smooth_months": 1}, "gauges": {
        "a": {"role": "alarm", "components": [{"id": "x", "series": ["A"], "transform": "level"}]},
        "b": {"role": "alarm", "components": [{"id": "x", "series": ["B"], "transform": "level"}]},
        "c": {"role": "watch", "components": [{"id": "x", "series": ["C"], "transform": "level"}]}}}
    n = 120
    wide = pd.DataFrame({"A": np.arange(n, dtype=float), "B": np.arange(n, dtype=float),
                         "C": -np.arange(n, dtype=float)}, index=months("2010-01-01", n))
    res = wt.compute(cfg, wide)
    last = res["gauges"].iloc[-1]
    assert last["a"] > 1 and last["c"] < -1
    assert res["alarm"].iloc[-1] == pytest.approx((last["a"] + last["b"]) / 2)


def test_exits_are_finished_spells_with_what_followed():
    lab = labels([(H, 12), (S, 6), (H, 10), (G, 8), (H, 5)])
    ex = wt.exits(lab)
    assert list(ex["next"]) == [S, G]                       # the open spell at the end is not an exit
    assert ex["end"].iloc[0] == pd.Period("2000-12", "M")
    assert list(ex["months"]) == [12, 10]


def test_the_track_record_finds_the_first_warning_and_what_followed_it():
    lab = labels([(H, 24), (S, 6), (H, 24), (G, 6), (H, 6)])
    alarm = pd.Series(-0.5, index=lab.index)
    alarm["2001-08":"2001-12"] = 1.0                         # five months before the stagflation exit
    rec = wt.track_record(alarm, lab, warn=0.3)
    first, second = rec["per_exit"].iloc[0], rec["per_exit"].iloc[1]
    assert first["first"] == pd.Period("2001-08", "M") and first["lead"] == 4
    assert pd.isna(second["lead"])                           # quiet before the goldilocks exit
    assert rec["flagged"] == 5 and rec["then_stagflation"] == 1.0 and rec["then_goldilocks"] == 0.0


def test_the_open_spell_is_left_out_of_the_track_record():
    lab = labels([(H, 24), (S, 6), (H, 12)])
    alarm = pd.Series(-0.5, index=lab.index)
    alarm.iloc[-6:] = 2.0                                    # high now, in the spell still running
    rec = wt.track_record(alarm, lab, warn=0.3)
    assert rec["flagged"] == 0 and rec["months"] == 24


def test_before_exits_averages_the_months_before_each_one():
    lab = labels([(H, 12), (S, 6)])
    frame = pd.DataFrame({"g": np.arange(18, dtype=float)}, index=lab.index)
    be = wt.before_exits(frame, wt.exits(lab), months=6)
    assert be["g"].iloc[0] == pytest.approx(np.mean([5, 6, 7, 8, 9, 10]))   # the exit month itself is out


def greenspan_frame(wage_growth: tuple, price_growth: tuple, margin: tuple) -> pd.DataFrame:
    """Three years monthly: growth rates and the margin step from the first value to the second."""
    idx = months("2020-01-01", 48)
    late = np.arange(48) >= 24

    def level(rates):
        g = np.where(late, rates[1], rates[0]) / 100 / 12
        return 100 * np.exp(np.cumsum(g))

    return pd.DataFrame({"W": level(wage_growth), "P": level(price_growth),
                         "UP": np.where(late, margin[1], margin[0]), "UPR": 100.0}, index=idx)


IDS = {"wages": "W", "core": "P", "unit_profit": "UP", "unit_price": "UPR"}


def test_the_productivity_check_holds_when_pay_and_margins_hold_and_inflation_eases():
    g = wt.greenspan(greenspan_frame((4, 4), (3, 2), (15, 15)), IDS)
    assert g["met"].iloc[-1] == 3


def test_the_productivity_check_fails_on_the_condition_that_broke():
    g = wt.greenspan(greenspan_frame((4, 2), (3, 2), (15, 15)), IDS)
    now = g.loc[pd.Period("2022-06", "M")]
    assert not now["ok_wages"] and now["ok_margin"] and now["ok_core"] and now["met"] == 2
    g = wt.greenspan(greenspan_frame((4, 4), (2, 4), (15, 12)), IDS)
    now = g.loc[pd.Period("2022-06", "M")]
    assert now["ok_wages"] and not now["ok_margin"] and not now["ok_core"]


def test_readings_name_the_alarm_state():
    assert wt.reading(0.5, 0.3) == "warning"
    assert wt.reading(0.1, 0.3) == "building"
    assert wt.reading(-0.4, 0.3) == "quiet"
    assert wt.reading(float("nan"), 0.3) == "not scored"


def test_btos_shares_are_read_by_reference_period_and_gaps_skipped():
    responses = pd.DataFrame([
        ["Question ID", "Question", "Answer ID", "Answer", "202602", "202601", "202526"],
        ["7", "Did this business use AI?", "1", "Yes", "18.2%", "17.7%", "."],
        ["7", "Did this business use AI?", "2", "No", "70.0%", "71.0%", "."],
        ["28", "Will this business use AI?", "1", "Yes", "22.1%", "22.4%", "21.3%"],
    ])
    dates = pd.DataFrame([
        ["Sample Year", "Cycle", "Panel", "Smpdt", "Col Start", "Col End", "Ref Start", "Ref End"],
        [3, 1, 1, 202526, None, None, None, pd.Timestamp("2025-12-28")],
        [None, 1, 2, 202601, None, None, None, pd.Timestamp("2026-01-11")],
        [None, 1, 3, 202602, None, None, None, pd.Timestamp("2026-01-25")],
    ])
    out = parse_btos(responses, dates)
    assert list(out["use_now"]) == [17.7, 18.2]              # the "." cycle is skipped
    assert out["use_now"].index[-1] == pd.Timestamp("2026-01-25")
    assert len(out["use_in_6m"]) == 3


def test_ramp_chart_data_is_read_from_the_escaped_page_payload():
    from src.sources.public import parse_ramp_ai_index
    payload = ('{"adoptionOverall":[{"date_month":"2026-09-01","adoption_rate_pct":55.8},'
               '{"date_month":"2026-10-01","adoption_rate_pct":56.8}],'
               '"spendPerEmployee":[{"date_month":"2026-09-01","median_pepm":15.38,'
               '"top_10_percent_median_pepm":793.25,"top_1_percent_median_pepm":"$undefined"},'
               '{"date_month":"2026-10-01","median_pepm":16.32,"top_10_percent_median_pepm":800.84,'
               '"top_1_percent_median_pepm":10199.2}],"dataVersion":"x"}')
    page = '<script>self.__next_f.push([1,"' + payload.replace('"', '\\"') + '"])</script>'
    out = parse_ramp_ai_index(page)
    assert list(out["RAMP_AI_ADOPTION"]) == [55.8, 56.8]
    assert out["RAMP_AI_PEPM_MEDIAN"].index[-1] == pd.Timestamp("2026-10-01")
    assert len(out["RAMP_AI_PEPM_TOP1"]) == 1            # the withheld month is skipped
    with pytest.raises(ValueError):
        parse_ramp_ai_index("<html>no data</html>")


def test_the_chain_links_read_their_own_thresholds():
    assert wt.link_adoption(3.0, 2) == wt.SEEN and wt.link_adoption(1.0, 2) == wt.NOT_YET
    assert wt.link_adoption(float("nan"), 2) == wt.NOT_SCORED
    assert wt.link_output(3.2, 3.1, 1.6) == wt.SEEN
    assert wt.link_output(2.2, 3.1, 1.6) == wt.PARTLY
    assert wt.link_output(1.2, 3.1, 1.6) == wt.NOT_YET


def test_costs_count_only_when_pay_holds():
    assert wt.link_costs(-1.0, 0.0, 0.25) == wt.SEEN
    assert wt.link_costs(-1.6, -0.6, 0.25) == wt.PARTLY       # costs slowed by more than pay did
    assert wt.link_costs(-0.6, -0.7, 0.25) == wt.NOT_YET      # slower pay explains all of it
    assert wt.link_costs(0.5, 0.0, 0.25) == wt.NOT_YET


def test_inflation_link_needs_the_gap_inside_the_line_and_growth_above_potential():
    assert wt.link_inflation(0.3, -0.5, 0.2, band=0.41, tol=0.25) == wt.SEEN
    assert wt.link_inflation(0.3, -0.5, -0.2, band=0.41, tol=0.25) == wt.PARTLY   # growth fell: not goldilocks
    assert wt.link_inflation(1.21, 0.30, 0.17, band=0.41, tol=0.25) == wt.NOT_YET
    assert wt.link_inflation(float("nan"), 0, 0.1, band=0.41, tol=0.25) == wt.NOT_SCORED


def test_year_change_compares_with_twelve_months_before():
    s = pd.Series(np.arange(24, dtype=float), index=pd.period_range("2024-01", periods=24, freq="M"))
    assert wt.year_change(s) == 12.0
    assert np.isnan(wt.year_change(s.iloc[:6]))


def test_period_readings_by_kind():
    idx = pd.period_range("1995-12", periods=61, freq="M")
    lv = pd.Series(100 * 1.02 ** (np.arange(61) / 12), index=idx)      # 2% a year
    assert wt.period_stat(lv, "1996-01", "2000-12", "growth") == pytest.approx(2.0)
    assert wt.recent_stat(lv, 24, "growth") == pytest.approx(2.0)
    flat = pd.Series(np.r_[np.ones(31), 3 * np.ones(30)], index=idx)
    assert wt.period_stat(flat, "1996-01", "2000-12", "start") == 1.0
    assert wt.recent_stat(flat, 24, "start") == 3.0
    assert wt.recent_stat(flat, 24, "mean") == 3.0


def test_compare_reads_direction_and_tolerance():
    assert wt.compare(240, 55, helps=1, similar=25) == wt.WORSE          # fewer jobs added
    assert wt.compare(2.1, 7.1, helps=-1, similar=0.5) == wt.WORSE       # higher tariffs
    assert wt.compare(2.8, 2.7, helps=1, similar=0.25) == wt.SIMILAR
    assert wt.compare(1.3, 0.2, helps=-1, similar=0.5) == wt.BETTER
    assert wt.compare(float("nan"), 1, helps=1, similar=0.5) == wt.NOT_SCORED


def test_new_transforms():
    idx = months("2020-01-01", 3)
    wide = pd.DataFrame({"J": [100.0, 150.0, 175.0], "D": [10.0] * 3, "S": [10.0] * 3, "G": [1000.0] * 3},
                        index=idx)
    jobs = wt.component_value({"series": ["J"], "transform": "diff"}, wide)
    assert list(jobs) == [50.0, 25.0]
    rate = wt.component_value({"series": ["D", "S", "G"], "transform": "share_of_share"}, wide)
    assert rate.iloc[-1] == pytest.approx(10.0)                         # 10 of 100 in imports


def test_every_then_and_now_row_computes():
    cfg = wt.load_config()
    wide = pd.DataFrame({sid: np.linspace(100, 200, 400) for sid in wt.required_series(cfg)},
                        index=months("1990-01-01", 400))
    for row in cfg["then_now"]["rows"]:
        if "regime_column" in row:
            continue
        v = wt.component_value(row, wide)
        assert not v.empty, row["label"]
        assert not np.isnan(wt.period_stat(v, str(cfg["then_now"]["start"]), str(cfg["then_now"]["end"]), row["kind"])), row["label"]
