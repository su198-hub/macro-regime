"""The live call log: a publish may add a row, never change or lose one."""

import pandas as pd
import pytest

from src import call_log as cl


def results(month="2026-07-31", called="goldilocks"):
    idx = pd.DatetimeIndex(["2026-06-30", month])
    probs = pd.DataFrame({"goldilocks": [0.6, 0.7], "stagflation": [0.4, 0.3]}, index=idx)
    calls = pd.DataFrame({"called": [called, called], "leading": ["goldilocks"] * 2}, index=idx)
    drivers = pd.DataFrame({"growth": [0.1, 0.23456], "growth__coverage": [1.0, 1.0]}, index=idx)
    return {"probabilities": probs, "calls": calls, "drivers": drivers, "provisional": []}


def test_row_holds_the_call_its_odds_and_the_drivers():
    r = cl.row(results(), "2026-08-01T00:00:00+00:00", "v2", "abc", "def")
    assert r["month"] == "2026-07" and r["called"] == "goldilocks"
    assert r["p_goldilocks"] == 0.7 and r["p_stagflation"] == 0.3
    assert r["d_growth"] == 0.2346 and "d_growth__coverage" not in r
    assert r["provisional_month"] == ""


def test_row_refuses_when_nothing_was_scored():
    res = results()
    res["probabilities"] = res["probabilities"] * float("nan")
    with pytest.raises(ValueError):
        cl.row(res, "t", "v2", "a", "b")


def test_merge_keeps_every_row_once_and_never_rewrites_one():
    a = pd.DataFrame([{"published_at": "2026-08-01", "called": "goldilocks"}])
    b = pd.DataFrame([{"published_at": "2026-08-01", "called": "stagflation"},
                      {"published_at": "2026-07-01", "called": "hard_landing"}])
    out = cl.merge(a, b, new={"published_at": "2026-09-01", "called": "goldilocks"})
    assert list(out["published_at"]) == ["2026-07-01", "2026-08-01", "2026-09-01"]
    assert out.set_index("published_at").at["2026-08-01", "called"] == "goldilocks"


def test_merge_round_trips_through_csv():
    log = cl.merge(new=cl.row(results(), "2026-08-01T00:00:00+00:00", "v2", "abc", "def"))
    back = cl.read(log.to_csv(index=False))
    pd.testing.assert_frame_equal(back, log)
    assert cl.read("").empty and cl.read(None).empty


def test_config_hash_ignores_line_endings(tmp_path):
    unix, win = tmp_path / "a.yml", tmp_path / "b.yml"
    unix.write_bytes(b"x: 1\ny: 2\n")
    win.write_bytes(b"x: 1\r\ny: 2\r\n")
    assert cl.config_hash(unix) == cl.config_hash(win)
