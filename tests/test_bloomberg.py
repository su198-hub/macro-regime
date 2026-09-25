"""Bloomberg adapter and the publish guard, tested without a terminal.

blpapi is not on PyPI and CI has no terminal, so the transport methods are
replaced with canned rows. What is tested is everything around them: codes,
the vintage rule, the empty case, and that licensed data cannot be published.
"""

import argparse
import datetime as dt
import types

import pandas as pd
import pytest

import ingest
from src.sources.bloomberg import BloombergSource
from src.store import Store

ROWS = [(dt.date(2026, 1, 2), 31.0), (dt.date(2026, 1, 5), 31.4), (dt.date(2026, 1, 6), 31.4)]


def source(rows=ROWS, name="Name"):
    s = BloombergSource(session=object())
    s._history = lambda ticker, field, start, period=None: (
        s.calls.append((ticker, field, start, period)) or rows)
    s._reference = lambda ticker, fields: {"NAME": name}
    s.calls = []
    return s


def test_a_ticker_defaults_to_the_last_price():
    assert BloombergSource.split("US CDS USD SR 5Y D14 Corp") == (
        "US CDS USD SR 5Y D14 Corp", "PX_LAST", None)
    assert BloombergSource.split("LF98OAS Index|PX_MID") == ("LF98OAS Index", "PX_MID", None)


def test_a_forecast_period_after_a_second_bar_is_passed_as_an_override():
    """Without it BEST_EPS means the current year, so the horizon would drift."""
    assert BloombergSource.split("SPX Index|BEST_EPS|3FY") == (
        "SPX Index", "BEST_EPS", "3FY")
    s = source()
    s.fetch_with_vintages("SPX Index|BEST_EPS|3FY")
    assert s.calls[0][1:] == ("BEST_EPS", None, "3FY")
    assert s.describe("SPX Index|BEST_EPS|3FY")["units"] == "BEST_EPS @3FY"


def test_a_traded_price_is_dated_at_its_own_observation():
    df = source().fetch_with_vintages("US CDS USD SR 5Y D14 Corp")
    assert list(df.columns) == ["series_id", "observation_date", "vintage_date", "value"]
    assert (df["observation_date"] == df["vintage_date"]).all()
    # Two days at the same spread are two observations, not a repeated vintage.
    assert df["value"].tolist() == [31.0, 31.4, 31.4]


def test_the_field_after_the_bar_is_what_gets_requested():
    s = source()
    s.fetch_with_vintages("LF98OAS Index|PX_MID", start=dt.date(2000, 1, 1))
    assert s.calls[0] == ("LF98OAS Index", "PX_MID", dt.date(2000, 1, 1), None)


def test_fetch_current_stamps_today_and_asks_only_for_recent_days():
    s = source()
    df = s.fetch_current("US CDS USD SR 5Y D14 Corp")
    assert set(df["vintage_date"]) == {dt.date.today()}
    assert s.calls[0][2] >= dt.date.today() - dt.timedelta(days=14)


def test_no_data_returns_the_right_empty_shape():
    df = source(rows=[]).fetch_with_vintages("X Corp")
    assert df.empty
    assert list(df.columns) == ["series_id", "observation_date", "vintage_date", "value"]


def test_describe_uses_the_terminal_name():
    assert source(name="United States of America").describe("US CDS USD SR 5Y D14 Corp")[
        "title"] == "United States of America"


def test_host_and_port_can_be_pointed_elsewhere(monkeypatch):
    monkeypatch.setenv("BBG_HOST", "bbg-box")
    monkeypatch.setenv("BBG_PORT", "8195")
    s = BloombergSource(session=object())
    assert (s.host, s.port) == ("bbg-box", 8195)


@pytest.mark.parametrize("vendor", sorted(ingest.RESTRICTED_VENDORS))
def test_publish_refuses_a_store_holding_licensed_data(tmp_path, vendor):
    """publish pushes to a public branch; licensed series must never reach it."""
    db = tmp_path / "store.duckdb"
    store = Store(db)
    store.upsert_observations(pd.DataFrame({
        "series_id": ["CDS"], "observation_date": [dt.date(2026, 1, 2)],
        "vintage_date": [dt.date(2026, 1, 2)], "value": [31.0]}))
    store.record_meta("CDS", vendor)
    store.close()
    with pytest.raises(SystemExit):        # coverage guard fires first, and should
        ingest.cmd_publish(argparse.Namespace(db=str(db), allow_licensed=False,
                                              config="config/indicators.yml",
                                              sources="config/sources.yml"))


@pytest.mark.parametrize("vendor", sorted(ingest.RESTRICTED_VENDORS))
def test_allow_licensed_gets_past_the_licence_guard(tmp_path, vendor, capsys, monkeypatch):
    """The override exists for a vendor agreement, and must leave a trace.

    NOTHING HERE MAY REACH GIT. An earlier version of this test called
    cmd_publish and relied on git failing in the test environment. Git did not
    fail: on 25 September 2026 it force-pushed this fixture -- one row, one
    series -- over the real snapshot, and the hosted app showed "No data yet"
    until it was rebuilt. subprocess.run is stubbed so that can never recur,
    and the model-coverage guard added alongside would now stop it anyway.
    """
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: calls.append(a) or types.SimpleNamespace(returncode=0))
    db = tmp_path / "store.duckdb"
    store = Store(db)
    store.upsert_observations(pd.DataFrame({
        "series_id": ["CDS"], "observation_date": [dt.date(2026, 1, 2)],
        "vintage_date": [dt.date(2026, 1, 2)], "value": [31.0]}))
    store.record_meta("CDS", vendor)
    store.close()
    with pytest.raises(SystemExit, match="cannot score the model"):
        ingest.cmd_publish(argparse.Namespace(db=str(db), allow_licensed=True,
                                              config="config/indicators.yml",
                                              sources="config/sources.yml"))
    assert not calls, "publish reached git from a test"


def test_publish_refuses_a_store_that_cannot_serve_the_model(tmp_path):
    """The guard that should have existed. A store holding a handful of series
    cannot score the model, so pushing it would replace the snapshot with
    something useless -- which is exactly what happened once."""
    db = tmp_path / "store.duckdb"
    store = Store(db)
    store.upsert_observations(pd.DataFrame({
        "series_id": ["UNRATE"], "observation_date": [dt.date(2026, 1, 2)],
        "vintage_date": [dt.date(2026, 1, 2)], "value": [4.0]}))
    store.record_meta("UNRATE", "macrobond")
    store.close()
    with pytest.raises(SystemExit, match="cannot score the model"):
        ingest.cmd_publish(argparse.Namespace(db=str(db), allow_licensed=False,
                                              config="config/indicators.yml",
                                              sources="config/sources.yml"))


def test_the_guard_is_the_default(tmp_path):
    """A publish that forgets the flag entirely still refuses."""
    db = tmp_path / "store.duckdb"
    store = Store(db)
    store.upsert_observations(pd.DataFrame({
        "series_id": ["CDS"], "observation_date": [dt.date(2026, 1, 2)],
        "vintage_date": [dt.date(2026, 1, 2)], "value": [31.0]}))
    store.record_meta("CDS", "bloomberg")
    store.close()
    with pytest.raises(SystemExit):        # coverage guard or licence guard, either way
        ingest.cmd_publish(argparse.Namespace(db=str(db), config="config/indicators.yml",
                                              sources="config/sources.yml"))
