"""Capital IQ adapter, tested without credentials.

A fake transport stands in for the API, so the parsing, the vintage rule and
the refusals are all verified before a real key exists. What cannot be checked
here is the response shape — that is what `ingest.py ciq-probe` is for.
"""

import datetime as dt

import pytest

from src.sources import ciq
from src.sources.ciq import CiqAuthError, CiqSource


class FakeResponse:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Records what was sent and replies with whatever it was given."""

    def __init__(self, payload=None, status=200):
        self.payload, self.status, self.sent = payload or {}, status, []
        self.auth = None
        self.closed = False

    def post(self, url, json=None, timeout=None, headers=None):
        self.sent.append({"url": url, "body": json})
        return FakeResponse(self.payload, self.status, text="denied")

    def close(self):
        self.closed = True


GDS = {"GDSSDKResponse": [{
    "Headers": ["IQ_CDS_SPREAD_5YR"],
    "Rows": [{"Row": ["01/02/2026", "42.5"]},
             {"Row": ["01/03/2026", "43.1"]},
             {"Row": ["01/04/2026", "43.1"]}],
}]}


def source(payload=None, status=200):
    fake = FakeSession(payload, status)
    return CiqSource(username="u", password="p", pause=0, session=fake), fake


@pytest.fixture
def no_stored_login(monkeypatch):
    """Hide any real login on this machine, so tests never touch it."""
    monkeypatch.delenv("CIQ_USERNAME", raising=False)
    monkeypatch.delenv("CIQ_PASSWORD", raising=False)
    monkeypatch.setattr(ciq, "user_env", lambda name: None)


def test_credentials_are_required_and_never_defaulted(no_stored_login):
    with pytest.raises(RuntimeError, match="ciq-login"):
        CiqSource(username=None, password=None)


def test_credentials_come_from_the_environment(no_stored_login, monkeypatch):
    monkeypatch.setenv("CIQ_USERNAME", "env-user")
    monkeypatch.setenv("CIQ_PASSWORD", "env-pass")
    s = CiqSource(session=FakeSession())
    assert s.username == "env-user"
    assert s.session.auth == ("env-user", "env-pass")


def test_a_stored_login_is_found_without_restarting_the_terminal(no_stored_login,
                                                                 monkeypatch):
    """ciq-login writes the Windows user environment, which shells started
    earlier never inherit. The adapter reads it live instead."""
    stored = {"CIQ_USERNAME": "stored-user", "CIQ_PASSWORD": "stored-pass"}
    monkeypatch.setattr(ciq, "user_env", stored.get)
    s = CiqSource(session=FakeSession())
    assert s.session.auth == ("stored-user", "stored-pass")


def test_the_process_environment_wins_over_a_stored_login(no_stored_login, monkeypatch):
    monkeypatch.setattr(ciq, "user_env", {"CIQ_USERNAME": "stored"}.get)
    monkeypatch.setenv("CIQ_USERNAME", "session")
    monkeypatch.setenv("CIQ_PASSWORD", "p")
    assert CiqSource(session=FakeSession()).username == "session"


def test_login_refuses_to_run_without_a_terminal(monkeypatch, capsys):
    """A password prompt answered by a script is a password in a script."""
    import argparse
    import sys

    import ingest
    if sys.platform != "win32":
        pytest.skip("the login stores to the Windows user environment")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    ingest.cmd_ciq_login(argparse.Namespace(clear=False))
    assert "your own terminal" in capsys.readouterr().out


def test_a_refusal_says_it_may_be_the_entitlement_not_the_password():
    """The likeliest cause of a 403 here is a Pro seat without the API."""
    s, _ = source(GDS, status=403)
    with pytest.raises(CiqAuthError, match="not entitled"):
        s.fetch_current("IQ1|IQ_CDS_SPREAD_5YR")


def test_a_code_must_name_both_the_entity_and_the_field():
    s, _ = source(GDS)
    with pytest.raises(ValueError, match="identifier\\|mnemonic"):
        s.fetch_current("IQ_CDS_SPREAD_5YR")


def test_a_market_price_is_dated_at_its_own_observation():
    """It is never revised, so it was known on the day it printed. Dating it at
    fetch time instead would hide it from every vintage before today."""
    s, _ = source(GDS)
    df = s.fetch_with_vintages("IQ1|IQ_CDS_SPREAD_5YR")
    assert list(df.columns) == ["series_id", "observation_date", "vintage_date", "value"]
    assert (df["observation_date"] == df["vintage_date"]).all()
    # Every trading day is kept. Two days at the same spread are two
    # observations, not one value repeated across vintages, so the
    # unchanged-repeat rule that thins a revised series must not touch them.
    assert df["value"].tolist() == [42.5, 43.1, 43.1]
    assert df["observation_date"].tolist() == [
        dt.date(2026, 1, 2), dt.date(2026, 1, 3), dt.date(2026, 1, 4)]


def test_fetch_current_stamps_today_instead():
    s, _ = source(GDS)
    df = s.fetch_current("IQ1|IQ_CDS_SPREAD_5YR")
    assert set(df["vintage_date"]) == {dt.date.today()}
    assert len(df) == 3  # no vintage tidying on a current read


def test_restated_fundamentals_are_refused_for_a_vintage_read():
    """CIQ carries today's restated numbers back through history. Storing that
    as a vintage record would put revised values into months that never saw
    them, and every backtest downstream would be quietly wrong."""
    s, _ = source(GDS)
    with pytest.raises(ValueError, match="restated fundamental"):
        s.fetch_with_vintages("IQ1|IQ_CAPEX")
    # The same field is fine as a current read.
    assert not s.fetch_current("IQ1|IQ_CAPEX").empty


def test_a_vendor_error_message_is_surfaced_not_swallowed():
    s, _ = source({"GDSSDKResponse": [{"ErrMsg": "Invalid mnemonic", "Rows": []}]})
    with pytest.raises(RuntimeError, match="Invalid mnemonic"):
        s.fetch_current("IQ1|IQ_NONSENSE")


def test_an_empty_response_returns_the_right_empty_shape():
    s, _ = source({"GDSSDKResponse": [{"Headers": [], "Rows": []}]})
    df = s.fetch_current("IQ1|IQ_CDS_SPREAD_5YR")
    assert df.empty
    assert list(df.columns) == ["series_id", "observation_date", "vintage_date", "value"]


def test_dates_the_parser_cannot_read_are_dropped_not_guessed():
    s, _ = source({"GDSSDKResponse": [{"Rows": [
        {"Row": ["not a date", "1.0"]}, {"Row": ["01/05/2026", "2.0"]}]}]})
    df = s.fetch_current("IQ1|IQ_X")
    assert df["value"].tolist() == [2.0]


def test_the_request_carries_the_identifier_mnemonic_and_a_date_window():
    s, fake = source(GDS)
    s.fetch_with_vintages("IQ1|IQ_CDS_SPREAD_5YR", start=dt.date(2020, 1, 1))
    req = fake.sent[0]["body"]["inputRequests"][0]
    assert req["identifier"] == "IQ1"
    assert req["mnemonic"] == "IQ_CDS_SPREAD_5YR"
    assert req["properties"]["startDate"] == "01/01/2020"
    assert "endDate" in req["properties"]


def test_the_endpoint_can_be_pointed_elsewhere(no_stored_login, monkeypatch):
    """S&P has moved this before, and entitlements differ by contract."""
    monkeypatch.setenv("CIQ_ENDPOINT", "https://example.invalid/api")
    s = CiqSource(username="u", password="p", session=FakeSession())
    assert s.endpoint == "https://example.invalid/api"


def test_close_releases_the_session():
    s, fake = source(GDS)
    s.close()
    assert fake.closed
