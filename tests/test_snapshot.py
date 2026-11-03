"""A published snapshot must rewind exactly like the store it came from."""

import datetime as dt

import pandas as pd
import pytest

from src.snapshot import export_snapshot, load_snapshot, read_manifest
from src.store import Store

D = dt.date


def test_roundtrip_preserves_vintages_and_meta_but_not_judgement(tmp_path):
    src = Store(tmp_path / "src.duckdb")
    src.upsert_observations(pd.DataFrame(
        [("x", D(2024, 1, 1), D(2024, 2, 1), 1.0),
         ("x", D(2024, 1, 1), D(2024, 4, 1), 1.2),
         ("y", D(2024, 1, 1), D(2024, 2, 1), 5.0)],
        columns=["series_id", "observation_date", "vintage_date", "value"]))
    src.record_meta("x", "macrobond", title="Series x", frequency="monthly", source_code="mbx")
    src.record_meta("y", "macrobond", title="Series y", frequency="monthly", source_code="mby")
    src.add_judgement(D(2024, 3, 1), "demand", 1, 0.5, "analyst", "private note")

    manifest = export_snapshot(src, tmp_path / "snap")
    assert manifest["rows"] == 3 and manifest["series"] == 2
    assert read_manifest(str(tmp_path / "snap"))["published_at"] == manifest["published_at"]

    copy = load_snapshot(str(tmp_path / "snap"), tmp_path / "copy.duckdb")
    for day in (D(2024, 3, 1), D(2024, 5, 1)):
        pd.testing.assert_frame_equal(src.as_of(["x", "y"], day), copy.as_of(["x", "y"], day))
    assert copy.sources() == {"macrobond"}
    assert copy.con.execute("SELECT source_code FROM series_meta WHERE series_id='x'").fetchone()[0] == "mbx"
    assert copy.con.execute("SELECT COUNT(*) FROM judgement").fetchone()[0] == 0


def test_a_snapshot_that_disagrees_with_its_manifest_is_refused(tmp_path):
    """Stale files under a new manifest must not be cached as that version."""
    store = Store(tmp_path / "src.duckdb")
    store.upsert_observations(pd.DataFrame({
        "series_id": ["A"], "observation_date": [dt.date(2026, 1, 1)],
        "vintage_date": [dt.date(2026, 1, 2)], "value": [1.0]}))
    manifest = export_snapshot(store, tmp_path / "snap")
    store.close()
    ok = load_snapshot(str(tmp_path / "snap"), tmp_path / "a.duckdb", manifest=manifest)
    ok.close()
    with pytest.raises(RuntimeError, match="manifest says"):
        load_snapshot(str(tmp_path / "snap"), tmp_path / "b.duckdb",
                      manifest={**manifest, "rows": manifest["rows"] + 1}, tries=1)


def test_a_failed_download_is_not_cached_as_the_snapshot(tmp_path, monkeypatch):
    """The hosted app must fall back under the name "local", and pick the
    snapshot up once it can be read, without waiting for the next publish."""
    import shutil

    import streamlit as st

    from views import common

    store = Store(tmp_path / "src.duckdb")
    store.upsert_observations(pd.DataFrame(
        [("x", D(2024, 1, 1), D(2024, 2, 1), 1.0)],
        columns=["series_id", "observation_date", "vintage_date", "value"]))
    snap = tmp_path / "snap"
    manifest = export_snapshot(store, snap)
    store.close()
    shutil.move(snap / "observations.parquet", tmp_path / "held.parquet")
    monkeypatch.setenv("MACRO_REGIME_DATA_URL", str(snap))
    monkeypatch.setattr(common, "SNAPSHOT_RETRY_SECONDS", 0)
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr("src.snapshot.time.sleep", lambda s: None)

    assert common.data_version() == "local"
    assert common._last_good()["failed"][0] == manifest["published_at"]

    shutil.move(tmp_path / "held.parquet", snap / "observations.parquet")
    assert common.data_version() == manifest["published_at"]
    assert "failed" not in common._last_good()
    assert common.get_store().latest(["x"])["x"].iloc[0] == 1.0
    st.cache_resource.clear()
