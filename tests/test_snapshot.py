"""A published snapshot must rewind exactly like the store it came from."""

import datetime as dt

import pandas as pd

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
