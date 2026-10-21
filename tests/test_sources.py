"""Vintage tidying must not change what Store.as_of would return."""

import datetime as dt

import pandas as pd

from src.sources.base import tidy_vintages
from src.store import Store

D = dt.date


def frame(rows):
    return pd.DataFrame(rows, columns=["series_id", "observation_date", "vintage_date", "value"])


def test_repeats_collapse_and_revisions_survive():
    raw = frame([
        ("x", D(2024, 1, 1), D(2024, 2, 1), 1.0),
        ("x", D(2024, 1, 1), D(2024, 3, 1), 1.0),   # unchanged repeat: dropped
        ("x", D(2024, 1, 1), D(2024, 4, 1), 1.2),   # revision: kept
        ("x", D(2024, 1, 1), D(2024, 5, 1), 1.2),   # unchanged repeat: dropped
        ("x", D(2024, 2, 1), D(2024, 3, 1), 2.0),
    ])
    out = tidy_vintages(raw)
    assert list(zip(out["observation_date"], out["vintage_date"], out["value"])) == [
        (D(2024, 1, 1), D(2024, 2, 1), 1.0),
        (D(2024, 1, 1), D(2024, 4, 1), 1.2),
        (D(2024, 2, 1), D(2024, 3, 1), 2.0),
    ]


def test_projections_dated_after_their_vintage_are_dropped():
    raw = frame([
        ("n", D(2026, 1, 1), D(2026, 2, 1), 4.2),
        ("n", D(2030, 1, 1), D(2026, 2, 1), 4.1),   # CBO projection
    ])
    assert tidy_vintages(raw)["observation_date"].tolist() == [D(2026, 1, 1)]


def test_as_of_is_unchanged_by_tidying(tmp_path):
    raw = frame([
        ("x", D(2024, 1, 1), D(2024, 2, 1), 1.0),
        ("x", D(2024, 1, 1), D(2024, 3, 1), 1.0),
        ("x", D(2024, 1, 1), D(2024, 4, 1), 1.2),
        ("x", D(2024, 2, 1), D(2024, 3, 1), 2.0),
        ("x", D(2024, 2, 1), D(2024, 4, 1), 2.0),
    ])
    full, tidy = Store(tmp_path / "full.duckdb"), Store(tmp_path / "tidy.duckdb")
    full.upsert_observations(raw)
    tidy.upsert_observations(tidy_vintages(raw))
    for day in (D(2024, 2, 15), D(2024, 3, 15), D(2024, 4, 15)):
        pd.testing.assert_frame_equal(full.as_of(["x"], day), tidy.as_of(["x"], day))


def test_quantgov_reads_one_member_of_a_remote_zip_and_sums_by_year():
    """Only the byte ranges asked for are served, as S3 would."""
    import io
    import zipfile

    from src.sources.quantgov import QuantgovSource

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big/other.csv", "x" * 50_000)
        z.writestr("big/usregdata6.csv", "year,part,restrictions\n2019,a,10\n2019,b,5\n2020,a,12\n")
    blob = buf.getvalue()

    class Resp:
        def __init__(self, content=b"", headers=None):
            self.content, self.headers = content, headers or {}

        def raise_for_status(self):
            pass

    class Session:
        ranges = []

        def head(self, url, **kw):
            return Resp(headers={"Content-Length": str(len(blob))})

        def get(self, url, headers=None, **kw):
            lo, hi = map(int, headers["Range"].removeprefix("bytes=").split("-"))
            self.ranges.append((lo, hi))
            return Resp(blob[lo:hi + 1])

        def close(self):
            pass

    src = QuantgovSource()
    src.session = Session()
    out = src.fetch_current("https://example.test/r.zip|big/usregdata6.csv")
    assert out["observation_date"].tolist() == [D(2019, 12, 31), D(2020, 12, 31)]
    assert out["value"].tolist() == [15.0, 12.0]
