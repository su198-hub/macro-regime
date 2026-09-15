"""Point-in-time storage.

The grain is (series_id, observation_date, vintage_date). Nothing in this
project reads a series without stating a vintage. That constraint is the
reason the backtest can be trusted, so it is enforced at the schema level
rather than by convention.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    series_id        VARCHAR NOT NULL,
    observation_date DATE    NOT NULL,
    vintage_date     DATE    NOT NULL,
    value            DOUBLE,
    PRIMARY KEY (series_id, observation_date, vintage_date)
);

CREATE TABLE IF NOT EXISTS series_meta (
    series_id   VARCHAR PRIMARY KEY,
    source      VARCHAR,
    title       VARCHAR,
    units       VARCHAR,
    frequency   VARCHAR,
    has_vintages BOOLEAN,
    last_sync   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS judgement (
    id          VARCHAR PRIMARY KEY,
    as_of       DATE    NOT NULL,
    driver      VARCHAR NOT NULL,
    direction   INTEGER NOT NULL,     -- -1, 0, +1
    confidence  DOUBLE  NOT NULL,     -- 0 to 1
    analyst     VARCHAR,
    source      VARCHAR,
    note        VARCHAR
);

CREATE TABLE IF NOT EXISTS regime_history (
    as_of       DATE    NOT NULL,
    config_hash VARCHAR NOT NULL,
    regime      VARCHAR NOT NULL,
    probability DOUBLE  NOT NULL,
    PRIMARY KEY (as_of, config_hash, regime)
);
"""


class Store:
    def __init__(self, path: str | Path = "data/regime.duckdb"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)
        # Added after the first stores were created; the vendor's own code
        # for the series, which differs from series_id for anything but FRED.
        self.con.execute(
            "ALTER TABLE series_meta ADD COLUMN IF NOT EXISTS source_code VARCHAR")

    # ---------- writing ----------

    def upsert_observations(self, df: pd.DataFrame) -> int:
        """df columns: series_id, observation_date, vintage_date, value."""
        if df.empty:
            return 0
        expected = {"series_id", "observation_date", "vintage_date", "value"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        self.con.register("incoming", df[list(expected)])
        self.con.execute(
            """
            INSERT INTO observations
            SELECT series_id, observation_date, vintage_date, value FROM incoming
            ON CONFLICT (series_id, observation_date, vintage_date)
            DO UPDATE SET value = excluded.value
            """
        )
        self.con.unregister("incoming")
        return len(df)

    def record_meta(self, series_id, source, title="", units="", frequency="",
                    has_vintages=True, source_code=None):
        self.con.execute(
            """
            INSERT INTO series_meta (series_id, source, title, units, frequency,
                                     has_vintages, last_sync, source_code)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (series_id) DO UPDATE SET
                source = excluded.source, title = excluded.title,
                units = excluded.units, frequency = excluded.frequency,
                has_vintages = excluded.has_vintages,
                last_sync = excluded.last_sync,
                source_code = excluded.source_code
            """,
            [series_id, source, title, units, frequency, has_vintages,
             dt.datetime.now(), source_code or series_id],
        )

    def sources(self) -> set[str]:
        """Which vendors the stored series came from."""
        return {r[0] for r in self.con.execute(
            "SELECT DISTINCT source FROM series_meta").fetchall()}

    def add_judgement(self, as_of, driver, direction, confidence, analyst,
                      note, source=""):
        """Qualitative observations get the same first-class treatment as data.

        Without this table the analyst overlay lives in someone's notebook and
        you can never ask whether judgement led or lagged the quantitative call.
        """
        import uuid
        self.con.execute(
            "INSERT INTO judgement VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [str(uuid.uuid4()), as_of, driver, int(direction), float(confidence),
             analyst, source, note],
        )

    # ---------- reading ----------

    def as_of(self, series_ids: list[str], vintage: dt.date) -> pd.DataFrame:
        """Wide frame of every series as it was known on `vintage`.

        For each observation date, takes the latest vintage at or before the
        given date. This is the only read path used by the transform layer.
        """
        self.con.register("wanted", pd.DataFrame({"series_id": series_ids}))
        df = self.con.execute(
            """
            WITH ranked AS (
                SELECT o.series_id, o.observation_date, o.value,
                       ROW_NUMBER() OVER (
                           PARTITION BY o.series_id, o.observation_date
                           ORDER BY o.vintage_date DESC
                       ) AS rn
                FROM observations o
                JOIN wanted w USING (series_id)
                WHERE o.vintage_date <= ?
            )
            SELECT series_id, observation_date, value
            FROM ranked WHERE rn = 1
            ORDER BY observation_date
            """,
            [vintage],
        ).df()
        self.con.unregister("wanted")
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot(index="observation_date", columns="series_id",
                        values="value")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()

    def first_published(self, series_id: str, vintage: dt.date, last_n: int = 24) -> pd.DataFrame:
        """When each of a series' latest observations first appeared.

        Used to estimate when the next release is due. Only vintages on or
        before `vintage` count, so a rewound view estimates from what was
        known then.
        """
        return self.con.execute(
            """
            SELECT observation_date, MIN(vintage_date) AS published
            FROM observations
            WHERE series_id = ? AND vintage_date <= ?
            GROUP BY observation_date
            ORDER BY observation_date DESC
            LIMIT ?
            """,
            [series_id, vintage, last_n],
        ).df()

    def latest(self, series_ids: list[str]) -> pd.DataFrame:
        return self.as_of(series_ids, dt.date.today())

    def vintage_dates(self, series_id: str) -> list[dt.date]:
        rows = self.con.execute(
            "SELECT DISTINCT vintage_date FROM observations "
            "WHERE series_id = ? ORDER BY vintage_date",
            [series_id],
        ).fetchall()
        return [r[0] for r in rows]

    def coverage(self) -> pd.DataFrame:
        """What you actually have. Check this before trusting any backtest."""
        return self.con.execute(
            """
            SELECT series_id,
                   COUNT(DISTINCT observation_date) AS observations,
                   COUNT(DISTINCT vintage_date)     AS vintages,
                   MIN(observation_date)            AS first_obs,
                   MAX(observation_date)            AS last_obs,
                   MIN(vintage_date)                AS first_vintage
            FROM observations GROUP BY series_id ORDER BY series_id
            """
        ).df()

    def close(self):
        self.con.close()
