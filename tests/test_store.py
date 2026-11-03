"""The store is shared by every session of the app, each on its own thread."""

import datetime as dt
import threading

import pandas as pd

from src.store import Store


def test_threads_reading_one_store_each_get_their_own_answer(tmp_path):
    store = Store(tmp_path / "s.duckdb")
    rows = [(f"S{k}", dt.date(2000, 1, 1) + dt.timedelta(days=31 * m), dt.date(2026, 1, 1), float(k))
            for k in range(8) for m in range(300)]
    store.upsert_observations(pd.DataFrame(rows, columns=["series_id", "observation_date",
                                                          "vintage_date", "value"]))
    wrong, errors = [], []

    def read(k):
        try:
            for _ in range(25):
                got = store.as_of([f"S{k}"], dt.date(2026, 6, 1))
                if list(got.columns) != [f"S{k}"] or not (got[f"S{k}"] == k).all():
                    wrong.append(k)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=read, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and not wrong
    store.close()


def test_close_releases_the_file_for_every_thread(tmp_path):
    path = tmp_path / "s.duckdb"
    store = Store(path)
    t = threading.Thread(target=lambda: store.sources())
    t.start()
    t.join()
    store.close()
    path.rename(tmp_path / "moved.duckdb")
