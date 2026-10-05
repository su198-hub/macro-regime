"""Published data snapshots.

The Macrobond COM client only runs on a Windows desk machine, so the hosted
app cannot fetch data itself. Instead `ingest.py publish` exports the store
from that machine and pushes it to the repository's `data` branch, and the
hosted app rebuilds a store from it.

A snapshot is two Parquet files and a manifest. It carries observations with
every vintage, so the hosted app can rewind exactly like the local one, and
series metadata. It never carries the judgement log.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from pathlib import Path

import requests
import time
from urllib.parse import quote

from .store import Store

FILES = ("observations.parquet", "series_meta.parquet", "manifest.json")


def _sql_path(p: Path) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


def export_snapshot(store: Store, out_dir: str | Path) -> dict:
    """Write the snapshot files to out_dir and return the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store.con.execute(
        f"COPY (SELECT series_id, observation_date, vintage_date, value FROM observations "
        f"ORDER BY series_id, observation_date, vintage_date) "
        f"TO '{_sql_path(out / 'observations.parquet')}' (FORMAT parquet, COMPRESSION zstd)")
    store.con.execute(
        f"COPY (SELECT series_id, source, title, units, frequency, has_vintages, last_sync, "
        f"source_code FROM series_meta ORDER BY series_id) "
        f"TO '{_sql_path(out / 'series_meta.parquet')}' (FORMAT parquet)")
    rows, series, latest_vintage = store.con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT series_id), MAX(vintage_date) FROM observations"
    ).fetchone()
    manifest = {
        "published_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rows": int(rows),
        "series": int(series),
        "sources": sorted(store.sources()),
        "latest_vintage": str(latest_vintage),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _fetch(base: str, name: str, dest: Path, timeout: int = 120, version: str = "") -> Path:
    """Copy one snapshot file from a URL or a local directory into dest.

    `version` is sent as a query string so a CDN cannot answer with the file
    from the previous publish: raw.githubusercontent.com caches for about five
    minutes, and a manifest read just after a publish could otherwise be
    paired with the old observations.
    """
    target = dest / name
    if base.startswith(("http://", "https://")):
        url = f"{base.rstrip('/')}/{name}" + (f"?v={quote(version)}" if version else "")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    else:
        shutil.copyfile(Path(base) / name, target)
    return target


def read_manifest(base: str, timeout: int = 20) -> dict:
    if base.startswith(("http://", "https://")):
        r = requests.get(f"{base.rstrip('/')}/manifest.json", timeout=timeout)
        r.raise_for_status()
        return r.json()
    return json.loads((Path(base) / "manifest.json").read_text())


def load_snapshot(base: str, db_path: str | Path, manifest: dict | None = None,
                  tries: int = 3) -> Store:
    """Build a store at db_path from the snapshot at base (URL or directory).

    With a manifest, the files are fetched for that publish and the row count
    checked against it, retrying a few times: the app caches a store under the
    manifest's version until the next publish, so loading stale files under a
    new version would stick until then.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    version = str((manifest or {}).get("published_at", ""))
    for attempt in range(tries):
        _build(base, db_path, version)
        store = Store(db_path)
        rows = store.con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        if manifest is None or rows == manifest.get("rows"):
            return store
        store.close()
        if attempt < tries - 1:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"snapshot has {rows:,} rows, its manifest says {manifest.get('rows'):,}; "
                       f"the files may still be from the previous publish")


def _build(base: str, db_path: Path, version: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in FILES[:2]:
            _fetch(base, name, tmp, version=version)
        partial = db_path.with_suffix(".partial")
        for p in (partial, Path(str(partial) + ".wal")):
            if p.exists():
                p.unlink()
        store = Store(partial)
        store.con.execute(
            f"INSERT INTO observations SELECT series_id, observation_date, vintage_date, value "
            f"FROM read_parquet('{_sql_path(tmp / 'observations.parquet')}')")
        store.con.execute(
            f"INSERT INTO series_meta (series_id, source, title, units, frequency, has_vintages, "
            f"last_sync, source_code) SELECT series_id, source, title, units, frequency, "
            f"has_vintages, last_sync, source_code "
            f"FROM read_parquet('{_sql_path(tmp / 'series_meta.parquet')}')")
        store.close()
    # Swap in only once complete, so a failed download never leaves half a store.
    os.replace(partial, db_path)
