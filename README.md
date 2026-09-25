# Data snapshot

Published by `python ingest.py publish` from the machine that holds the
Macrobond connection. The hosted app reads it via MACRO_REGIME_DATA_URL.
This branch is replaced on every publish, so it never accumulates history.

- observations.parquet: series_id, observation_date, vintage_date, value
- series_meta.parquet: source, vendor code, title, units, frequency
- manifest.json: when it was published and what it holds
