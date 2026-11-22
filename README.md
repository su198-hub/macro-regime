# Data snapshot

Published by `python ingest.py publish` from the machine that holds the
Macrobond connection. The hosted app reads it via MACRO_REGIME_DATA_URL.
This branch is replaced on every publish, so it never accumulates history,
except for calls.csv, which each publish carries over and adds a row to.

- observations.parquet: series_id, observation_date, vintage_date, value
- series_meta.parquet: source, vendor code, title, units, frequency
- manifest.json: when it was published and what it holds
- calls.csv: the call published each time, its odds, the driver readings and
  fingerprints of the config and code (src/call_log.py). Never rewritten.
