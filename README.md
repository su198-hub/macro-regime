# Long-Term Regime Monitor

Scores six drivers of the US economy monthly and calls which of four regimes
the next three years look like: goldilocks, high growth with high inflation,
stagflation, or a hard landing.

The default framework is v2 (`config/indicators_v2.yml`,
`config/regimes_v2.yml`). Its drivers are growth, inflation, monetary stance,
monetary credibility, fiscal policy and reforms. Each regime sits where the
drivers stood in the run-ups to it, learned from 1968 onward and shrunk toward
a judgment prior (`scripts/realign_positions.py`). v1 (`config/indicators.yml`,
`config/regimes.yml`) describes the month itself rather than the years ahead,
and stays available from the Framework menu. The first framework listed in
`config/countries.yml` is the default.

Everything reads through a point-in-time store, so the dashboard can be rewound
to see the call as it would have been made, from data published by then. How
far back that goes depends on the stored vintages: for v2, to April 2020 (see
Known limitations). The Regime explorer goes back to 1968 on today's data.

## Running it

```bash
pip install -r requirements-dev.txt

python ingest.py demo          # synthetic data, no credentials needed
streamlit run app.py
```

That gets you a working dashboard in about a minute. The numbers are invented;
the demo exists so you can see every page work before any data work.

On Windows (PowerShell), keeping the database out of OneDrive:

```powershell
.venv\Scripts\activate
$env:MACRO_REGIME_DB = "$env:LOCALAPPDATA\macro-regime\regime.duckdb"
python ingest.py demo          # once
streamlit run app.py           # opens http://localhost:8501
```

Stop the dashboard (Ctrl+C) before running `ingest.py` again. DuckDB lets
only one process write to the file.

## Dependencies

`requirements.txt` pins every package to an exact version, which is what the
hosted app installs on each boot and what CI tests. The direct dependencies and
their allowed ranges are in `requirements.in`; `requirements-dev.txt` adds
pytest. To upgrade, follow the steps at the top of `requirements.txt`. CI's
weekly "latest" job runs the tests on the newest releases, so a breaking
upgrade shows up before the pins move.

## Hosted on Streamlit Community Cloud

GitHub stores the code and runs `.github/workflows/tests.yml` on every push:
the unit tests, then every main page rendered headlessly on demo data. The
dashboard itself is hosted by Streamlit Community Cloud, which deploys from the
repo and redeploys on each push.

1. Sign in at <https://share.streamlit.io> with GitHub.
2. Create app → this repo, branch `main`, main file `app.py`.
3. Advanced settings → Python 3.12, and under Secrets:

   ```toml
   MACRO_REGIME_DATA_URL = "https://raw.githubusercontent.com/su198-hub/macro-regime/data"
   MACRO_REGIME_SEED_DEMO = "1"
   ```

The hosted app cannot reach Macrobond, so real data is published to it from
the desk machine:

```powershell
$env:MACRO_REGIME_DB = "$env:LOCALAPPDATA\macro-regime\macrobond.duckdb"
python ingest.py sync       # refresh from Macrobond
python ingest.py publish    # push the snapshot to the repo's data branch
```

`publish` force-pushes observations (with every vintage) and series metadata
to the `data` branch, replacing the previous snapshot. It never includes the
judgement log. The app checks for a new snapshot every ten minutes and shows
when the data was published. If a snapshot cannot be downloaded, the app shows
local data with a warning and tries again every two minutes; without
`MACRO_REGIME_DATA_URL` it uses local data, seeded with demo data if
`MACRO_REGIME_SEED_DEMO` is set.

The store holds Bloomberg series, which can be published only with
`--allow-licensed`, on the vendor's approval. Adding a Bloomberg or Capital IQ
series to a published config is a new redistribution decision.

The hosted disk is temporary: anything saved in the Judgement tab there is
lost when the app restarts or new data is published.

## The live call log

Each `publish` also appends the call it publishes to `calls.csv` on the data
branch: the month scored, the regime called and its odds, each driver's
reading, the newest provisional month, and fingerprints of the config and the
code (`src/call_log.py`). The log is carried across the force-push and never
rewritten, and a copy is kept next to the store on the desk machine. If the
data branch cannot be read, `publish` stops rather than overwrite the log.

Every other grade of the model is on history it was tuned against. This log is
the record that is not: each row's outcome is known about three years and four
months after its month.

## How the call is graded

The answer key (`src/truth.py`) labels each month by what the next three years
turned out to be: real GDP growth against CBO potential, and core PCE inflation
against long-run expectations, on today's revised data. The call is graded
against it from 1976 to 2023 by `python scripts/realign_positions.py`:

- **Held out:** each episode called from positions learned without it or any
  month sharing its three years. 71% of regimes right on average.
- **Benchmarks:** the current regime continuing scores 34%, and so does the mix
  of regimes seen so far. Chance is 25%.

Read the 71% as a range. It rests on 12 episodes; resampling whole episodes
puts it between about 58% and 86% (5th to 95th percentile, checked
3 November 2026). The temperature, persistence rule, driver weights and
indicator set were all chosen while looking at the same grade, so it flatters
the model somewhat.

## Layout

```
config/countries.yml       which countries exist, their frameworks and config files
config/indicators_v2.yml   v2 indicators, weights, normalization and reasons
config/regimes_v2.yml      v2 regime positions (written by realign_positions.py), settings
config/indicators.yml      v1 indicators
config/regimes.yml         v1 archetypes
config/bench.yml           candidate indicators and the evidence for and against each
config/history_events.yml  events and false alarms pinned on the Regime explorer
config/sources.yml         each series' vendor and code
src/store.py               DuckDB vintage store, grain is (series, obs, vintage)
src/snapshot.py            export and rebuild of the published snapshot
src/call_log.py            the live call log
src/sources/               vendor adapters: Macrobond, FRED/ALFRED, Bloomberg, Capital IQ, EIA, ...
src/transform.py           transforms and normalization, pure functions
src/drivers.py             config to driver scores, confirmed and provisional
src/regimes.py             distance to each regime, softmax, persistence
src/truth.py               the answer key: what each month's next three years turned out to be
src/outlook.py             learning the run-up positions, held-out and real-time grading
src/ui.py                  HTML fragments, charts and styling
src/overrides.py           session-local changes to weights, centers and settings
app.py                     Streamlit entry point: page setup and navigation
views/dashboard.py         the dashboard: today's call and what is behind it
views/outlook.py           Regime explorer (/outlook): regime shapes, any month since 1968
views/control.py           control room: change the assumptions, per session
views/methodology.py       methodology, generated from the live config
views/bench.py             the indicator bench
views/history.py           the answer key month by month
views/energy.py            energy page, hidden from the menu (/energy), not part of the regime model
scripts/realign_positions.py  learns v2's positions and grades the call
ingest.py                  backfill / sync / coverage / demo / publish / truth
tests/                     unit tests; CI also renders every main page on demo data
```

## Adding a country

The dashboard's country menu is built from `config/countries.yml`. The United
States is live; Japan, Germany and the United Kingdom are listed as planned, so
they appear in the footer but not in the menu. A country becomes live when its
three config files exist:

1. an indicator file with the same drivers and that country's series,
2. a sources file mapping each series name to its vendor code,
3. a regimes file, or the shared one if the regimes really do carry over.

Then point its entry at those files and set `status: live`. No code changes.

Two things that do not travel. **Centers**: trend growth, the inflation
target, the noncyclical unemployment rate and average capacity utilization are
national numbers, and a center copied from the US will score another economy
wrong in a way that looks plausible. **Regime positions**: "inflation
expectations above target" means something different where the target was
undershot for twenty years. Re-derive both rather than assuming.

One store holds every country, so series names must be globally unique. Prefix
anything that is not already a unique vendor code (`JP_UNRATE`, not `UNRATE`);
`tests/test_countries.py` checks this across whatever configs exist.

Ingestion takes `--country`, which picks both config files from the registry:

```bash
python ingest.py --country us backfill
```

## Real data from Macrobond

Macrobond is the default source (`config/sources.yml`). It needs a Data+,
Professional or Enterprise licence and, for the COM client used here, Windows
with Macrobond Analysis installed and signed in on the same machine. The
package is not in the requirements because the hosted app and CI run on Linux
without it.

```powershell
.venv\Scripts\activate
pip install macrobond-data-api
$env:MACRO_REGIME_DB = "$env:LOCALAPPDATA\macro-regime\macrobond.duckdb"
python ingest.py backfill      # every series with its stored vintages
python ingest.py truth         # the answer key's inputs
python ingest.py coverage
streamlit run app.py
```

Keep real data in its own database file. `backfill` refuses to write into a
store that holds demo data. Re-run `python ingest.py sync` to refresh; for
Macrobond it refetches each series' full history.

`config/sources.yml` maps each series name used in the indicator files to its
Macrobond code, with notes where the match is not like-for-like with FRED. Set
`use: fred` on a series to take it from FRED instead (needs `FRED_API_KEY`,
free at <https://fredaccount.stlouisfed.org/apikeys>).

The COM client rules out hosted schedulers: a scheduled refresh has to run on
this machine (Windows Task Scheduler). Check your Macrobond licence before
showing its data anywhere other than your own machine. `ingest.py publish`
makes the data public if the repository is public.

## Known limitations

- **Rewinding stops in 2020 for v2.** Macrobond stores a series from its first
  recorded revision, which for many series is recent: the fed funds rate from
  April 2020, M2 from August 2020. Before a series' first stored vintage a
  rewind has no reading for it, so the date picker starts at the earliest date
  the framework can be scored. Loading ALFRED's revision history for the
  FRED-sourced series would take it back decades.
- **Graded on revised data.** For the same reason, the grade uses today's
  revised driver readings, and each indicator's center and scale are set from
  its full history.
- **Fiscal policy and reforms barely move the three-year grade.** Dropping
  either leaves it within a point (checked 3 November 2026). They are kept for
  what they measure, not for their record.
- **Very confident calls have done worse.** When the held-out odds on the top
  regime exceeded 80%, it was right in 48% of those 66 months, nearly all from
  1983 and 1995–97. Between 50% and 80%, the odds match the record.
- Quarterly and annual series are forward-filled to monthly, so those
  indicators move in steps.
