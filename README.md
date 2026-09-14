# Macro regime dashboard

Five drivers — demand, inflation expectations, supply constraint, policy stance,
investment spending — scored monthly from US data and mapped to four regimes:
goldilocks, high growth with high inflation, stagflation, hard landing.

Everything reads through a point-in-time store, so the dashboard can be rewound
to any date and will show the call you would have made then.

## Running it

```bash
pip install -r requirements.txt

python ingest.py demo          # synthetic data, no credentials needed
streamlit run app.py
```

That gets you a working dashboard in about a minute. The numbers are invented;
it exists so you can see the shape before committing to data work.

On Windows (PowerShell), keeping the database out of OneDrive:

```powershell
.venv\Scripts\activate
$env:MACRO_REGIME_DB = "$env:LOCALAPPDATA\macro-regime\regime.duckdb"
python ingest.py demo          # once
streamlit run app.py           # opens http://localhost:8501
```

Stop the dashboard (Ctrl+C) before running `ingest.py` again. DuckDB lets
only one process write to the file.

## Hosted on Streamlit Community Cloud

GitHub stores the code and runs `.github/workflows/tests.yml` on every push.
The dashboard itself is hosted by Streamlit Community Cloud, which deploys
from the repo and redeploys on each push.

1. Sign in at <https://share.streamlit.io> with GitHub.
2. Create app → this repo, branch `main`, main file `app.py`.
3. Advanced settings → Python 3.12, and under Secrets:
   `MACRO_REGIME_SEED_DEMO = "1"`

The hosted disk is temporary. The demo data is rebuilt whenever the app
restarts, and anything saved in the Judgement tab there is lost with it.

For real data, get a free key at
<https://fredaccount.stlouisfed.org/apikeys>:

```bash
export FRED_API_KEY=your_key
python ingest.py backfill      # full ALFRED vintage history, slow, run once
python ingest.py coverage      # check what you actually got
python ingest.py sync          # current values, put this on a schedule
```

Backfill takes roughly ten to twenty minutes for the current indicator set.

## Before you trust anything

Run `python ingest.py coverage` and read the vintages column. Any series
showing one vintage has no revision history, which means a backtest including
it is as-revised rather than point-in-time. Breakevens and the funds rate are
legitimately single-vintage because they are never revised. GDP-adjacent series
showing one vintage are a problem.

## What to do first

Not code. Two things:

1. **Curate `config/indicators.yml`.** The set shipped here is a plausible
   starting point, not a recommendation. Every indicator carries a `why` field.
   Delete the ones whose reason you don't believe, add the ones you actually
   watch, and argue about the weights. This file is the project.

2. **Hand-label a regime history back to 1970.** Do it before you look at any
   model output, from what was knowable at the time, and store it alongside
   this repo. It is the only thing that will tell you whether the archetypes in
   `config/regimes.yml` are any good, and it is the asset that survives when
   you rewrite the code.

## Layout

```
config/indicators.yml   indicator set, weights, normalisation, reasons
config/regimes.yml      regime archetypes in driver space, persistence settings
src/store.py            DuckDB vintage store, grain is (series, obs, vintage)
src/sources/fred.py     ALFRED adapter, pulls full revision history
src/sources/macrobond.py  Macrobond adapter, needs Data+ (see below)
src/transform.py        transforms and normalisation, pure functions
src/drivers.py          config to five driver scores
src/regimes.py          archetype distance, softmax, persistence
app.py                  Streamlit dashboard
ingest.py               backfill / sync / coverage / demo
tests/                  the transforms and persistence logic
```

## Swapping FRED for Macrobond

`src/sources/macrobond.py` is written and unactivated. Requires a Data+,
Professional or Enterprise licence — a standard Analysis seat will not
authenticate. Check with:

```bash
pip install macrobond-data-api
python -c "from macrobond_data_api.com import ComClient; \
           print(ComClient().__enter__().get_one_series('usgdp').values[:3])"
```

If that returns numbers, change the import in `ingest.py` and map the
`source.fred` fields in the indicator config to Macrobond series codes.

The COM client is Windows-only and needs the desktop app on the same machine,
which rules out hosted schedulers. The Web API client runs anywhere but is a
separate entitlement. If you plan to run this pipeline anywhere other than a
desk machine, price the Web API before you build around COM.

## Known limitations

- Quarterly and annual series are forward-filled to monthly, so those drivers
  move in steps. The deficit series is annual and contributes very little
  timely information.
- SLOOS lending standards and CBO's NAIRU are quarterly and annual
  respectively, and both lag.
- The regime archetypes have not been validated against a labelled history.
  Until you do that, treat the probabilities as a structured way to read the
  drivers, not as a forecast.
- Calibrating thresholds on 2021 to 2023 will overfit. That episode is vivid
  and atypical.
