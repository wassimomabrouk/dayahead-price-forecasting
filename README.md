# Day-ahead electricity price forecasting, DE/LU

Forecasting the hourly day-ahead auction price for the German/Luxembourg
bidding zone under a strict information constraint: only what a market
participant actually held at gate closure.

**Status:** in progress. Sections 0 to 3 complete.

## The question

How much day-ahead price forecast skill is achievable using only information
available at auction gate closure, and does that skill survive the 2021 to
2022 energy crisis?

## The constraint

Gate closure for delivery day D is 12:00 local time on D-1. Every feature in
this project was published at or before that instant.

Allowed: prices already cleared, published day-ahead forecasts of load, wind
and solar, calendar terms.

Not allowed: realized load, realized generation, intraday forecasts, or
anything else published after noon on D-1.

The realized series are ingested but never used as features. They exist so
the size of the leak can be measured rather than asserted.

## Data

Bundesnetzagentur SMARD, public API, no authentication. Sample runs from
2018-10-01, the date the DE/AT/LU bidding zone split and DE/LU began trading
separately. The start is a structural market break, not a data limitation.

## Usage

Run from the repository root. No install step.

```
py -m dayahead.cli ingest --dry-run
py -m dayahead.cli ingest
py -m dayahead.cli validate
py -m dayahead.cli report
py -m pytest
```

Ingestion is cached and resumable. A cold run is roughly 3,700 requests.

## Layout

```
dayahead/          package
  config.py        paths, series registry, market constants
  data/            ingestion and validation
  features/        feature construction and the gate closure filter
  models/          naive, SARIMAX, LightGBM, N-HiTS
  evaluation/      rolling origin backtest, metrics, conformal intervals
  value/           battery arbitrage valuation
  forecast/        daily run and track record
tests/             pytest suite
notebooks/         EDA and final reporting only
DESIGN.md          pre-committed specification
```

## Licence

MIT
