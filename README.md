# Day-ahead electricity price forecasting, DE/LU

Forecasting the hourly day-ahead auction price for the German and Luxembourg
bidding zone, under the constraint that only information published before the
noon auction may be used.

Eight years of data, eight models, a twelve-month held-out year opened once,
and a daily forecast that has been running since September 2026.

---

## Two findings

**The most accurate model cannot run.**

Five of the six exogenous inputs it depends on (`fc_solar`, `fc_wind_on`,
`fc_wind_off`, `fc_gen_total` and `fc_residual`) are published by SMARD
*after* the price they would have been used to predict is already public.
Only the day-ahead load forecast, `fc_load`, arrives in time. Measured directly on 11 September 2026: the load forecast for the
next day appeared before 10:00, the auction cleared at 12:00, the price was
published around 12:45, and the wind and solar forecasts arrived between 17:11
and 18:11. Roughly five hours too late to be an input to the decision they
describe.

A backtest cannot see this. In a historical table every value is simply
present and nothing records when it arrived. It surfaced on the first attempt
to forecast a day that had not happened yet.

`fc_residual` is the costly loss. It is load minus wind minus solar, so it
inherits the delay of its components, and it was the single most physically
meaningful feature in the project: it approximates where the market sits on
the merit-order curve.

Rebuilding the model on what genuinely arrives in time, price history,
calendar terms and `fc_load`, costs **8.8%** on MAE, 18.33 to 19.95 EUR/MWh,
and produces a system that runs every morning. The repository contains both,
because the gap between them is the result.

**A 17.7% better forecast earned no additional money.**

Dispatching a 1 MW / 2 MWh battery on the champion's forecasts over the
held-out year captured 89.0% of perfect-foresight arbitrage value. Dispatching
on a naive baseline that repeats yesterday's prices captured 89.0% as well,
58,391 EUR against 58,375 across a year.

Arbitrage depends on correctly *ranking* the hours of a day, cheapest against
dearest. It never uses the price level. A model can be substantially more
accurate in absolute terms while selecting the same two troughs and the same
two peaks as a naive rule.

Which error metric matters is set by the decision, not by convention.

---

## Results, held-out year

Twelve months, 2025-09 to 2026-08, never used in development. Opened once,
with the champion and its calibration fixed beforehand.

| model | MAE | RMSE | skill vs weekly naive | interval coverage |
|---|---|---|---|---|
| **N-HiTS, champion** | **23.45** | 129.97 | **0.320** | 0.821 |
| daily naive baseline | 28.50 | 45.18 | 0.201 | 0.789 |
| weekly naive baseline | 35.68 | 54.91 | 0.000 | 0.780 |

Nominal interval coverage is 80%. The champion's 82.1% is the closest any
model reached at any point in the project.

The RMSE of 129.97 is inflated by eighteen hours in February where the model
diverged numerically. Excluding them gives MAE 18.33 and RMSE 42.99. Both
figures are reported; the headline is what the code produced on the day the
held-out set was opened, and it was not rerun afterwards.

---

## The constraint, and what it costs

The rule: to forecast delivery day D, use only what was published before the
auction for D cleared at noon on D-1. No realised outturn, no intraday
revisions, no prices from D itself.

Enforcement is structural rather than documentary. Every feature carries an
availability timestamp assigned from market rules, and a test suite rejects
any feature whose timestamp exceeds issuance, including one that injects a
deliberately leaky column and requires it to fail.

Three measurements bracket the constraint:

| information set | MAE | |
|---|---|---|
| realised outturn substituted for the forecasts | 18.75 | 20.0% better |
| published day-ahead forecasts | 23.45 | reference |
| only what arrives in time (`fc_load` alone) | 19.95 | 8.8% worse than matched full set |

The first figure is what a leaky implementation would report. The third is
what a system that can actually run achieves. Most published work on this
problem sits somewhere between without saying where.

---

## Eight predictions, six wrong

`DESIGN.md` was written and committed before any model was fitted. It fixes
the metrics, the fold structure, the regime boundaries, the selection rule and
eight predictions about what would happen.

| | prediction | outcome |
|---|---|---|
| 12.1 | held-out MAE exceeds backtest MAE | confirmed |
| 12.2 | plain ARIMA will not beat the weekly naive | **refuted** |
| 12.3 | SARIMAX beats both baselines | confirmed |
| 12.4 | LightGBM beats SARIMAX | **refuted** |
| 12.5 | N-HiTS will not beat LightGBM | **refuted** |
| 12.6 | skill is lowest during the energy crisis | confirmed |
| 12.7 | gradient boosting quantiles under-cover in the tails | **refuted** |
| 12.8 | SARIMAX calibrates better than LightGBM | **refuted** |

Each refutation is recorded with the mechanism that caused it. The most
instructive is 12.4: gradient boosting on price levels collapsed during the
2022 crisis because a tree cannot predict a value above the maximum it was
trained on. True prices reached 871 EUR/MWh; predictions capped at 418.6.
Predicting the difference from a naive baseline instead removes the ceiling
and recovers most of the loss.

---

## Bugs, and how each was found

None were caught by an error message. Every one surfaced as a number that did
not fit.

| defect | symptom | cost |
|---|---|---|
| Fixed 24-hour offsets in the lag features | 384 rows missing | broke every day after a DST transition |
| State space models extended with missing values | bias of +31.9 EUR/MWh | day-ahead forecasts became month-long extrapolations |
| N-HiTS skipping short DST days without advancing its history | 9.6% of the held-out year unpredicted | invisible: missing rows reduce the count, not the score |
| Cached weekly blocks never refreshed | ingest reported success, fetched nothing | the daily job could never see new data |
| A variant suffix missing from a lookup | half a year silently dropped | metrics computed on 4,320 rows instead of 8,664 |

Three of the five were invisible to the metrics, because absent predictions
shrink the sample rather than degrade a score. Each now has a regression test.

---

## Reproducing it

Python 3.11 or later. No installation step; run from the repository root.

```bash
pip install -e ".[models,dev]" neuralforecast

python -m dayahead.cli ingest      # ~3,700 requests, cached and resumable
python -m dayahead.cli validate    # assemble and check the panel
python -m dayahead.cli features    # build and audit the feature matrix
python -m dayahead.cli backtest --models all   # 59 folds, several hours
python -m dayahead.cli conformal   # calibrate the intervals
python -m dayahead.cli select      # apply the selection rule
python -m dayahead.cli value       # battery arbitrage valuation
python -m pytest                   # 102 tests
```

The daily forecast, which is what the scheduled job runs:

```bash
python -m dayahead.cli forecast
python -m dayahead.cli page
```

Data is public: Bundesnetzagentur SMARD, no authentication required.

---

## Layout

```
dayahead/
  config.py        paths, series registry, market constants
  data/            SMARD client, validation, panel assembly
  features/        feature construction and the gate closure filter
  models/          naive, ARIMA/SARIMA/SARIMAX, LightGBM, N-HiTS
  evaluation/      backtest harness, metrics, conformal calibration
  value/           battery arbitrage valuation
  forecast/        daily run, track record, published page
tests/             102 tests
reports/           every result, committed
DESIGN.md          the full specification and every result in sequence
```

`DESIGN.md` is the substance. It was frozen before modelling, and results were
appended in sequence without editing what came before, including the parts
that turned out to be wrong.

---

## Limitations

The dispatch rule in the valuation is deliberately simple: one cycle a day, no
degradation, no grid fees, no imbalance exposure, no intraday market. It
measures the relative value of forecast information, not a profit and loss.

Conformal calibration corrects the average and cannot correct the shape of an
interval. During the 2022 crisis no model reached nominal coverage, the best
being 75.3%. These intervals are trustworthy in a market resembling the recent
past and are not trustworthy through a structural break.

The champion is beaten by both naive baselines on negative-price hours, MAE
60.27 against 40.11, and that subset has grown from 1.2% of hours in 2018 to
7.3% in 2026. The failure is inherited from the accuracy of the driver
forecasts rather than produced by the model, but it is the fastest-growing
part of the problem.

---

Built by Wassim Mabrouk. Data from Bundesnetzagentur SMARD under their terms
of use.
