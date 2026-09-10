# Design specification

Written before any exploratory analysis and before any model is fitted.
Committed to git before the locked test set is opened.

The point of this document is that the evaluation protocol is chosen without
knowledge of which choices would flatter the results. Anything decided after
looking at the data is a modelling choice and is labelled as such below.

---

## 1. Question

How much day-ahead price forecast skill is achievable for the German and
Luxembourg bidding zone using only information available at auction gate
closure, and does that skill survive the 2021 to 2022 energy crisis?

Two things make this a research question rather than a modelling exercise.
The information constraint is binding and is commonly violated. The regime
break makes the question falsifiable: a model that only works in calm periods
is a failed model, and this document commits to reporting that outcome if it
occurs.

---

## 2. Information constraint

Gate closure for the day-ahead auction is 12:00 local time on D-1 for
delivery day D. A feature is admissible only if it was published at or before
that instant.

**Admissible**

- Prices from auctions that have already cleared
- Published day-ahead forecasts of load, wind onshore, wind offshore, solar,
  total generation and residual load
- Calendar terms, which are known arbitrarily far ahead

**Not admissible**

- Realized load, realized generation, realized residual load
- Intraday generation forecasts (SMARD category 32), which revise the
  day-ahead values after gate closure
- Any price for a delivery hour on day D or later

**The non-obvious constraint.** Prices for delivery day D clear at the auction
held at noon on D-1. At the moment of issuance the newest price known is for
the last hour of D-1, so the freshest price information is already a full day
old in auction terms. A one-hour price lag looks conservative and is leakage.
`config.MIN_PRICE_LAG_HOURS = 24` encodes this and `tests/test_config.py`
asserts it.

**Enforcement.** Every feature carries an explicit availability timestamp,
assigned from the market rules rather than inferred from the data, since
SMARD publishes delivery timestamps only. `tests/test_gate_closure.py`
asserts that no assigned availability timestamp exceeds issuance, that no
forbidden series reaches the feature matrix, that the minimum price lag is
respected, and that a deliberately leaky column injected into the matrix
causes a failure. The last of these is what stops the check from being
decorative. What the tests enforce is the assignment; section 2a states
what the assignment itself rests on.

---

## 2a. Limitation: forecast publication timing

Established after section 3b, by checking Commission Regulation (EU)
543/2013 rather than by inspecting data. Recorded as a disclosure. It does
not alter any pre-committed evaluation choice in this document.

The day-ahead total load forecast must be published no later than two hours
before day-ahead gate closure, so by 10:00 on D-1. It is unambiguously
inside the gate.

The day-ahead wind and solar generation forecast has a publication deadline
of 18:00 on D-1, which falls after the noon auction. SMARD serves a delivery
timestamp, not a publication timestamp, so it cannot be verified from this
data that the exact values used here existed at 12:00 on D-1.

What can be verified: these are day-ahead vintage series, distinct from the
intraday forecasts (SMARD category 32) and from realized outturn, and no
realized series enters a model.

The claim this project makes is therefore the narrower and accurate one: no
realized outturn and no intraday revision enters the feature set, and the
exogenous inputs are day-ahead forecasts of the kind a participant held at
gate closure. The claim it does not make is that every value carries a
publication timestamp before noon on D-1.

In practice German TSOs publish ahead of the regulatory deadline and
participants hold wind and solar forecasts at noon of comparable quality.
This construction is also standard in the published price forecasting
literature, which uses ENTSO-E day-ahead prices together with day-ahead load
and RES generation forecasts for the same task. Neither fact constitutes
verification, and both are stated here rather than relied on silently.

**Consequence for interpretation.** Any residual optimism from this source
inflates all models equally, including the baselines, so the skill scores and
the relative model ranking are unaffected. Absolute error levels may be
slightly optimistic relative to a participant using only vendor forecasts
available at noon.

---

## 3. Data facts established in sections 0 to 2

These are measured, not assumed.

- Sample 2018-10-01 to 2026-09-07, hourly, 69,576 rows after trimming to the
  common publication frontier.
- The start date is the DE/AT/LU bidding zone split. Prices before it belong
  to a different market. This is a structural break, not a data limitation.
- `fc_residual` equals `fc_load - fc_wind_on - fc_wind_off - fc_solar` to
  within 0.02 MW on quantities near 50,000 MW, with correlation 1.000000.
  The difference is publication rounding. The five columns are linearly
  dependent, so any model requiring a full-rank design matrix uses one side
  of the identity or the other, never both.
- One 24-hour publication outage on the load forecast, local 2020-01-31.
  Both sides of the identity are missing, so it cannot be reconstructed from
  any admissible series. It is excluded, not imputed: fabricating a published
  forecast that never existed would contradict the premise of the project.
  24 of 69,576 rows, 0.03%. Rows stay in the index to keep the hourly grid
  unbroken and are flagged via `is_usable`.
- 8 short and 8 long local days, consistent with 8 DST cycles. Conversion
  uses Europe/Berlin, never a fixed offset.
- Negative prices occur in 3.61% of hours overall, rising from 1.2% in 2018
  to 7.3% in 2026 as solar capacity grows.

---

## 4. Target and horizon

Target: hourly day-ahead auction price for DE/LU, EUR/MWh.

Horizon: all 24 delivery hours of day D, forecast from the information set at
12:00 local on D-1. Effective lead times run from 12 to 36 hours.

The unit of prediction is a day, not an hour. A forecast is 24 numbers issued
together, and evaluation respects that.

---

## 5. Baselines

**B1, primary denominator.** Price at the same hour one week earlier, lag 168
hours. It carries the daily shape, the weekday effect and the recent level at
once, and is genuinely hard to beat in power markets.

**B2, secondary.** Price at the same hour on the previous day, lag 24 hours.
Often stronger than B1 in stable periods and weaker across weekday and
weekend boundaries.

Skill score is reported against B1. Skill against B2 is reported alongside so
the choice of denominator cannot flatter the result.

---

## 6. Metrics

Point forecasts:

- **MAE**, EUR/MWh. Headline. Interpretable and robust to spikes.
- **RMSE**, EUR/MWh. Reported alongside. The gap between MAE and RMSE
  measures spike behaviour and is itself a finding.
- **Skill score**, `1 - MAE_model / MAE_B1`.

Probabilistic forecasts, at quantiles 0.1 through 0.9 in steps of 0.1:

- **Pinball loss**, averaged across quantiles.
- **Empirical coverage** of the nominal 80% interval, q0.1 to q0.9.
- **Mean interval width**, EUR/MWh. Coverage is trivially achieved by wide
  intervals, so width is reported with it.

**MAPE and sMAPE are excluded.** The price crosses zero in 3.9% of hours and
is negative in 3.61%, so percentage error is undefined or unbounded. This is
a property of the market, not a preference.

---

## 7. Split protocol

**Locked test set.** Local 2025-09-01 00:00 to 2026-08-31 23:00, the last 12
complete months. Approximately 8,760 hours. Opened once, at section 12, after
the champion is selected. The ragged days after 2026-08-31 are not used for
evaluation.

**Backtest window.** Local 2018-10-01 to 2025-08-31. All model development,
tuning and selection happens here.

**Fold structure.** Expanding window, monthly origins.

- Minimum initial training period: 24 months, so the first training set ends
  2020-09-30.
- Fold *m* trains on everything from the sample start to the last hour of
  month *m-1*, then forecasts every day of month *m* under the gate-closure
  protocol.
- First evaluated month: 2020-10. Last: 2025-08. 59 folds.

Retraining is monthly rather than daily. That is a deliberate cost decision
and it is stated because it makes the backtest slightly pessimistic relative
to a system retrained every day.

**No embargo is applied,** and the reason is structural rather than an
oversight: the training set ends at the last delivery hour of month *m-1*,
and the first forecast of month *m* is issued at noon on the last day of
month *m-1*, at which point those prices have already cleared.

---

## 8. Regime boundaries

Fixed from market history, not from inspecting this data.

| Regime | Local window | Anchor |
|---|---|---|
| Pre-crisis | 2018-10-01 to 2021-08-31 | Baseline period before the European gas price escalation |
| Crisis | 2021-09-01 to 2022-12-31 | Sustained TTF gas price rise from autumn 2021, Russian supply curtailment through 2022 |
| Post-crisis | 2023-01-01 onward | Gas prices normalised, EU emergency measures wound down |

**Disclosure.** A yearly aggregate summary of the price series was produced in
section 2 before these boundaries were fixed, so the crisis period was known
to be centred on 2022. The boundaries above are anchored to external market
events rather than to any inflection point identified in this data, but the
prior knowledge is disclosed rather than denied.

**Consequence for reporting.** The locked test window falls entirely inside
the post-crisis regime. The regime comparison is therefore made across
backtest folds, which span all three regimes, and the locked test is a
single-regime final confirmation. These are two distinct claims and are
reported separately. Neither is presented as evidence for the other.

---

## 9. Feature construction rules

The concrete feature list is frozen in section 4 after exploratory analysis,
which is legitimate: feature choice is a modelling decision. These rules
constrain that choice and are fixed now.

1. Every feature carries an availability timestamp at or before issuance.
2. Price lags are at least 24 hours.
3. No column from `config.FORBIDDEN` enters the matrix under any name or
   transformation.
4. Exogenous forecasts are used at their delivery hour, as future-known
   covariates. Subject to the publication timing limitation in section 2a.
5. Linear models use one side of the residual identity, never both.
6. German public holidays enter as calendar features. Holidays vary by
   federal state and affect load, so the treatment is documented explicitly
   rather than left implicit.
7. The feature set is frozen before any backtest fold is run and is not
   revised in response to backtest results.

---

## 10. Model ladder

1. **B1 and B2**, seasonal naive baselines.
2. **ARIMA**, price history only. A diagnostic, not a competitor. It measures
   how much is predictable from price autocorrelation alone.
3. **SARIMA**, adding the 24-hour seasonal cycle. Weekly and annual structure
   enter as deterministic terms, since a seasonal ARIMA can represent one
   seasonal period and this series has three.
4. **SARIMAX**, adding the exogenous forecasts. The gap from step 3 measures
   the value of the fundamentals information.
5. **LightGBM**, one model per delivery hour, quantile objective.
6. **N-HiTS**, via neuralforecast, under a fixed compute budget.

All six are evaluated by the same harness, on the same folds, with the same
metrics. The harness is built before any model beyond the baselines.

---

## 11. Model selection rule

Champion is the model with the lowest mean pinball loss across backtest folds
**restricted to the post-crisis regime**, ties broken by MAE.

Restricting selection to post-crisis is a deliberate choice, made now:
deployment and the locked test both sit in that regime, so selecting on the
full backtest would weight a market structure that no longer exists.
Full-backtest rankings are reported alongside so the effect of this choice is
visible.

Selection uses backtest folds only. The locked test set plays no part in it.

---

## 12. Pre-committed expectations

Recorded before fitting so that confirmation and refutation are equally
informative. These are predictions, not requirements.

1. Locked test MAE will exceed the mean backtest fold MAE, because the test
   window has the highest negative-price frequency in the sample. If test
   performance is better than backtest, that is a signal to investigate, not
   to celebrate.
2. Plain ARIMA will not beat B1.
3. SARIMAX will beat both baselines.
4. LightGBM will beat SARIMAX on MAE, with the advantage concentrated in
   high residual load hours where the merit order is non-linear.
5. N-HiTS will not beat LightGBM. Deep forecasting architectures are
   generally strongest across many related series, and this is a single
   series, which is the least favourable setting for them.
6. Skill score will be lowest in the crisis regime.
7. Gradient boosting quantiles will be under-covered in the tails before
   conformal calibration.
8. SARIMAX may achieve better interval coverage than LightGBM despite worse
   MAE, because its intervals come from an explicit error model. If this
   occurs it is reported as a finding rather than buried.

---

## 13. Pre-committed robustness checks

Run and reported regardless of whether they are favourable.

1. **Leak quantification.** Refit the champion with realized load and
   realized residual load substituted for their forecast counterparts, and
   report the MAE improvement. This measures the size of the leak that the
   gate-closure constraint prevents, converting the project's central claim
   from an assertion into a number.
2. **Regime-restricted refit.** Train the champion on post-2023 data only and
   evaluate on the locked test. Establishes whether the crisis period helps
   or hurts.
3. **Fold dispersion.** Report the full distribution of fold-level MAE, not
   only the mean. A good average across 59 folds can hide catastrophic
   individual months.
4. **Both baselines.** Skill reported against B1 and B2.
5. **Hour-of-day breakdown.** MAE by delivery hour.
6. **Conditional subsets.** MAE conditional on realized negative prices and
   on realized prices above 200 EUR/MWh.

---

## 14. What counts as failure

If the champion does not beat B1 on the locked test set, that is the result
and it is reported as the headline. A forecasting study that honestly reports
a naive baseline winning is more useful than one that keeps searching until
something wins.

If skill collapses entirely in the crisis regime, that is likewise a finding
about the limits of fundamentals-based forecasting under supply shock, and it
is reported as such.

---

## 15. Business valuation

Statistical error metrics do not convey value. Section 13 converts forecast
skill into euros using a battery arbitrage decision rule: charge in the
cheapest hours of the day and discharge in the most expensive, dispatched on
the forecast.

Evaluated over the locked test year under three information sets: the B1
naive forecast, the champion forecast, and perfect foresight. The headline is
the share of perfect-foresight arbitrage value captured.

Rule parameters (capacity, duration, round-trip efficiency, cycles per day)
are fixed in section 13 before results are computed and are documented there.
The dispatch rule uses the forecast quantiles rather than the point forecast
alone, which is what makes the probabilistic work earn its place.

---

## 16. Scope boundaries

Out of scope, deliberately, so that the project ends:

- Other bidding zones
- Intraday and balancing markets
- Sub-hourly resolution
- Model registry, containerisation, serving API, drift-triggered retraining

A daily scheduled run and a dashboard are in scope, because day-ahead
forecasting genuinely is a recurring daily job and a system that only exists
inside a notebook does not represent how the problem is solved.

---

## 17. Section 5 baseline results

Recorded when the harness first ran, before any model beyond the baselines
existed. Placed here rather than in a report so that the state of knowledge at
this point in the project is fixed in git history.

| model | MAE | RMSE | skill vs B1 | pinball | coverage | width |
|---|---|---|---|---|---|---|
| B1 weekly naive | 42.47 | 68.56 | 0.000 | 18.64 | 0.626 | 77.19 |
| B2 daily naive | 32.17 | 52.23 | 0.243 | 14.04 | 0.644 | 60.61 |

59 folds, 2020-10 to 2025-08, 43,091 evaluated hours per model.

**B2 beats B1 by 24%.** Section 5 described B1 as hard to beat and made it the
primary denominator. On this data it is the easier target. The denominator is
not changed: switching it after seeing that it flatters results is precisely
what pre-commitment exists to prevent. The standing requirement to report
skill against both baselines now does real work rather than serving as a
formality, and no headline skill figure is reported against B1 alone.

**Interval coverage is far below nominal.** Against a nominal 80%: 62.6% for
B1 and 64.4% for B2 overall, and 27.7% for B1 within the crisis regime. The
cause is structural rather than a coding error. The residual spread is
unconditional and is fitted on an expanding window dominated by calm early
data, then applied to volatile later months. Post-crisis B1 intervals are
100.96 EUR/MWh wide and still cover only 77.6%, so they are simultaneously too
wide on average and too narrow when it matters.

This was anticipated in direction but not in magnitude. Section 3b showed the
price standard deviation is U-shaped in forecast residual load, 21.67 mid-range
against 37.62 and 65.66 in the tails, so an unconditional spread was always
going to be miscalibrated. A 27.7% realised coverage against an 80% nominal is
a larger failure than that reasoning implied. The conformal calibration in
section 10 is therefore load bearing rather than a refinement, and it must be
conditional on residual load rather than a single global width.

**Fold dispersion justifies robustness check 3.** B1 has mean fold MAE 42.40
against a median of 31.88, a maximum of 141.79 and a standard deviation of
30.48. The distribution is heavily right skewed. Reporting the mean alone
would overstate typical error by roughly a third, in the direction that makes
any later model look better by comparison.

**Skill is a ratio whose denominator also degrades.** B2 records its highest
skill in the crisis regime (0.308) while its absolute MAE is worst there
(55.86), because B1 deteriorates faster than B2 does. A model can therefore
appear to gain skill in the crisis while forecasting it worse in absolute
terms.

Expectation 12.6, that skill will be lowest in the crisis regime, could fail
for this reason alone, with no bearing on model quality. The expectation is
left unedited, since amending a pre-committed prediction after seeing data
that bears on it would void the commitment. Instead, absolute MAE by regime is
reported next to skill by regime throughout, and the verdict on 12.6 is argued
against both figures rather than the ratio alone.

---

## 18. Deviation: how the classical models are fitted

Recorded before section 7 was implemented, after benchmarking rather than
after seeing any result. The model ladder is unchanged: ARIMA, SARIMA and
SARIMAX are all still fitted and all still serve the purposes described in
section 10. What changes is only how the data is arranged when fitting them.

**What was specified.** Section 10 step 3 describes a seasonal ARIMA on the
hourly price series with seasonal period 24, fitted on the expanding window
of section 7.

**Why it cannot be done.** Fit time for that specification grows from 4.0
seconds at 2,000 observations to 14.5 at 5,000 and 16.9 at 10,000. At the
full expanding window of roughly 60,000 hourly observations the process was
terminated by the operating system before completing. This follows from the
dimension of the state space representation, so it is a property of the
specification and not of the machine it was run on.

**The two options.**

A rolling hourly window preserves the 24-hour seasonal period but abandons
the expanding window. That would make the classical models the only ones in
the ladder trained on a different quantity of data, so any later comparison
against LightGBM would confound model specification with training window.

A per-delivery-hour framing fits 24 separate series, one for each delivery
hour, each at daily frequency with roughly 2,500 observations, with seasonal
period 7 for the weekly cycle. It preserves the expanding window. Estimated
cost is 33 to 48 minutes for the full ladder across 59 folds.

**The choice, and the principle behind it.** The per-delivery-hour framing is
adopted. The expanding window is a pre-committed evaluation choice; the
seasonal period is an implementation detail. Where only one of the two can
survive a feasibility constraint, the evaluation choice is kept.

The daily cycle is not discarded. It is represented by estimating a separate
model for each delivery hour rather than by a seasonal term inside one model,
which is the standard construction in the published electricity price
forecasting literature.

**A side effect worth stating.** This aligns the classical models with the
LightGBM construction already planned in section 10 step 5, which is also one
model per delivery hour. Expectation 12.4 therefore compares model classes on
identical data arrangements rather than comparing a pooled model against a
per-hour one.

**Status.** This is a deviation, not a correction. Section 10 as written was
wrong about what was computationally possible, and that is disclosed here
rather than by silently editing section 10. No evaluation choice, metric,
fold structure, regime boundary or pre-committed expectation is altered.

---

## 19. Section 7 results: the classical ladder

Full backtest, 59 folds, 2020-10 to 2025-08, 43,091 evaluated hours per
model. Fitted per delivery hour as recorded in section 18.

| model | MAE | RMSE | bias | skill vs B1 | skill vs B2 | pinball | coverage | width |
|---|---|---|---|---|---|---|---|---|
| B1 weekly naive | 42.47 | 68.56 | -0.18 | 0.000 | -0.320 | 18.64 | 0.626 | 77.19 |
| B2 daily naive | 32.17 | 52.23 | +0.00 | 0.243 | 0.000 | 14.04 | 0.644 | 60.61 |
| ARIMA | 30.73 | 48.19 | -0.91 | 0.277 | 0.045 | 20.37 | 0.514 | 56.34 |
| SARIMA | 27.89 | 43.90 | -0.93 | 0.343 | 0.133 | 20.75 | 0.475 | 49.38 |
| SARIMAX | 20.16 | 35.09 | -0.14 | 0.525 | 0.373 | 15.52 | 0.576 | 41.88 |

### The measured value of the fundamentals

The gap from SARIMA to SARIMAX is what section 7 was built to produce: what
the published day-ahead wind, solar and load forecasts are worth once
autoregressive structure, weekly seasonality and calendar effects are already
in the model.

MAE falls from 27.89 to 20.16, a reduction of 27.7%. By regime the reduction
is 21.2% in the crisis (50.93 to 40.15), 34.0% post-crisis (22.36 to 14.76)
and 34.9% pre-crisis (10.46 to 6.81).

The effect is smallest in the crisis. Fundamentals explain less when the price
level is set by gas scarcity rather than by the domestic merit order, which is
consistent with the mechanism rather than merely with the ranking.

### Verdicts on pre-committed expectations

**12.2 FAILED.** The prediction was that plain ARIMA would not beat B1. ARIMA
records MAE 30.73 against B1's 42.47, a skill of 0.277, and it also edges past
B2 at 32.17. The reasoning behind the expectation was that price
autocorrelation alone carries little information. That was wrong: in the
per-delivery-hour framing of section 18, an ARIMA on the daily series for a
single hour is a considerably stronger object than an ARIMA on the pooled
hourly series would have been, because the daily cycle has already been
removed by construction. The expectation was written before that framing was
adopted and was not revised to match, which is the correct outcome for a
pre-commitment even though it made the prediction easier to falsify.

**12.3 CONFIRMED.** SARIMAX beats both baselines, on every regime, on MAE and
RMSE.

**12.6 CONFIRMED, and without the ambiguity anticipated in section 17.** Skill
is lowest in the crisis for SARIMAX, 0.503 against 0.549 post-crisis and 0.553
pre-crisis. The concern recorded in section 17 was that skill could mislead
because B1 degrades in the crisis as well. Here the two measures agree:
absolute MAE is also worst in the crisis, 40.15 against 14.76 and 6.81. The
verdict does not depend on which measure is used.

### The selection rule cannot yet be applied

Section 11 selects the champion on mean pinball loss. On these results that
rule would choose B2, the daily naive baseline, at 14.04, ahead of SARIMAX at
15.52, despite SARIMAX forecasting 37% better on MAE.

The mechanism is visible in the interval widths. SARIMAX intervals average
41.88 EUR/MWh against B2's 60.61. At the median quantile pinball loss reduces
to half the absolute error, where SARIMAX wins decisively. It loses at the
outer quantiles because its intervals are far too narrow, and mean pinball
averages across all nine.

Coverage confirms this directly: 0.576 for SARIMAX against a nominal 0.80, and
0.192 within the crisis. Every model in the ladder shares the defect, because
all of them derive quantiles from an unconditional spread of training
residuals.

The selection rule is not changed. It is not the rule that is failing but the
inputs to it, and a rule rewritten to avoid an inconvenient answer is not a
pre-commitment. What follows instead is that section 10 is a prerequisite for
section 11 rather than a refinement after it: conformal calibration must
repair the intervals before pinball loss can rank models meaningfully.

### Fold dispersion

| model | mean | median | sd | min | max |
|---|---|---|---|---|---|
| B1 weekly naive | 42.40 | 31.88 | 30.48 | 11.02 | 141.79 |
| B2 daily naive | 32.11 | 26.68 | 18.36 | 9.28 | 79.88 |
| ARIMA | 30.66 | 23.89 | 18.72 | 8.70 | 80.46 |
| SARIMA | 27.85 | 21.40 | 17.39 | 7.56 | 75.80 |
| SARIMAX | 20.12 | 13.89 | 14.74 | 4.82 | 63.45 |

Every distribution is right skewed, so the mean overstates typical error for
all five models. The worst folds are the same months across the ladder,
concentrated in 2022-08, 2022-09, 2021-12 and 2022-03, which are the periods
of steepest price movement during the crisis. No model has an idiosyncratic
failure month, which suggests the difficulty is a property of those periods
rather than of any particular specification.

### A bug found and fixed before these numbers were produced

The first implementation extended the state space models across the test
window with missing endogenous values. The Kalman filter treats missing values
as no observation, so the conditioning state never advanced and each nominal
one-day-ahead forecast was in fact an extrapolation of up to a month from the
fold boundary. It raised no error and produced no warning.

It was caught by the bias column. ARIMA showed +31.9 EUR/MWh, SARIMA +26.1 and
SARIMAX +23.2, against +7.8 and +1.0 for the baselines. A systematic
over-prediction of that size is not a modelling weakness. On the fold
following the August 2022 price peak, ARIMA recorded an MAE of 184.5 where the
corrected figure is 80.5.

Passing the realised series to the filter is not leakage. A one-step-ahead
prediction with dynamic=False conditions on observations through t-1 only, and
in the per-hour framing t-1 is the previous day at the same hour, cleared at
the auction two days before delivery and therefore known at gate closure.
Parameters are estimated on training data alone; only the filter state
advances.

A regression test in tests/test_backtest.py asserts that the conditioning
state moves, and the failure mode is documented in the module rather than
silently corrected.

---

## 20. Section 8 results: gradient boosting on the price level

Full backtest, 59 folds, same protocol and same folds as section 19.
LightGBM fitted per delivery hour, nine quantile models per hour, point
forecast taken as the median quantile.

| model | MAE | RMSE | bias | skill vs B1 | pinball | coverage | width |
|---|---|---|---|---|---|---|---|
| SARIMAX | 20.16 | 35.09 | -0.14 | 0.525 | 15.52 | 0.576 | 41.88 |
| SARIMA | 27.89 | 43.90 | -0.93 | 0.343 | 20.75 | 0.475 | 49.38 |
| ARIMA | 30.73 | 48.19 | -0.91 | 0.277 | 20.37 | 0.514 | 56.34 |
| B2 daily naive | 32.17 | 52.23 | +0.00 | 0.243 | 14.04 | 0.644 | 60.61 |
| **LightGBM** | **41.35** | **77.34** | **-29.29** | **0.026** | 18.79 | 0.543 | 47.12 |
| B1 weekly naive | 42.47 | 68.56 | -0.18 | 0.000 | 18.64 | 0.626 | 77.19 |

LightGBM places second from last, ahead only of the weaker baseline.

### The aggregate hides the mechanism

| regime | MAE | bias | coverage |
|---|---|---|---|
| pre-crisis | 13.36 | -10.04 | 0.466 |
| crisis | 108.97 | -104.30 | 0.192 |
| post-crisis | 17.16 | +1.59 | 0.745 |

Post-crisis the model is genuinely competitive: MAE 17.16 beats ARIMA at
24.71 and SARIMA at 22.36, and its coverage of 0.745 is the best of any
model in the ladder at that point. In the crisis the bias of -104.30 is
almost exactly equal to the MAE of 108.97, meaning the model under-predicts
in nearly every hour rather than erring in both directions.

### Diagnosis: trees cannot extrapolate

Distribution of crisis-regime predictions against outturn, 11,685 hours:

| | outturn | prediction |
|---|---|---|
| mean | 218.2 | 113.9 |
| std | 134.4 | 72.3 |
| 75th percentile | 282.7 | 153.5 |
| max | 871.0 | 418.6 |

A gradient boosting model predicts leaf averages, so it cannot emit a value
above the maximum of its training target. The expanding training window
reaching into 2021 and 2022 contained prices mostly between 30 and 100
EUR/MWh, and the crisis moved outturn far outside that range. The prediction
ceiling of 418.6 against an outturn maximum of 871, together with a predicted
standard deviation roughly half the realised one, is that ceiling made
visible.

SARIMAX is not subject to it. A linear model extrapolates freely, which turns
out to be the correct inductive bias precisely when the target leaves the
range it was estimated on. This is a sharper version of the section 19
finding than the MAE ranking alone conveys: SARIMAX does not merely fit
better, it fails more gracefully when the market does something it has not
seen.

### Verdicts on pre-committed expectations

**12.4 REFUTED.** The prediction was that LightGBM would beat SARIMAX on MAE,
with the advantage concentrated in high residual load hours where the merit
order is non-linear. It loses overall, 41.35 against 20.16, and collapses in
the crisis. The reasoning behind the expectation was sound as far as it went:
section 3b did establish convexity in the price response to residual load,
with a 6.6x ratio between the smallest and largest decile-to-decile step, and
trees do capture that better than a linear specification does. What the
expectation missed is that capturing curvature within the observed range is a
different capability from extrapolating beyond it, and that the second
mattered more over this sample.

**12.7 REFUTED.** The prediction was that gradient boosting quantiles would
be under-covered in the tails before conformal calibration. Post-crisis
LightGBM records coverage of 0.745 against a nominal 0.80, better than
SARIMAX at 0.786 in width terms and better than every classical model overall
in the regime where its point forecast works. Conditional quantiles behaved
roughly as intended. The interval problem in this ladder is not specific to
gradient boosting, it is common to every model that derives spread from an
unconditional residual distribution.

### What follows, and what does not

A variant is added in section 8b that predicts the difference from the B2
naive baseline rather than the price level, so the target stays inside the
training range even when the price does not. The level version is retained in
every table and is not replaced. Reporting only the parameterisation that
worked, after seeing that the first one failed, would be the exact
substitution this document exists to prevent.

Expectation 12.4 remains refuted regardless of how the anchored variant
performs. It named a model class and a comparison, and that comparison has
been made. If the anchored version does beat SARIMAX, the finding is that the
target parameterisation mattered more than the model class, which is a more
useful result than the original prediction would have been had it simply held.

---

## 21. Section 9 results: N-HiTS, and the full ladder

Full backtest, 59 folds, same protocol and same folds as sections 19 and 20.
43,091 evaluated hours per model.

| model | MAE | RMSE | bias | skill vs B1 | pinball | coverage | width |
|---|---|---|---|---|---|---|---|
| **N-HiTS** | **18.88** | **32.20** | -0.83 | **0.563** | **7.71** | **0.693** | 46.66 |
| SARIMAX | 20.16 | 35.09 | -0.14 | 0.525 | 15.52 | 0.576 | 41.88 |
| LightGBM anchored | 23.91 | 41.64 | -1.76 | 0.437 | 10.06 | 0.646 | 49.94 |
| SARIMA | 27.89 | 43.90 | -0.93 | 0.343 | 20.75 | 0.475 | 49.38 |
| ARIMA | 30.73 | 48.19 | -0.91 | 0.277 | 20.37 | 0.514 | 56.34 |
| B2 daily naive | 32.17 | 52.23 | +0.00 | 0.243 | 14.04 | 0.644 | 60.61 |
| LightGBM level | 41.35 | 77.34 | -29.29 | 0.026 | 18.79 | 0.543 | 47.12 |
| B1 weekly naive | 42.47 | 68.56 | -0.18 | 0.000 | 18.64 | 0.626 | 77.19 |

N-HiTS leads on every metric that ranks models: MAE, RMSE, skill, pinball
loss and coverage. It also has the tightest fold dispersion, standard
deviation 11.40 against SARIMAX at 14.74, and the lowest worst fold, 56.84
against 63.45.

### Verdict on the pre-committed expectation

**12.5 REFUTED.** The prediction was that N-HiTS would not beat LightGBM,
reasoning that deep forecasting architectures are strongest across many
related series and that a single series is their least favourable setting.
It beats both LightGBM variants, and everything else in the ladder.

The reasoning was not obviously wrong in general, and it is a common finding
on tabular forecasting problems. What it underweighted is that a single
series observed hourly for eight years is not a small-data problem in the way
that phrasing suggests. Nearly 60,000 observations with strong repeated
structure is ample for a 4 million parameter model, particularly one whose
inductive bias, multi-rate decomposition of a periodic signal, matches the
structure section 3b measured directly.

### The interval result matters more than the point forecast

The margin on MAE over SARIMAX is 6%. The margin on pinball loss is 50%, and
the coverage difference is larger still.

Coverage within the crisis regime, where every other model failed:

| model | crisis coverage | crisis width | post-crisis width |
|---|---|---|---|
| N-HiTS | 0.683 | 80.84 | 38.07 |
| LightGBM anchored | 0.478 | 69.51 | 50.16 |
| SARIMAX | 0.192 | 36.52 | 53.91 |
| SARIMA | 0.153 | 39.12 | 63.57 |

N-HiTS roughly doubles its interval width in the crisis relative to
post-crisis, from 38.07 to 80.84. SARIMAX moves in the opposite direction,
narrowing from 53.91 to 36.52 in precisely the period where uncertainty was
greatest, which is what an unconditional residual spread fitted on an
expanding window will do when recent history is calmer than the present.

This is the first genuinely conditional uncertainty estimate in the project.
Section 3b established that price variance is U-shaped in forecast residual
load, 21.67 EUR/MWh mid-range against 37.62 and 65.66 in the tails, and
argued that intervals would have to respond to that. Only this model does.

### Two caveats that qualify the result

**The framing is not held constant.** Every other model in the ladder is
fitted per delivery hour, per section 18. N-HiTS is one model on the hourly
sequence, per section 10 step 6. Part of the margin may come from seeing the
sequence directly rather than from the architecture, and this experiment
cannot separate the two. Testing that would require refitting a classical
model on the pooled hourly series, which section 18 established is not
computationally feasible, or fitting N-HiTS per hour, which would discard the
structure it exists to exploit. The ambiguity is recorded rather than
resolved.

**Coverage is still short of nominal.** 0.693 against 0.80. Best in the
ladder and still under-covered by 11 percentage points. Section 10 remains
necessary; it is no longer the difference between usable and unusable
intervals.

### Consequences for section 11

Section 11 selects on mean pinball loss restricted to post-crisis folds. On
these results N-HiTS leads post-crisis pinball at 6.72 against SARIMAX at
7.54 and LightGBM anchored at 7.10, so the selection rule and the MAE
ranking now agree. The disagreement recorded in section 20, where MAE and
pinball pointed at different models, has resolved without any rule being
changed.

Section 10 runs before section 11 regardless. Conformal calibration may
alter the ranking, and applying a selection rule to intervals known to be
miscalibrated would make the choice arbitrary even when the answer looks
stable.

---

## 22. Section 10 results: conformal calibration

Calibration uses the stored backtest predictions rather than a fresh
hold-out. Fold m is corrected using folds m-12 to m-1, all of which closed
before fold m began. 53 of 59 folds are calibrated; the first six lack enough
history and are excluded, and all three variants are compared on the same
38,723 hours so the difference is calibration and not a different sample.

### Overall, nominal coverage 0.80

| model | coverage | width | pinball | coverage | width | pinball |
|---|---|---|---|---|---|---|
| | *uncalibrated* | | | *conformal global* | | |
| N-HiTS | 0.696 | 50.11 | 8.29 | **0.799** | **61.31** | 8.23 |
| LightGBM anchored | 0.642 | 53.79 | 10.94 | 0.777 | 83.51 | 10.78 |
| B2 daily naive | 0.630 | 63.99 | 15.11 | 0.775 | 112.87 | 14.67 |
| B1 weekly naive | 0.617 | 82.16 | 20.11 | 0.776 | 152.61 | 19.40 |
| SARIMAX | 0.573 | 45.00 | 16.95 | 0.733 | 122.87 | 16.66 |
| SARIMA | 0.466 | 52.41 | 22.53 | 0.737 | 162.42 | 21.54 |
| ARIMA | 0.498 | 59.78 | 22.11 | 0.731 | 161.12 | 21.21 |
| LightGBM level | 0.525 | 50.75 | 20.67 | 0.685 | 120.74 | 17.85 |

### Only one model calibrates cleanly

N-HiTS reaches 0.799 against a nominal 0.800 for a 22% increase in width,
50.11 to 61.31. Nothing else converges. SARIMAX requires intervals 2.7 times
wider, 45.00 to 122.87, and still reaches only 0.733. SARIMA requires 3.1
times wider for 0.737.

The mechanism is that a conformal correction is marginal. It shifts each
quantile by a constant and can therefore repair the average, but it cannot
repair the shape of an interval that fails to respond to conditions. A model
whose spread is the same on every day must be widened enough to cover its
hard days, which leaves it absurdly wide on its easy ones, and the marginal
target is still missed because the correction is fitted on a mixture the
model cannot distinguish.

Section 21 recorded that N-HiTS was the only model whose width tracked the
regime, roughly doubling in the crisis while SARIMAX narrowed. This section
is the consequence of that difference stated in coverage terms: conditional
intervals can be corrected, unconditional ones can only be inflated.

### The conditional variant did not work, contrary to the section 3b reasoning

Conditional calibration bins the calibration residuals by forecast residual
load, on the reasoning from section 3b that price variance is U-shaped in
that variable: 21.67 EUR/MWh in the middle deciles against 37.62 and 65.66 at
the ends.

It does not beat the global variant on coverage. N-HiTS records 0.799 under
both. SARIMAX is worse conditional than global, 0.715 against 0.733, as is
ARIMA, 0.711 against 0.731. Conditional is slightly better on pinball loss
for every model, but slightly worse on the coverage it was introduced to fix.

The expectation was wrong in a specific and instructive way. Section 3b
measured the variance of the *price* against residual load, and that
measurement stands. What the conditional scheme needs is different: that the
*models' forecast errors* are miscalibrated along the same axis. Those are
not the same claim. A model that already uses residual load as a feature,
which all of these do, has had the opportunity to absorb that structure into
its point forecast, so the residual miscalibration lies elsewhere.

Both variants are reported. Global is adopted as the calibration method for
section 11, on coverage, which is what the section was for.

### The crisis remains uncovered

Coverage within the crisis regime after global calibration:

| model | uncalibrated | calibrated | width after |
|---|---|---|---|
| N-HiTS | 0.683 | 0.753 | 93.27 |
| LightGBM anchored | 0.478 | 0.684 | 134.63 |
| B2 daily naive | 0.367 | 0.652 | 134.98 |
| SARIMAX | 0.192 | 0.499 | 149.09 |
| SARIMA | 0.153 | 0.472 | 166.56 |

No model reaches nominal coverage in the crisis, and the best is N-HiTS at
0.753. This is a limitation of the method rather than a defect in the
implementation. A rolling twelve month calibration window can only correct
for error behaviour it has already seen, and the first months of the crisis
had no precedent in the preceding year. The alternative, a calibration window
long enough to contain a comparable episode, would mean correcting today's
forecasts with residuals from a market structure that no longer exists.

Stated plainly: these intervals are trustworthy in a market resembling the
recent past and are not trustworthy through a structural break. That is worth
saying explicitly rather than reporting an aggregate coverage figure that
averages the two situations.

### Consequence for section 11

Selection uses conformal-global calibrated predictions restricted to
post-crisis folds. The ranking on calibrated post-crisis pinball loss decides
the champion, and section 12 opens the locked test set once, with the
champion and its calibration procedure both fixed.

---

## 23. Section 11 result: champion selection

The rule from section 11, applied unchanged: lowest mean pinball loss across
backtest folds restricted to the post-crisis regime, ties broken by MAE, on
conformal global calibrated predictions per section 22. 53 calibrated folds,
23,369 post-crisis hours, eight models.

### Ranking on the selection criterion

| model | MAE | RMSE | pinball | coverage | width |
|---|---|---|---|---|---|
| **N-HiTS** | 16.31 | 28.30 | **6.67** | 0.814 | 49.63 |
| LightGBM anchored | 17.64 | 28.76 | 7.19 | 0.832 | 66.07 |
| SARIMAX | 14.76 | 24.59 | 8.90 | 0.862 | 125.25 |
| LightGBM level | 17.16 | 27.27 | 9.92 | 0.796 | 102.21 |
| B2 daily naive | 26.92 | 41.53 | 11.50 | 0.845 | 113.39 |
| ARIMA | 24.71 | 36.82 | 13.98 | 0.870 | 172.20 |
| B1 weekly naive | 32.69 | 50.65 | 14.17 | 0.859 | 154.59 |
| SARIMA | 22.36 | 33.79 | 14.58 | 0.866 | 179.75 |

**Champion: N-HiTS**, ahead of LightGBM anchored by 0.520 pinball.

### The regime restriction did not change the answer

Section 11 committed to selecting on post-crisis folds only, on the grounds
that deployment and the locked test both sit in that regime and that the full
backtest would weight a market structure that no longer exists. That
commitment was made before any model was fitted.

On these results the restriction makes no difference: N-HiTS leads both
rankings. The restriction was worth committing to in advance and turned out
not to bind, which is the ordinary outcome for a precaution.

### A note on why SARIMAX has the best post-crisis MAE and finishes third

SARIMAX records the lowest post-crisis MAE in the table, 14.76 against 16.31
for N-HiTS, and still ranks third on the selection criterion. The reason is
in the last column. Its calibrated intervals average 125.25 EUR/MWh against
49.63 for N-HiTS, two and a half times wider for slightly better central
accuracy, and pinball loss charges for that width across all nine quantiles.

This is the selection rule working as intended rather than a quirk. Section 6
chose pinball loss precisely because a point forecast alone does not describe
a forecast that will be used to make a decision under uncertainty, and
section 13 will dispatch a battery on these intervals. A model that is
marginally more accurate at the median while being unable to say how
uncertain it is would be the wrong choice for that purpose.

Had the rule been MAE, the champion would have been SARIMAX. That is stated
here so the dependence of the outcome on a pre-committed choice is visible
rather than implied.

### Post-calibration coverage now overshoots outside the crisis

Post-crisis coverage sits above nominal for most models: 0.870 for ARIMA,
0.866 for SARIMA, 0.862 for SARIMAX, 0.814 for N-HiTS, against a target of
0.80.

This is the section 22 finding seen from the other side. The conformal
correction is fitted across all regimes, so a single global width sized to
cover crisis errors necessarily over-covers the calmer period. The overall
figures near 0.80 in section 22 are an average of over-coverage post-crisis
and under-coverage of 0.47 to 0.50 during the crisis, not uniform calibration.

N-HiTS is the least affected in both directions, 0.814 post-crisis and 0.753
in the crisis, while the models with unconditional spreads swing between
roughly 0.47 and 0.87. That is a further consequence of conditional intervals
being correctable and unconditional ones only being inflatable, and it did
not enter the selection rule, which considered pinball loss alone.

### State before section 12

The champion is fixed and recorded in reports/champion.json, together with
both rankings and the flag locked_test_opened: false. The calibration
procedure is fixed. The locked test set has not been read.

Section 12 opens it once.

---

## 24. Section 12: the locked test result

Opened once, on 10 September 2026. Champion, calibration method, metrics and
split protocol were all fixed beforehand and are recorded in sections 6, 7,
11 and 22. 8,757 delivery hours, 2025-09 to 2026-08, twelve monthly folds
under the same expanding-window protocol as the backtest.

### Headline

| model | MAE | RMSE | bias | skill vs B1 | pinball | coverage | width |
|---|---|---|---|---|---|---|---|
| **N-HiTS, champion** | **23.45** | 129.97 | -6.52 | **0.320** | 9.29 | 0.821 | 80.65 |
| B2 daily naive | 28.50 | 45.18 | -0.15 | 0.201 | 12.07 | 0.789 | 86.67 |
| B1 weekly naive | 35.68 | 54.91 | -0.58 | 0.000 | 14.98 | 0.780 | 112.68 |
| N-HiTS with realised inputs | 18.75 | 36.80 | -1.42 | 0.457 | 7.61 | 0.808 | 55.10 |
| N-HiTS, post-2023 training | 22.39 | 108.49 | -4.02 | 0.351 | 9.60 | 0.805 | 91.75 |

The champion beats both baselines. Calibrated coverage is 0.821 against a
nominal 0.80, the closest of any model at any point in the project.

The row for realised inputs is the leak measurement, not a forecast. It is
never reported as one.

### Expectation 12.1: CONFIRMED

Predicted, before the test set was opened, that locked test MAE would exceed
the backtest average because the test window contains the highest
negative-price frequency in the sample. Backtest 18.88, locked test 23.45.

The direction was right. The size, a 24% deterioration, is larger than the
reasoning implied.

### Expectation 12.8: REFUTED

Predicted that SARIMAX might achieve better interval coverage than LightGBM
despite worse MAE, because its intervals derive from an explicit error model
rather than from independently fitted quantiles.

The mechanism did not hold. Uncalibrated, SARIMAX covered 0.573 against 0.642
for anchored LightGBM. After calibration SARIMAX required intervals 2.7 times
wider and still reached only 0.733. An explicit error model confers no
advantage when the error variance is not constant, which section 3b had
already established it is not.

### Two failures the locked test found that 59 backtest folds did not

**Eighteen diverged hours.** On 27 and 28 February 2026 the champion emitted
predictions as extreme as -5,276 EUR/MWh against an outturn near zero. The
observed price range across the entire sample is -500 to 936. The inputs for
those days were unremarkable: residual load 1,965 to 46,897 MW, solar 0 to
39,189, outturn -0.8 to 118.1. This is a numerical failure inside the model,
not a response to unusual data.

Eighteen hours out of 8,757, 0.21% of the year, inflate MAE by 22% and RMSE
threefold. Excluding them gives MAE 18.33 and RMSE 42.99 against the reported
23.45 and 129.97.

arima.py has carried a divergence guard since section 7, added after SARIMA
diverged to an MAE of 1.4e10 on one fold. The same guard was not added to
nhits.py. It has been added now, and it is not retroactive.

**A DST cascade costing 837 hours.** The champion produced no forecast for
9.6% of the test year. The cause was a single line: a local delivery day
spans 23 or 25 hours across a DST transition while the N-HiTS horizon is
fixed at 24, and such a day was skipped with a bare `continue`. That also
skipped the statement extending the conditioning history, so the window
stopped advancing and every subsequent day in the fold produced nothing. Two
transitions, on 2025-10-26 and 2026-03-29, silently emptied the remainder of
their folds.

The missing hours were not random: mean outturn 74.83 against 101.59 on the
hours that were predicted, so the champion was scored on the more expensive
subset. Recomputing the baselines on the same 7,920 hours gives B1 34.49 and
B2 28.01 against 35.68 and 28.50, so the comparison changes little and the
skill figure of 0.320 stands. That it happens to be close is luck, not
design.

Both are fixed with regression tests. Neither fix has been evaluated on the
locked test set and neither will be. The reported figures are what the code
as it stood produced. 18.33 is an estimate of what a corrected implementation
would have scored, not a measurement.

**This is the held-out set doing its job.** Both defects survived 59 backtest
folds across eight years. The DST cascade in particular was invisible in
every aggregate metric, because missing predictions reduce the row count
rather than degrade a score. Section 4 established the DST discipline that
would have prevented it and section 9 failed to apply it.

### The six pre-committed robustness checks

**1. Leak quantification.** Substituting realised load and realised residual
load for their published forecasts improves MAE from 23.45 to 18.75, a gain
of 20.0%. That is what the gate closure constraint costs, and what a leaky
implementation would report in its place.

The figure is a lower bound. Wind and solar have no realised counterpart in
the ingested series, so only two of the exogenous inputs were substituted. A
fully leaky implementation would report a larger improvement still.

**2. Training window.** Restricting training to post-2023 data improves MAE
from 23.45 to 22.39. Less data is better here: the crisis period actively
harms the model, which is consistent with section 8's finding that a market
structure no longer present is a poor guide to the current one. The
restricted variant also has lower RMSE, 108.49 against 129.97, since it
diverged less severely.

**3. Month to month dispersion.** Mean 23.65, median 17.67, standard
deviation 20.74, range 9.84 to 87.51. The distribution is dominated by
2026-02 at 87.5, the month containing the divergence; every other month falls
between 9.8 and 28.4. Reporting the mean alone would misrepresent typical
performance by roughly a third.

**4. Both baselines.** Skill 0.320 against B1 and 0.177 against B2. The
weaker denominator, fixed in section 5 before any result existed, flatters
the headline by 14 percentage points.

**5. By delivery hour.** Best: 00h at 8.2, 01h at 10.9, 05h at 13.3. Worst:
13h at 43.2, 12h at 38.1, 14h at 36.7. Error is concentrated in the midday
solar hours, which section 3b identified as where negative prices occur, at
up to 12% of hours between 11h and 15h.

**6. Conditional subsets.**

| subset | n | B1 | B2 | N-HiTS | skill vs B1 |
|---|---|---|---|---|---|
| ordinary, 0 to 200 | 7,997 | 32.27 | 25.83 | **19.31** | 0.385 |
| spike above 200 | 229 | 109.69 | 94.88 | **91.77** | 0.166 |
| negative price | 531 | 55.24 | **40.11** | 60.27 | **-0.258** |

### The champion is beaten by doing nothing on negative prices

On 531 hours, 6.1% of the test year, N-HiTS records MAE 60.27 against 55.24
for the weekly naive baseline and 40.11 for the daily one. Skill is -0.258.
Both baselines forecast negative-price hours better than the champion does.

The leak measurement locates the cause. With realised load substituted, the
same architecture scores 31.88 on those hours, a 47% improvement against 20%
overall. Negative prices arise when residual load approaches zero, and
residual load is the difference between three forecast quantities, so its
proportional error is largest exactly where its level is smallest. The
model's failure on these hours is inherited from the accuracy of its inputs
rather than produced by the model itself.

This is a real limitation and it is worsening. Negative-price frequency has
risen from 1.2% of hours in 2018 to 7.3% in 2026, so the subset on which this
forecaster is worse than a naive baseline is the one growing fastest.

### What the result is

A day-ahead price forecaster that, on a year it had never seen, beats a
weekly naive baseline by 32% and a daily naive baseline by 18% on MAE, with
calibrated 80% intervals achieving 82.1% coverage.

It fails on negative prices, it diverged for eighteen hours, and a DST bug
cost it 9.6% of the evaluation year. All three are reported because the
alternative is a number that reads better and means less.
