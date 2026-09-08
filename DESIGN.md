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
