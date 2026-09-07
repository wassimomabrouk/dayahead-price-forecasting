# Design specification

Written before any modelling. Committed before the test set is opened.

**Status: draft. Section 3 fills this in and freezes it.**

## Data facts established in sections 0 to 2

- Sample: 2018-10-01 to present, hourly, 69,576 rows at the section 2 cutoff.
- Start date is the DE/AT/LU bidding zone split, a structural market break.
- `fc_residual` equals `fc_load - fc_wind_on - fc_wind_off - fc_solar`
  exactly, to floating point. The five columns are linearly dependent.
- One 24-hour publication outage on the load forecast, local 2020-01-31.
  Excluded, not imputed.
- 8 short and 8 long local days, DST handled via Europe/Berlin.
- Negative prices in 3.61% of hours, rising from 1.2% in 2018 to 7.3% in
  2026. MAPE is unusable.

## To be frozen in section 3

- Baseline definition
- Metric set
- Fold structure for the rolling origin backtest
- Regime boundaries
- Locked test window
- Feature set
