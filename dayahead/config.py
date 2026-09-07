"""
Central configuration.

Everything that another module might otherwise hardcode lives here: paths,
the series registry, and the market constants that define the gate-closure
rule. Nothing in this file depends on any other module in the package.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

PANEL = PROCESSED / "panel.parquet"
PANEL_FULL = PROCESSED / "panel_full.parquet"

# ----------------------------------------------------------------- SMARD API
SMARD_BASE = "https://www.smard.de/app"
REGION = "DE"
RESOLUTION = "hour"

# Sample start. On 1 October 2018 the DE/AT/LU bidding zone split and DE/LU
# began trading separately, so the price series before this date belongs to a
# different market. The start is a structural break, not a data gap.
WINDOW_START = datetime(2018, 9, 30, 22, 0, tzinfo=timezone.utc)

# Series registry.
#   role: "target"    the thing being forecast
#          "exog"     published day-ahead forecasts, allowed at gate closure
#          "forbidden" realized outturn, never a feature, used only to
#                      quantify the size of the leak in section 12
SERIES: dict[str, dict] = {
    "price_delu":    {"filter": "4169", "role": "target",    "unit": "EUR/MWh",
                      "label": "Marktpreis DE/LU"},
    "fc_gen_total":  {"filter": "122",  "role": "exog",      "unit": "MW",
                      "label": "Prognostizierte Erzeugung Gesamt, Day-Ahead"},
    "fc_wind_on":    {"filter": "123",  "role": "exog",      "unit": "MW",
                      "label": "Prognose Wind Onshore, Day-Ahead"},
    "fc_wind_off":   {"filter": "3791", "role": "exog",      "unit": "MW",
                      "label": "Prognose Wind Offshore, Day-Ahead"},
    "fc_solar":      {"filter": "125",  "role": "exog",      "unit": "MW",
                      "label": "Prognose Photovoltaik, Day-Ahead"},
    "fc_load":       {"filter": "411",  "role": "exog",      "unit": "MW",
                      "label": "Prognostizierter Stromverbrauch, Netzlast"},
    "fc_residual":   {"filter": "4362", "role": "exog",      "unit": "MW",
                      "label": "Prognostizierte Residuallast"},
    "load_real":     {"filter": "410",  "role": "forbidden", "unit": "MW",
                      "label": "Realisierter Stromverbrauch"},
    "residual_real": {"filter": "4359", "role": "forbidden", "unit": "MW",
                      "label": "Realisierte Residuallast"},
}

TARGET = "price_delu"
EXOG = [k for k, v in SERIES.items() if v["role"] == "exog"]
FORBIDDEN = [k for k, v in SERIES.items() if v["role"] == "forbidden"]
CORE = [TARGET] + EXOG
ALL_SERIES = list(SERIES)

# Linear dependence, verified to within 0.02 MW on quantities around
# 50,000 MW, which is publication rounding rather than a different
# definition. Correlation is 1.000000.
#   fc_residual == fc_load - fc_wind_on - fc_wind_off - fc_solar
# Any model with a design matrix that must be full rank uses one side or the
# other, never both.
RESIDUAL_IDENTITY = ("fc_residual", ["fc_load", "fc_wind_on", "fc_wind_off", "fc_solar"])

# ------------------------------------------------------------ market timing
LOCAL_TZ = "Europe/Berlin"

# Day-ahead auction gate closure, local time. A forecast for delivery day D is
# issued at 12:00 local on D-1, so the information set is everything published
# at or before that instant and nothing after.
GATE_CLOSURE_HOUR_LOCAL = 12

# Prices for delivery day D clear at the auction on D-1. So at the moment of
# issuance the most recent price known is for the last hour of D-1, and the
# newest price information is already a full day old in auction terms. A lag
# shorter than this is leakage even though it looks conservative.
MIN_PRICE_LAG_HOURS = 24

FORECAST_HORIZON_HOURS = 24

# ---------------------------------------------------------- known data holes
# Single-day SMARD publication outage on the load forecast, local
# 2020-01-31 00:00 to 23:00. Both sides of the residual identity are missing,
# so the hours cannot be reconstructed from any allowed series. They are not
# imputed: fabricating a published forecast that never existed would
# contradict the premise of the project. The rows stay in the index to keep
# the hourly grid unbroken and are marked unusable instead.
KNOWN_GAPS = [
    {
        "series": ["fc_load", "fc_residual"],
        "start_utc": "2020-01-30T23:00:00Z",
        "end_utc": "2020-01-31T22:00:00Z",
        "hours": 24,
        "handling": "excluded, not imputed",
    },
]


def block_url(filter_id: str, timestamp: int) -> str:
    return (f"{SMARD_BASE}/chart_data/{filter_id}/{REGION}/"
            f"{filter_id}_{REGION}_{RESOLUTION}_{timestamp}.json")


def index_url(filter_id: str) -> str:
    return f"{SMARD_BASE}/chart_data/{filter_id}/{REGION}/index_{RESOLUTION}.json"


def raw_dir(short: str) -> Path:
    return RAW / short
