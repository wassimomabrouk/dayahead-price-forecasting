"""
Exogenous features from the published day-ahead forecasts.

These are future-known covariates: the forecast for delivery hour t is used
at delivery hour t, offset_days = 0. Admissibility rests on the publication
timing discussed in DESIGN.md section 2a.

The derived columns come from section 3b. The decile table showed the price
response to residual load is convex, with a 6.6x ratio between the smallest
and largest decile-to-decile step, and that negative prices are a solar
phenomenon: residual load averages 3.4 GW when the price is negative against
32.9 GW overall, with solar at 23.3 GW.

That is why solar appears both inside residual load and as its own column,
and why renewable share is constructed explicitly. A tree can find such
interactions, but giving it the physically meaningful quantity directly costs
nothing and helps the linear models, which cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from .gate import FeatureSpec

DIRECT = ["fc_residual", "fc_load", "fc_wind_on", "fc_wind_off", "fc_solar",
          "fc_gen_total"]


def build(panel_local: pd.DataFrame,
          index: pd.DatetimeIndex) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    p = panel_local.reindex(index)
    df = pd.DataFrame(index=index)
    specs: list[FeatureSpec] = []

    def add(name: str, values, source: str, note: str = ""):
        df[name] = values
        specs.append(FeatureSpec(name=name, kind="exog", offset_days=0,
                                 source=source, note=note))

    for c in DIRECT:
        add(f"x_{c}", p[c].astype("float64").to_numpy(), c,
            "published day-ahead forecast at this delivery hour")

    wind = (p["fc_wind_on"] + p["fc_wind_off"]).astype("float64")
    add("x_wind_total", wind.to_numpy(), "fc_wind_on", "onshore plus offshore")

    res = (wind + p["fc_solar"].astype("float64"))
    add("x_renewables", res.to_numpy(), "fc_solar", "wind plus solar")

    load = p["fc_load"].astype("float64")
    share = np.divide(res.to_numpy(), load.to_numpy(),
                      out=np.full(len(index), np.nan),
                      where=load.to_numpy() > 0)
    add("x_renewable_share", share, "fc_load", "renewables over load")

    # Ramp within the delivery day, both known at issuance since both come
    # from the same published forecast.
    resid = p["fc_residual"].astype("float64")
    add("x_residual_ramp_1h", resid.diff().to_numpy(), "fc_residual",
        "change in forecast residual load from the previous hour")

    # Position of this hour's residual load within its own delivery day.
    by_day = resid.groupby(resid.index.normalize())
    day_min = by_day.transform("min")
    day_max = by_day.transform("max")
    span = (day_max - day_min).to_numpy()
    pos = np.divide((resid - day_min).to_numpy(), span,
                    out=np.full(len(index), np.nan), where=span > 0)
    add("x_residual_day_position", pos, "fc_residual",
        "where this hour sits in the delivery day's residual load range")
    add("x_residual_day_max", day_max.to_numpy(), "fc_residual",
        "highest forecast residual load on the delivery day")
    add("x_residual_day_min", day_min.to_numpy(), "fc_residual",
        "lowest forecast residual load on the delivery day")

    # Section 3b showed low residual load is where negative prices live and
    # where variance is largest. A direct indicator helps the linear models
    # represent a region the merit order treats very differently.
    add("x_residual_below_10gw", (resid < 10_000).astype(float).to_numpy(),
        "fc_residual", "flag for the low residual load regime")

    return df, specs
