"""
Exploratory analysis.

Descriptive only. Nothing here selects a model, fixes an evaluation window or
touches the locked test set. Its job is to inform feature construction in
section 4, which is a modelling decision and is allowed to be data driven.

The realized series appear in one figure, comparing published forecasts
against outturn. That is a description of input quality, not a feature. The
realized columns never enter a model.

    py -m dayahead.cli eda
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .data.validate import load_panel

REGIMES = [
    ("pre-crisis", "2018-10-01", "2021-08-31"),
    ("crisis", "2021-09-01", "2022-12-31"),
    ("post-crisis", "2023-01-01", "2026-12-31"),
]

FIG_W, FIG_H = 11, 4.2
DPI = 130


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for EDA.\n"
            "  py -m pip install matplotlib"
        ) from exc
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9,
    })
    return plt


def local_frame() -> pd.DataFrame:
    """Usable rows only, converted to local time, with calendar columns."""
    panel = load_panel()
    df = panel[panel["is_usable"]].tz_convert(cfg.LOCAL_TZ).copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["date"] = df.index.date
    df["regime"] = "unassigned"
    for name, start, end in REGIMES:
        mask = (df.index >= pd.Timestamp(start, tz=cfg.LOCAL_TZ)) & \
               (df.index <= pd.Timestamp(end + " 23:59", tz=cfg.LOCAL_TZ))
        df.loc[mask, "regime"] = name
    return df


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation, computed directly to avoid a dependency."""
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[:-k], x[k:]) / denom
                             for k in range(1, nlags + 1)])


# --------------------------------------------------------------------- plots
def fig_price_history(df, plt):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    p = df[cfg.TARGET].astype("float64")
    ax.plot(p.index, p.values, lw=0.25, alpha=0.4, color="#4a6fa5")
    ax.plot(p.index, p.rolling(24 * 7, min_periods=1).mean().values,
            lw=1.4, color="#c1440e", label="7 day rolling mean")
    for name, start, end in REGIMES:
        a = pd.Timestamp(start, tz=cfg.LOCAL_TZ)
        b = min(pd.Timestamp(end + " 23:59", tz=cfg.LOCAL_TZ), p.index[-1])
        if a < p.index[-1]:
            ax.axvspan(a, b, alpha=0.07,
                       color={"pre-crisis": "green", "crisis": "red",
                              "post-crisis": "blue"}[name])
            ax.text(a + (b - a) / 2, p.max() * 0.94, name, ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("DE/LU day-ahead price, hourly")
    ax.legend(loc="upper left")
    fig.savefig(cfg.FIGURES / "01_price_history.png")
    plt.close(fig)


def fig_seasonal_profiles(df, plt):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W + 2, FIG_H))
    for name, _, _ in REGIMES:
        sub = df[df["regime"] == name]
        if sub.empty:
            continue
        axes[0].plot(sub.groupby("hour")[cfg.TARGET].mean(), marker="o",
                     ms=3, label=name)
        axes[1].plot(sub.groupby("dow")[cfg.TARGET].mean(), marker="o", ms=3)
        axes[2].plot(sub.groupby("month")[cfg.TARGET].mean(), marker="o", ms=3)
    axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("mean EUR/MWh")
    axes[0].set_title("Daily cycle"); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("day of week (0=Mon)"); axes[1].set_title("Weekly cycle")
    axes[2].set_xlabel("month"); axes[2].set_title("Annual cycle")
    fig.suptitle("Three seasonal periods, by regime", y=1.02)
    fig.savefig(cfg.FIGURES / "02_seasonal_profiles.png")
    plt.close(fig)


def fig_acf(df, plt):
    p = df[cfg.TARGET].astype("float64").to_numpy()
    a = acf(p, 200)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.bar(range(len(a)), a, width=0.9, color="#4a6fa5")
    for lag, label in [(24, "24h"), (48, "48h"), (168, "168h")]:
        ax.axvline(lag, color="#c1440e", lw=1, ls="--")
        ax.text(lag, a.max() * 0.9, label, color="#c1440e", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (hours)"); ax.set_ylabel("autocorrelation")
    ax.set_title("Price autocorrelation")
    fig.savefig(cfg.FIGURES / "03_acf.png")
    plt.close(fig)
    return a


def fig_merit_order(df, plt):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W + 2, FIG_H), sharey=True)
    for ax, (name, _, _) in zip(axes, REGIMES):
        sub = df[df["regime"] == name]
        if sub.empty:
            continue
        ax.hexbin(sub["fc_residual"].astype("float64") / 1000,
                  sub[cfg.TARGET].astype("float64"),
                  gridsize=45, mincnt=1, cmap="viridis", bins="log")
        ax.axhline(0, color="w", lw=0.7, ls=":")
        ax.set_title(name); ax.set_xlabel("forecast residual load (GW)")
    axes[0].set_ylabel("price EUR/MWh")
    fig.suptitle("Merit order: price against forecast residual load", y=1.02)
    fig.savefig(cfg.FIGURES / "04_merit_order.png")
    plt.close(fig)


def fig_negative_prices(df, plt):
    neg = df[cfg.TARGET] < 0
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W + 1, FIG_H))
    by_hour = neg.groupby(df["hour"]).mean() * 100
    axes[0].bar(by_hour.index, by_hour.values, color="#4a6fa5")
    axes[0].set_xlabel("hour of day"); axes[0].set_ylabel("% of hours negative")
    axes[0].set_title("Negative prices by hour")

    pivot = (neg.groupby([df["year"], df["month"]]).mean() * 100).unstack()
    im = axes[1].imshow(pivot.to_numpy(dtype=float), aspect="auto",
                        cmap="Reds", origin="lower")
    axes[1].set_xticks(range(12), [str(m) for m in pivot.columns])
    axes[1].set_yticks(range(len(pivot.index)), [str(y) for y in pivot.index])
    axes[1].set_xlabel("month"); axes[1].set_title("% negative, year by month")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], label="%")
    fig.savefig(cfg.FIGURES / "05_negative_prices.png")
    plt.close(fig)


def fig_spikes(df, plt):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W + 1, FIG_H))
    p = df[cfg.TARGET].astype("float64")
    axes[0].hist(p.clip(-200, 500), bins=140, color="#4a6fa5")
    axes[0].set_yscale("log")
    axes[0].axvline(0, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("EUR/MWh (clipped to -200, 500)")
    axes[0].set_title("Price distribution, log count")

    spike = (p > 200).groupby(df["hour"]).mean() * 100
    axes[1].bar(spike.index, spike.values, color="#c1440e")
    axes[1].set_xlabel("hour of day"); axes[1].set_ylabel("% above 200")
    axes[1].set_title("Spikes by hour")
    fig.savefig(cfg.FIGURES / "06_spikes.png")
    plt.close(fig)


def fig_forecast_quality(df, plt):
    """
    How good are the exogenous inputs themselves?

    Descriptive only. The realized series are used here to characterise input
    quality and never as features.
    """
    err_load = (df["fc_load"] - df["load_real"]).astype("float64") / 1000
    err_res = (df["fc_residual"] - df["residual_real"]).astype("float64") / 1000
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W + 1, FIG_H))
    axes[0].hist(err_load.clip(-15, 15), bins=120, alpha=0.7,
                 label="load", color="#4a6fa5")
    axes[0].hist(err_res.clip(-15, 15), bins=120, alpha=0.6,
                 label="residual load", color="#c1440e")
    axes[0].set_xlabel("forecast minus realized (GW)")
    axes[0].set_title("Exogenous input error")
    axes[0].legend()

    monthly = err_res.abs().groupby([df["year"], df["month"]]).mean()
    axes[1].plot(range(len(monthly)), monthly.values, lw=1, color="#c1440e")
    axes[1].set_xlabel("months since sample start")
    axes[1].set_ylabel("mean abs error (GW)")
    axes[1].set_title("Residual load forecast error over time")
    fig.savefig(cfg.FIGURES / "07_input_quality.png")
    plt.close(fig)
    return err_load, err_res


def fig_volatility(df, plt):
    p = df[cfg.TARGET].astype("float64")
    d = p.diff().abs()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(d.index, d.rolling(24 * 30, min_periods=24).mean().values,
            lw=1.2, color="#4a6fa5")
    ax.set_ylabel("mean |hourly change|, EUR/MWh")
    ax.set_title("Volatility clustering, 30 day rolling")
    fig.savefig(cfg.FIGURES / "08_volatility.png")
    plt.close(fig)


# ---------------------------------------------------------------------- main
def run() -> dict:
    plt = _plt()
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    df = local_frame()

    print("=" * 78)
    print("SECTION 3b: EXPLORATORY ANALYSIS")
    print("=" * 78)
    print(f"\nusable rows: {len(df):,}   "
          f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} local")

    print("\n[1] Regime composition")
    for name, _, _ in REGIMES:
        sub = df[df["regime"] == name]
        if sub.empty:
            continue
        p = sub[cfg.TARGET].astype("float64")
        print(f"    {name:<14}{len(sub):>8,} rows   mean {p.mean():>7.2f}   "
              f"sd {p.std():>7.2f}   neg {100 * (p < 0).mean():>5.2f}%   "
              f">200 {100 * (p > 200).mean():>5.2f}%")

    fig_price_history(df, plt)
    fig_seasonal_profiles(df, plt)
    a = fig_acf(df, plt)

    print("\n[2] Autocorrelation at the seasonal lags")
    for lag in (1, 24, 25, 48, 72, 168, 169):
        print(f"    lag {lag:>4}h   {a[lag]:>7.4f}")
    daily = a[24] > a[23] and a[24] > a[25]
    weekly = a[168] > a[167] and a[168] > a[169]
    print(f"    local peak at 24h: {daily}    local peak at 168h: {weekly}")
    if daily and weekly:
        print("    Two seasonal periods are present in the price itself. A")
        print("    seasonal ARIMA carries one, so the weekly and annual cycles")
        print("    enter as deterministic terms in section 7.")
    else:
        print("    Expected seasonal peaks are not both present. Inspect")
        print("    figure 03 before fixing the section 7 specification.")

    fig_merit_order(df, plt)

    print("\n[3] Merit order relationship")
    gaps = {}
    for name, _, _ in REGIMES:
        sub = df[df["regime"] == name]
        if sub.empty:
            continue
        r = sub["fc_residual"].astype("float64")
        p = sub[cfg.TARGET].astype("float64")
        pear = r.corr(p)
        spear = r.corr(p, method="spearman")
        gaps[name] = spear - pear
        print(f"    {name:<14}pearson {pear:>6.3f}   spearman {spear:>6.3f}   "
              f"gap {spear - pear:>+6.3f}")
    print("    Spearman above Pearson indicates a monotone but non-linear")
    print("    relationship, which is where trees gain over a linear model.")
    print("    Spearman below Pearson indicates the opposite: the association")
    print("    is closer to linear than to rank-monotone, which would weaken")
    print("    the expectation in DESIGN.md section 12 point 4.")
    worst = max(gaps, key=lambda k: abs(gaps[k]))
    print(f"    largest gap: {worst} at {gaps[worst]:+.3f}")

    print("\n[4] Price by residual load decile, post-crisis")
    post = df[df["regime"] == "post-crisis"]
    if not post.empty:
        dec = pd.qcut(post["fc_residual"].astype("float64"), 10,
                      labels=False, duplicates="drop")
        tab = post.groupby(dec)[cfg.TARGET].agg(["mean", "std", "count"])
        tab["neg_pct"] = (post[cfg.TARGET] < 0).groupby(dec).mean() * 100
        print(tab.round(2).to_string())
        means = tab["mean"].astype(float).to_numpy()
        steps = np.diff(means)
        print(f"    decile-to-decile steps: min {steps.min():.2f}, "
              f"max {steps.max():.2f}, ratio {abs(steps.max() / steps.min()):.1f}x")
        print("    A large ratio means the price response to residual load is")
        print("    convex, which a linear specification cannot represent. A")
        print("    ratio near 1 means it is close to linear.")

    fig_negative_prices(df, plt)
    fig_spikes(df, plt)

    print("\n[5] Negative price conditions")
    neg = df[df[cfg.TARGET] < 0]
    if len(neg):
        print(f"    negative hours: {len(neg):,} ({100 * len(neg) / len(df):.2f}%)")
        print(f"    mean forecast residual load when negative: "
              f"{neg['fc_residual'].astype('float64').mean() / 1000:,.1f} GW")
        print(f"    mean forecast residual load overall:       "
              f"{df['fc_residual'].astype('float64').mean() / 1000:,.1f} GW")
        print(f"    mean forecast solar when negative: "
              f"{neg['fc_solar'].astype('float64').mean() / 1000:,.1f} GW")
        top = neg.groupby("hour").size().sort_values(ascending=False).head(5)
        print(f"    most negative-prone hours: {list(top.index)}")

    err_load, err_res = fig_forecast_quality(df, plt)
    print("\n[6] Exogenous input quality (descriptive, never a feature)")
    print(f"    load forecast     MAE {err_load.abs().mean():>6.3f} GW   "
          f"bias {err_load.mean():>+6.3f} GW")
    print(f"    residual forecast MAE {err_res.abs().mean():>6.3f} GW   "
          f"bias {err_res.mean():>+6.3f} GW")
    print("    This is the noise floor on the inputs. A price model cannot be")
    print("    more certain about tomorrow than its own drivers are.")

    fig_volatility(df, plt)

    print("\n[7] Volatility")
    d = df[cfg.TARGET].astype("float64").diff().abs()
    for name, _, _ in REGIMES:
        sub = d[df["regime"] == name]
        if len(sub):
            print(f"    {name:<14}mean |hourly change| {sub.mean():>7.2f} EUR/MWh")
    print("    If these differ materially across regimes, volatility is not")
    print("    constant, which is why point forecasts alone are insufficient")
    print("    and why sections 5 and 10 exist.")

    print(f"\n8 figures written to {cfg.FIGURES}")
    print("=" * 78)
    return {"rows": len(df)}
