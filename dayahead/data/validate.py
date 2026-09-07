"""
Panel assembly and validation.

Turns the raw block store into one hourly panel and runs the checks that
decide whether the data can be trusted.

The distinction that does the real work here is interior versus trailing
nulls. A null after a series' last observation is the publication frontier
and is expected. A null between the first and last observation is a genuine
hole and has to be accounted for. A raw null count cannot tell them apart.
"""

from __future__ import annotations

import json

import pandas as pd

from .. import config as cfg
from .smard import cached_blocks, read_block


# ------------------------------------------------------------------ assembly
def load_series(short: str) -> tuple[pd.Series, int]:
    """Every cached block for one series as a single UTC-indexed Series."""
    frames = []
    for ts in sorted(cached_blocks(short)):
        points = read_block(short, ts)
        if not points:
            continue
        idx = pd.to_datetime([p[0] for p in points], unit="ms", utc=True)
        vals = pd.array([p[1] for p in points], dtype="Float64")
        frames.append(pd.Series(vals, index=idx, name=short))
    if not frames:
        raise FileNotFoundError(f"no cached blocks for {short}, run ingest first")

    s = pd.concat(frames).sort_index()
    duplicates = int(s.index.duplicated().sum())
    if duplicates:
        # Later blocks are the fresher publication, so keep the last.
        s = s[~s.index.duplicated(keep="last")]
    return s, duplicates


def assemble() -> tuple[pd.DataFrame, dict]:
    """All series on one continuous hourly UTC grid."""
    cols, duplicates = {}, {}
    for short in cfg.ALL_SERIES:
        cols[short], duplicates[short] = load_series(short)

    start = min(s.index.min() for s in cols.values())
    end = max(s.index.max() for s in cols.values())
    grid = pd.date_range(start, end, freq="h", tz="UTC")

    panel = pd.DataFrame(index=grid)
    for short, s in cols.items():
        panel[short] = s.reindex(grid)

    off_grid = {k: int((~v.index.isin(grid)).sum()) for k, v in cols.items()}
    return panel, {"duplicate_timestamps": duplicates,
                   "off_grid_timestamps": off_grid,
                   "grid_rows": int(len(grid))}


# -------------------------------------------------------------------- checks
def interior_nulls(col: pd.Series) -> pd.DatetimeIndex:
    """Nulls strictly between the first and last observation."""
    valid = col.notna()
    if not valid.any():
        return col.index[:0]
    first, last = col[valid].index[0], col[valid].index[-1]
    body = col.loc[first:last]
    return body.index[body.isna()]


def null_runs(idx: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group timestamps into consecutive hourly runs."""
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for t in idx[1:]:
        if (t - prev) == pd.Timedelta(hours=1):
            prev = t
            continue
        out.append((start, prev))
        start = prev = t
    out.append((start, prev))
    return out


def check_residual_identity(panel: pd.DataFrame) -> dict:
    """
    Verify fc_residual == fc_load - wind_on - wind_off - solar.

    The identity closes to about 0.02 MW on quantities around 50,000 MW,
    which is SMARD publishing rounded values rather than a different
    definition. Correlation is 1.000000. The five columns are therefore
    linearly dependent for practical purposes, so any model needing a
    full-rank design matrix uses one side of the identity or the other,
    never both.
    """
    lhs, parts = cfg.RESIDUAL_IDENTITY
    derived = panel[parts[0]].copy()
    for c in parts[1:]:
        derived = derived - panel[c]
    diff = (panel[lhs] - derived).astype("float64")
    return {
        "corr": float(panel[lhs].astype("float64").corr(derived.astype("float64"))),
        "mean_abs_diff": float(diff.abs().mean()),
        "max_abs_diff": float(diff.abs().max()),
        # Tolerance set at 0.05 MW: above observed publication rounding,
        # far below any difference that would signal a different definition.
        "holds": bool(diff.abs().max() < 0.05),
        "tolerance_mw": 0.05,
    }


def check_dst(panel: pd.DataFrame) -> dict:
    """Local day lengths. Expect 23-hour and 25-hour days, nothing else."""
    local = panel.tz_convert(cfg.LOCAL_TZ)
    per_day = local.groupby(local.index.date).size()
    return {
        "n_23h_days": int((per_day == 23).sum()),
        "n_25h_days": int((per_day == 25).sum()),
        "n_other": int((~per_day.isin([23, 24, 25])).sum()),
    }


def complete_cutoff(panel: pd.DataFrame) -> pd.Timestamp:
    """Earliest last-valid timestamp across target and exogenous columns."""
    return min(panel[c].last_valid_index() for c in cfg.CORE)


def mark_usable(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_usable: true where every required column is present.

    Rows stay in the index even when unusable, because SARIMAX and lag
    construction both need an unbroken hourly grid. The known 2020-01-31
    outage is excluded here rather than imputed. Fabricating a published
    forecast that never existed would contradict the premise of the project.
    """
    out = panel.copy()
    out["is_usable"] = out[cfg.CORE].notna().all(axis=1)
    return out


def build(write: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: assemble, check, trim, mark, write."""
    panel, report = assemble()

    report["interior_nulls"] = {
        c: int(len(interior_nulls(panel[c]))) for c in cfg.ALL_SERIES
    }
    report["interior_null_runs"] = {
        c: [(str(a), str(b)) for a, b in null_runs(interior_nulls(panel[c]))]
        for c in cfg.ALL_SERIES if len(interior_nulls(panel[c]))
    }

    cutoff = complete_cutoff(panel)
    trimmed = mark_usable(panel.loc[:cutoff])

    report["cutoff_utc"] = str(cutoff)
    report["rows_full"] = int(len(panel))
    report["rows_trimmed"] = int(len(trimmed))
    report["rows_usable"] = int(trimmed["is_usable"].sum())
    report["rows_excluded"] = int((~trimmed["is_usable"]).sum())
    report["residual_identity"] = check_residual_identity(trimmed)
    report["dst"] = check_dst(trimmed)

    price = trimmed[cfg.TARGET].astype("float64")
    report["target_stats"] = {
        "n": int(len(price)),
        "mean": float(price.mean()), "sd": float(price.std()),
        "min": float(price.min()), "max": float(price.max()),
        "negative_hours": int((price < 0).sum()),
        "negative_pct": float(100 * (price < 0).mean()),
    }

    if write:
        cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
        cfg.REPORTS.mkdir(parents=True, exist_ok=True)
        trimmed.to_parquet(cfg.PANEL)
        panel.to_parquet(cfg.PANEL_FULL)
        with (cfg.REPORTS / "validation_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    return trimmed, report


def load_panel(full: bool = False) -> pd.DataFrame:
    """Read the built panel from disk."""
    path = cfg.PANEL_FULL if full else cfg.PANEL
    if not path.exists():
        raise FileNotFoundError(f"{path} missing, run: py -m dayahead.cli validate")
    return pd.read_parquet(path)
