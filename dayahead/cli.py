"""
Command line interface.

    py -m dayahead.cli ingest [--dry-run]
    py -m dayahead.cli validate
    py -m dayahead.cli report
    py -m dayahead.cli eda
    py -m dayahead.cli features
    py -m dayahead.cli backtest [--models all] [--quick]

Run from the repository root. No installation step required.
"""

from __future__ import annotations

import argparse
import sys

from . import config as cfg


def cmd_ingest(args) -> int:
    from .data.smard import ingest_all

    print("=" * 78)
    print("INGEST")
    print(f"window start: {cfg.WINDOW_START.isoformat()}")
    print(f"raw store:    {cfg.RAW}")
    if args.dry_run:
        print("MODE: dry run, nothing written")
    print("=" * 78)

    manifest = ingest_all(dry_run=args.dry_run)
    if args.dry_run:
        print("\nDry run complete.")
        return 0

    print("\n" + "=" * 78)
    print(f"  {'series':<16}{'stored':>8}{'expect':>8}{'points':>10}{'nulls':>8}")
    problems = []
    for short, r in manifest["series"].items():
        if "error" in r:
            print(f"  {short:<16}  ERROR {r['error']}")
            problems.append(short)
            continue
        flag = "" if r["blocks_stored"] == r["blocks_expected"] else "  INCOMPLETE"
        if flag or r["failed"]:
            problems.append(short)
        print(f"  {short:<16}{r['blocks_stored']:>8}{r['blocks_expected']:>8}"
              f"{r['points']:>10}{r['nulls']:>8}{flag}")
    print(f"\n  elapsed: {manifest['elapsed_seconds']}s")
    if problems:
        print(f"  incomplete: {problems}. Rerun, it fetches only what is missing.")
        return 1
    print("  All series complete.")
    return 0


def cmd_validate(args) -> int:
    from .data.validate import build

    print("=" * 78)
    print("VALIDATE AND ASSEMBLE")
    print("=" * 78)
    panel, report = build(write=True)

    print(f"\n  rows (full):     {report['rows_full']:,}")
    print(f"  cutoff:          {report['cutoff_utc']}")
    print(f"  rows (trimmed):  {report['rows_trimmed']:,}")
    print(f"  rows usable:     {report['rows_usable']:,}")
    print(f"  rows excluded:   {report['rows_excluded']:,}")

    print("\n  interior nulls:")
    for c, n in report["interior_nulls"].items():
        if n:
            print(f"    {c:<16}{n:>5}   {report['interior_null_runs'][c]}")
    if not any(report["interior_nulls"].values()):
        print("    none")

    ident = report["residual_identity"]
    print(f"\n  residual identity exact: {ident['exact']}  "
          f"(corr {ident['corr']:.6f}, max abs diff {ident['max_abs_diff']:.6g})")

    d = report["dst"]
    print(f"  DST: {d['n_23h_days']} short days, {d['n_25h_days']} long days, "
          f"{d['n_other']} unexpected")

    t = report["target_stats"]
    print(f"\n  price: mean {t['mean']:.2f}  sd {t['sd']:.2f}  "
          f"min {t['min']:.2f}  max {t['max']:.2f}")
    print(f"  negative hours: {t['negative_hours']:,} ({t['negative_pct']:.2f}%)")

    print(f"\n  written: {cfg.PANEL}")
    print(f"           {cfg.REPORTS / 'validation_report.json'}")
    print("=" * 78)
    return 0


def cmd_report(args) -> int:
    from .data.validate import load_panel

    panel = load_panel()
    local = panel.tz_convert(cfg.LOCAL_TZ)
    price = local[cfg.TARGET].astype("float64")
    yearly = price.groupby(price.index.year).agg(["count", "mean", "std", "min", "max"])
    yearly["neg_pct"] = price.groupby(price.index.year).apply(
        lambda x: 100 * (x < 0).mean())
    print("\nYearly price summary (local calendar year, EUR/MWh)")
    print(yearly.round(2).to_string())
    print(f"\nusable rows: {int(panel['is_usable'].sum()):,} of {len(panel):,}")
    return 0


def cmd_eda(args) -> int:
    from .eda import run

    run()
    return 0


def cmd_features(args) -> int:
    from .features.build import build_audit, build_features

    print("=" * 78)
    print("SECTION 4: FEATURE MATRIX")
    print("=" * 78)
    X, specs = build_features(verbose=True)

    print(f"\n  rows:      {len(X):,}")
    print(f"  features:  {len(specs)}")
    print(f"  usable:    {int(X['is_usable'].sum()):,}")
    dropped = ~X["is_usable"]
    per_day = X.index[dropped].normalize().value_counts()
    print(f"  dropped:   {int(dropped.sum()):,}  "
          f"({int((per_day == 24).sum())} whole days, "
          f"{int((per_day < 24).sum())} partial)")
    print("             whole days are the 7 day warm-up and the 2020-01-31")
    print("             outage; partials are the DST hours with no counterpart")
    print("             on the source day, which are left missing not filled")

    audit = build_audit(X.index, specs)
    print("\n  gate closure audit, slack in hours before issuance:")
    print(audit.groupby("kind")["min_slack_hours"]
          .agg(["min", "max", "count"]).to_string())
    bad = audit[~audit["admissible"]]
    if len(bad):
        print(f"\n  INADMISSIBLE: {bad['feature'].tolist()}")
        return 1
    print("\n  all features admissible at gate closure")

    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    X.to_parquet(cfg.PROCESSED / "features.parquet")
    audit.to_csv(cfg.REPORTS / "feature_audit.csv", index=False)
    print(f"\n  written: {cfg.PROCESSED / 'features.parquet'}")
    print(f"           {cfg.REPORTS / 'feature_audit.csv'}")
    print("=" * 78)
    return 0


def cmd_backtest(args) -> int:
    import pandas as pd

    from .evaluation.backtest import fold_table, run_backtest, summarise
    from .features.build import build_features
    from .models.naive import all_baselines

    print("=" * 78)
    print("SECTION 5: BACKTEST HARNESS")
    print("=" * 78)

    X, _ = build_features()
    models = all_baselines()
    if args.models == "all":
        from .models.arima import ladder
        models = models + ladder()
    print(f"\n  models: {[m.name for m in models]}")
    preds = run_backtest(X, models, verbose=True, every=args.every)

    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cfg.PROCESSED / "backtest_predictions.parquet")

    overall = summarise(preds)
    by_regime = summarise(preds, by=["regime"])
    by_hour = summarise(preds, by=["hour"])
    per_fold = fold_table(preds)

    overall.to_csv(cfg.REPORTS / "backtest_overall.csv", index=False)
    by_regime.to_csv(cfg.REPORTS / "backtest_by_regime.csv", index=False)
    per_fold.to_csv(cfg.REPORTS / "backtest_by_fold.csv", index=False)

    cols = ["model", "n", "mae", "rmse", "bias", "skill", "pinball",
            "coverage", "interval_width"]
    show = [c for c in cols if c in overall.columns]
    print("\n  overall")
    print(overall[show].round(3).to_string(index=False))

    print("\n  by regime")
    rcols = ["model", "regime"] + [c for c in show if c != "model"]
    print(by_regime[[c for c in rcols if c in by_regime.columns]]
          .round(3).to_string(index=False))

    print("\n  fold dispersion of MAE (DESIGN section 13 check 3)")
    disp = per_fold.groupby("model")["mae"].describe()[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(disp.round(2).to_string())

    print("\n  worst 3 folds per model")
    for name, g in per_fold.groupby("model"):
        worst = g.nlargest(3, "mae")[["fold", "regime", "mae"]]
        print(f"    {name}: " + ", ".join(
            f"{r.fold} {r.mae:.1f}" for r in worst.itertuples()))

    print(f"\n  written: {cfg.PROCESSED / 'backtest_predictions.parquet'}")
    print(f"           reports/backtest_overall.csv, _by_regime.csv, _by_fold.csv")
    print("=" * 78)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="dayahead")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="fetch raw SMARD blocks")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("validate", help="assemble and validate the panel")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("report", help="summarise the built panel")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("eda", help="exploratory analysis and figures")
    p.set_defaults(func=cmd_eda)

    p = sub.add_parser("features", help="build and audit the feature matrix")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("backtest", help="run the rolling origin backtest")
    p.add_argument("--models", choices=["baselines", "all"], default="baselines",
                   help="'all' adds ARIMA, SARIMA and SARIMAX (slow)")
    p.add_argument("--every", type=int, default=1,
                   help="run every Nth fold, for a fast smoke test")
    p.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
