"""
Command line interface.

    py -m dayahead.cli ingest [--dry-run]
    py -m dayahead.cli validate
    py -m dayahead.cli report
    py -m dayahead.cli eda
    py -m dayahead.cli features
    py -m dayahead.cli backtest [--models baselines|classical|gbm|gbm-anchored|nhits|all]
    py -m dayahead.cli conformal
    py -m dayahead.cli select
    py -m dayahead.cli locked-test   (opens the held-out year, once)
    py -m dayahead.cli value

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
    print(f"\n  residual identity holds: {ident['holds']}  "
          f"(corr {ident['corr']:.6f}, max abs diff {ident['max_abs_diff']:.4g} MW, "
          f"tolerance {ident['tolerance_mw']} MW)")

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

    models = []
    if args.models in ("baselines", "all"):
        models += all_baselines()
    if args.models in ("classical", "all"):
        from .models.arima import ladder
        models += ladder()
    if args.models in ("gbm", "all"):
        from .models.gbm import lightgbm
        models += [lightgbm()]
    if args.models in ("gbm-anchored", "all"):
        from .models.gbm import lightgbm_anchored
        models += [lightgbm_anchored()]
    if args.models in ("nhits", "all"):
        from .models.nhits import nhits
        models += [nhits()]
    if not models:
        print("  nothing to run")
        return 1
    print(f"\n  models: {[m.name for m in models]}")
    preds = run_backtest(X, models, verbose=True, every=args.every)

    cfg.PROCESSED.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS.mkdir(parents=True, exist_ok=True)

    # Predictions from models not in this run are reused rather than
    # recomputed, so a 45 minute LightGBM run does not require a two hour
    # SARIMAX rerun to produce a combined table. Rows for models that were
    # just run are always replaced, never merged, so a stale result cannot
    # survive a rerun of the same model. --fresh discards the cache entirely.
    store = cfg.PROCESSED / "backtest_predictions.parquet"
    just_run = set(preds["model"])
    if store.exists() and not args.fresh:
        old = pd.read_parquet(store)
        kept = old[~old["model"].isin(just_run)]
        if len(kept):
            print(f"  reusing cached predictions for "
                  f"{sorted(set(kept['model']))}")
            preds = pd.concat([kept, preds], ignore_index=True)
    preds.to_parquet(store)

    n_folds_by_model = preds.groupby("model")["fold"].nunique()
    if n_folds_by_model.nunique() > 1:
        print("\n  WARNING: models were run over different numbers of folds")
        print(n_folds_by_model.to_string())
        print("  The comparison below is not like for like. Rerun with "
              "--fresh, or rerun the short models over the full set.")

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


def cmd_conformal(args) -> int:
    import pandas as pd

    from .evaluation.conformal import calibrate_all, coverage_table
    from .features.build import build_features

    print("=" * 78)
    print("SECTION 10: CONFORMAL CALIBRATION")
    print("=" * 78)

    store = cfg.PROCESSED / "backtest_predictions.parquet"
    if not store.exists():
        print("  no backtest predictions. Run: py -m dayahead.cli backtest")
        return 1
    preds = pd.read_parquet(store)

    X, _ = build_features()
    features = X.copy()
    features.index.name = "timestamp"

    print(f"\n  models: {sorted(preds['model'].unique())}")
    print(f"  rows:   {len(preds):,}")

    glob, info_g = calibrate_all(preds, conditional=False)
    cond, info_c = calibrate_all(preds, features, conditional=True)

    any_meta = next(iter(info_g.values()))
    print(f"  calibrated folds: {any_meta['folds_calibrated']} of "
          f"{preds['fold'].nunique()}  "
          f"(first {len(any_meta['folds_skipped'])} lack enough history)")

    table = coverage_table(preds, glob, cond)
    order = ["uncalibrated", "conformal_global", "conformal_conditional"]
    table["variant"] = pd.Categorical(table["variant"], order, ordered=True)
    table = table.sort_values(["model", "variant"])

    print("\n  coverage against nominal 0.80, and interval width")
    print(table[["model", "variant", "n", "coverage", "width", "pinball"]]
          .round(3).to_string(index=False))

    by_regime = coverage_table(preds, glob, cond, by=["regime"])
    by_regime["variant"] = pd.Categorical(by_regime["variant"], order, ordered=True)
    by_regime = by_regime.sort_values(["model", "regime", "variant"])
    by_regime.to_csv(cfg.REPORTS / "conformal_by_regime.csv", index=False)
    table.to_csv(cfg.REPORTS / "conformal_overall.csv", index=False)

    print("\n  crisis regime only")
    crisis = by_regime[by_regime["regime"] == "crisis"]
    print(crisis[["model", "variant", "coverage", "width", "pinball"]]
          .round(3).to_string(index=False))

    # The global variant is the one adopted in DESIGN.md section 22, on
    # coverage, so that is what downstream sections consume. The conditional
    # variant is kept beside it for inspection rather than discarded, since
    # its failure to beat global is itself a reported result.
    glob.to_parquet(cfg.PROCESSED / "calibrated_predictions.parquet")
    cond.to_parquet(cfg.PROCESSED / "calibrated_predictions_conditional.parquet")
    print(f"\n  written: reports/conformal_overall.csv, _by_regime.csv")
    print(f"           {cfg.PROCESSED / 'calibrated_predictions.parquet'}")
    print("=" * 78)
    return 0


def cmd_select(args) -> int:
    """
    Apply the selection rule fixed in DESIGN.md section 11.

    Champion is the lowest mean pinball loss across backtest folds restricted
    to the post-crisis regime, ties broken by MAE, computed on conformal
    global calibrated predictions per section 22.

    The rule is not re-derived here and is not adjustable from the command
    line. Selection that can be steered by a flag is not a pre-commitment.
    """
    import json

    import pandas as pd

    from .evaluation.backtest import summarise
    from .evaluation.conformal import calibrate_all

    SELECTION_REGIME = "post-crisis"

    print("=" * 78)
    print("SECTION 11: MODEL SELECTION")
    print("=" * 78)

    store = cfg.PROCESSED / "backtest_predictions.parquet"
    if not store.exists():
        print("  no backtest predictions. Run: py -m dayahead.cli backtest")
        return 1
    preds = pd.read_parquet(store)

    calibrated, _ = calibrate_all(preds, conditional=False)
    print(f"\n  models:     {calibrated['model'].nunique()}")
    print(f"  folds:      {calibrated['fold'].nunique()} calibrated")
    print(f"  criterion:  mean pinball loss, {SELECTION_REGIME} folds only")
    print("  calibration: conformal global")

    post = calibrated[calibrated["regime"] == SELECTION_REGIME]
    ranking = summarise(post).sort_values(["pinball", "mae"]).reset_index(drop=True)
    full = summarise(calibrated).sort_values(["pinball", "mae"]).reset_index(drop=True)

    cols = ["model", "n", "mae", "rmse", "pinball", "coverage", "interval_width"]
    show = [c for c in cols if c in ranking.columns]

    print(f"\n  ranking on {SELECTION_REGIME} folds, the selection criterion")
    print(ranking[show].round(3).to_string(index=False))

    print("\n  ranking on all folds, for comparison only")
    print(full[show].round(3).to_string(index=False))

    champion = ranking.iloc[0]["model"]
    runner_up = ranking.iloc[1]["model"] if len(ranking) > 1 else None
    margin = (float(ranking.iloc[1]["pinball"] - ranking.iloc[0]["pinball"])
              if len(ranking) > 1 else float("nan"))

    same = full.iloc[0]["model"] == champion
    print(f"\n  CHAMPION: {champion}")
    print(f"  runner-up: {runner_up}, behind by {margin:.3f} pinball")
    print(f"  full-backtest ranking agrees: {same}")
    if not same:
        print(f"    on all folds the leader would be {full.iloc[0]['model']}. "
              "The restriction to post-crisis is doing work here, and was "
              "fixed in section 11 before any result existed.")

    record = {
        "champion": champion,
        "criterion": "mean pinball loss",
        "selection_regime": SELECTION_REGIME,
        "calibration": "conformal global, 12 fold rolling window",
        "runner_up": runner_up,
        "pinball_margin": margin,
        "agrees_with_full_backtest": bool(same),
        "ranking_selection_regime": ranking[show].to_dict("records"),
        "ranking_all_folds": full[show].to_dict("records"),
        "locked_test_opened": False,
    }
    with (cfg.REPORTS / "champion.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    ranking.to_csv(cfg.REPORTS / "selection_ranking.csv", index=False)

    print(f"\n  written: reports/champion.json, reports/selection_ranking.csv")
    print("  The locked test set has not been opened.")
    print("=" * 78)
    return 0


def cmd_locked_test(args) -> int:
    """Section 12. Opens the locked test set."""
    import json

    import pandas as pd

    from .evaluation.locked_test import (
        anatomy, calibrate_test, evaluate_locked_test,
    )

    champ_path = cfg.REPORTS / "champion.json"
    if not champ_path.exists():
        print("  no champion. Run: py -m dayahead.cli select")
        return 1
    with champ_path.open(encoding="utf-8") as f:
        champion = json.load(f)

    print("=" * 78)
    print("SECTION 12: LOCKED TEST EVALUATION")
    print("=" * 78)
    print(f"\n  champion:    {champion['champion']}")
    print(f"  criterion:   {champion['criterion']}, "
          f"{champion['selection_regime']} folds")
    print(f"  calibration: {champion['calibration']}")

    if champion.get("locked_test_opened") and not args.force:
        print("\n  The locked test set has already been opened, and the "
              "result is recorded")
        print("  in reports/locked_test.json. Repeating it and reporting the "
              "better run")
        print("  would defeat the purpose of holding it back. Pass --force "
              "only to")
        print("  reproduce, never to reselect.")
        return 1

    print("\n  opening the locked test set")
    parts = evaluate_locked_test(verbose=True)

    backtest = pd.read_parquet(cfg.PROCESSED / "backtest_predictions.parquet")
    test_preds = pd.concat(
        [parts["main"], parts["leak"], parts["recent"]], ignore_index=True)
    calibrated = calibrate_test(test_preds, backtest)

    a = anatomy(calibrated)

    print("\n  headline, locked test year")
    cols = ["model", "n", "mae", "rmse", "bias", "skill", "pinball",
            "coverage", "interval_width"]
    show = [c for c in cols if c in a["overall"].columns]
    print(a["overall"][show].round(3).to_string(index=False))

    # Expectation 12.1: test MAE should exceed the backtest average.
    from .evaluation.backtest import summarise
    bt_cal = summarise(backtest)
    champ = champion["champion"]
    bt_mae = float(bt_cal.loc[bt_cal["model"] == champ, "mae"].iloc[0])
    te_mae = float(a["overall"].loc[a["overall"]["model"] == champ,
                                    "mae"].iloc[0])
    print(f"\n  expectation 12.1: locked test MAE should exceed backtest MAE")
    print(f"    backtest {bt_mae:.2f}   locked test {te_mae:.2f}   "
          f"{'CONFIRMED' if te_mae > bt_mae else 'REFUTED'}")

    print("\n  check 1, leak quantification")
    leak_mae = float(a["overall"].loc[a["overall"]["model"] == "N-HiTS_LEAKY",
                                      "mae"].iloc[0])
    print(f"    published forecasts {te_mae:.2f}   realised outturn "
          f"{leak_mae:.2f}   improvement {100 * (1 - leak_mae / te_mae):.1f}%")
    print("    This is what the gate closure constraint costs, and what a")
    print("    leaky implementation would report instead.")

    print("\n  check 2, trained on post-2023 only")
    rec_mae = float(a["overall"].loc[a["overall"]["model"] == "N-HiTS_post2023",
                                     "mae"].iloc[0])
    print(f"    full history {te_mae:.2f}   post-2023 only {rec_mae:.2f}")

    print("\n  check 3, month to month dispersion")
    disp = a["by_month"].groupby("model")["mae"].describe()[
        ["mean", "std", "min", "50%", "max"]]
    print(disp.round(2).to_string())

    print("\n  check 5, MAE by delivery hour, champion")
    bh = a["by_hour"]
    bh = bh[bh["model"] == champ].set_index("hour")["mae"]
    print("    best hours:  " + ", ".join(
        f"{h:02d}h {v:.1f}" for h, v in bh.nsmallest(3).items()))
    print("    worst hours: " + ", ".join(
        f"{h:02d}h {v:.1f}" for h, v in bh.nlargest(3).items()))

    print("\n  check 6, conditional subsets")
    print(a["by_condition"].round(3).to_string(index=False))

    cfg.REPORTS.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(cfg.PROCESSED / "locked_test_predictions.parquet")
    for name, frame in a.items():
        frame.to_csv(cfg.REPORTS / f"locked_test_{name}.csv", index=False)

    champion["locked_test_opened"] = True
    champion["locked_test"] = {
        "champion_mae": te_mae,
        "backtest_mae": bt_mae,
        "expectation_12_1": "CONFIRMED" if te_mae > bt_mae else "REFUTED",
        "leak_mae": leak_mae,
        "leak_improvement_pct": 100 * (1 - leak_mae / te_mae),
        "post2023_mae": rec_mae,
        "overall": a["overall"][show].to_dict("records"),
    }
    with champ_path.open("w", encoding="utf-8") as f:
        json.dump(champion, f, indent=2, default=str)

    print(f"\n  written: reports/locked_test_*.csv, reports/champion.json")
    print("  The locked test set is now open. It is not reopened.")
    print("=" * 78)
    return 0


def cmd_value(args) -> int:
    """Section 13. Battery arbitrage valuation, spec fixed in section 25."""
    import pandas as pd

    from .value.battery import (
        CAPACITY_MWH, POWER_MW, ROUND_TRIP_EFFICIENCY, valuation,
    )

    store = cfg.PROCESSED / "locked_test_predictions.parquet"
    if not store.exists():
        print("  no locked test predictions. Run: py -m dayahead.cli locked-test")
        return 1
    preds = pd.read_parquet(store)

    print("=" * 78)
    print("SECTION 13: BATTERY ARBITRAGE VALUATION")
    print("=" * 78)
    print(f"\n  battery:  {POWER_MW:.0f} MW / {CAPACITY_MWH:.0f} MWh, "
          f"{ROUND_TRIP_EFFICIENCY:.0%} round trip, one cycle per day")
    print("  settled:  at realised prices, over the locked test year")
    print("  spec:     fixed in DESIGN.md section 25 before computation")

    result = valuation(preds)
    s = result["summary"]

    print(f"\n  perfect foresight ceiling: "
          f"{result['oracle_total']:,.0f} EUR per MW per year")

    print("\n  strategies")
    cols = ["strategy", "days", "days_traded", "revenue_eur",
            "revenue_per_traded_day", "share_of_perfect", "loss_making_days"]
    show = s[cols].copy()
    show["revenue_eur"] = show["revenue_eur"].round(0)
    show["revenue_per_traded_day"] = show["revenue_per_traded_day"].round(2)
    show["share_of_perfect"] = (100 * show["share_of_perfect"]).round(1)
    show = show.rename(columns={"share_of_perfect": "share_%"})
    print(show.to_string(index=False))

    def get(name, col):
        row = s[s["strategy"] == name]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    champ = get("champion_rule_A", "share_of_perfect")
    base = get("baseline_rule_A", "share_of_perfect")
    print(f"\n  headline: the champion captures {100 * champ:.1f}% of "
          f"perfect-foresight value")
    print(f"            against {100 * base:.1f}% for the daily naive baseline")
    uplift = get("champion_rule_A", "revenue_eur") - get("baseline_rule_A",
                                                         "revenue_eur")
    print(f"            worth {uplift:,.0f} EUR per MW per year")

    # Expectation 25.1: value share should lag the MAE advantage.
    mae_champ, mae_base = 23.45, 28.50
    mae_edge = 1 - mae_champ / mae_base
    value_edge = (champ - base) / base if base else float("nan")
    print(f"\n  expectation 25.1: value share lags the MAE advantage")
    print(f"    MAE advantage over B2   {100 * mae_edge:.1f}%")
    print(f"    value advantage over B2 {100 * value_edge:.1f}%")
    print(f"    {'CONFIRMED' if value_edge < mae_edge else 'REFUTED'}")

    if "champion_rule_B" in set(s["strategy"]):
        a = get("champion_rule_A", "revenue_per_traded_day")
        b = get("champion_rule_B", "revenue_per_traded_day")
        print(f"\n  expectation 25.2: abstaining on wide intervals adds little")
        print(f"    rule A {a:.2f} EUR per traded day, "
              f"rule B {b:.2f}, change {100 * (b / a - 1):+.1f}%")
        print(f"    {'CONFIRMED' if abs(b / a - 1) < 0.10 else 'REFUTED'}")

    cfg.REPORTS.mkdir(parents=True, exist_ok=True)
    s.to_csv(cfg.REPORTS / "battery_valuation.csv", index=False)
    for name, frame in result["daily"].items():
        frame.to_csv(cfg.REPORTS / f"battery_daily_{name}.csv", index=False)
    print(f"\n  written: reports/battery_valuation.csv, battery_daily_*.csv")
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
    p.add_argument("--models",
                   choices=["baselines", "classical", "gbm",
                            "gbm-anchored", "nhits", "all"],
                   default="baselines",
                   help="which models to run; results for models not run are "
                        "reused from the prediction cache")
    p.add_argument("--fresh", action="store_true",
                   help="discard cached predictions instead of reusing them")
    p.add_argument("--every", type=int, default=1,
                   help="run every Nth fold, for a fast smoke test")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("conformal", help="calibrate forecast intervals")
    p.set_defaults(func=cmd_conformal)

    p = sub.add_parser("select", help="apply the section 11 selection rule")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("locked-test",
                       help="section 12: open the held-out year, once")
    p.add_argument("--force", action="store_true",
                   help="reproduce an evaluation already recorded; never "
                        "for reselection")
    p.set_defaults(func=cmd_locked_test)

    p = sub.add_parser("value", help="section 13: battery arbitrage valuation")
    p.set_defaults(func=cmd_value)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
