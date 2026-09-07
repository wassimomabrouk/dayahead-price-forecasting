"""
Command line interface.

    py -m dayahead.cli ingest [--dry-run]
    py -m dayahead.cli validate
    py -m dayahead.cli report

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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
