"""Command-line entry point: python -m skill_or_luck <command>. Run it from the repository folder."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import data, plots
from .experiments import ASSETS, analyse_asset
from .report import own_record, write_report

RESULTS = Path("results")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="skill_or_luck", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    asset_help = f"any of: {', '.join(ASSETS)} (default: all)"

    fetch = commands.add_parser("fetch", help="download and validate price data")
    fetch.add_argument("assets", nargs="*", help=asset_help)
    fetch.add_argument("--latest", action="store_true",
                       help="EUR/USD from FRED (newest data) instead of the pinned copy")

    imp = commands.add_parser("import-csv", help="use a manually downloaded Date/Close CSV")
    imp.add_argument("asset", choices=list(ASSETS))
    imp.add_argument("path")

    run = commands.add_parser("run", help="run all experiments and write results/")
    run.add_argument("assets", nargs="*", help=f"any of: {', '.join(ASSETS)} (default: all with data)")
    run.add_argument("--n-perm", type=int, default=5000, help="random-entry strategies per rule")
    run.add_argument("--n-boot", type=int, default=5000, help="bootstrap resamples")

    own = commands.add_parser("own-record", help="what a personal trading record can show")
    own.add_argument("--wins", type=int, required=True)
    own.add_argument("--trades", type=int, required=True)
    own.add_argument("--payoff", type=float, required=True, help="average win / average loss")

    args = parser.parse_args(argv)
    unknown = [a for a in getattr(args, "assets", []) if a not in ASSETS]
    if unknown:
        parser.error(f"unknown asset(s): {', '.join(unknown)}; choose from {', '.join(ASSETS)}")

    if args.command == "fetch":
        failed = []
        for asset in args.assets or list(ASSETS):
            try:
                notes = data.fetch(asset, latest=args.latest)
            except Exception as exc:  # report and carry on with the other assets
                print(f"{asset}: download failed: {exc}")
                failed.append(asset)
                continue
            print(f"{asset}: saved to {data.RAW_DIR / (asset + '.csv')}")
            for note in notes:
                print(f"  note: {note}")
        if failed:
            raise SystemExit(1)
    elif args.command == "import-csv":
        for note in data.import_csv(args.asset, args.path):
            print(f"  note: {note}")
    elif args.command == "run":
        assets = args.assets or [a for a in ASSETS if (data.RAW_DIR / f"{a}.csv").exists()]
        missing = [a for a in assets if not (data.RAW_DIR / f"{a}.csv").exists()]
        if not assets or missing:
            parser.error(f"no data for {', '.join(missing) or 'any asset'}; "
                         "run `python -m skill_or_luck fetch` first")
        results = []
        for asset in assets:
            started = time.perf_counter()
            result = analyse_asset(asset, args.n_perm, args.n_boot, RESULTS / asset)
            plots.all_figures(result, RESULTS / asset)
            results.append(result)
            print(f"{asset}: done in {time.perf_counter() - started:.0f}s")
        write_report(results, RESULTS / "REPORT.md")
        print(f"report written to {RESULTS / 'REPORT.md'}")
    elif args.command == "own-record":
        if not 0 <= args.wins <= args.trades or args.trades < 1 or args.payoff <= 0:
            parser.error("need 0 <= wins <= trades, trades >= 1 and payoff > 0")
        print(own_record(args.wins, args.trades, args.payoff))


if __name__ == "__main__":
    main()
