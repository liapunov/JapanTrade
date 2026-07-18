"""Command line interface for the reproducible JapanTrade workflow."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from .analytics import country_product_ranking, load_normalized_data, product_country_comparison, search_hs
from .customsgrabber import CustomsGrabber
from .tradefile import TradeFile


def _years(value: str) -> tuple[int, int]:
    try:
        start, end = (int(part) for part in value.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("years must be START:END, for example 2024:2025") from exc
    return start, end


def _period(args) -> Optional[tuple[str, str]]:
    if not args.date_start and not args.date_end:
        return None
    if not (args.date_start and args.date_end):
        raise ValueError("Specify both --date-start and --date-end.")
    return args.date_start, args.date_end


def _write_or_print(data, output: Optional[Path]) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)
        print(f"Wrote {len(data)} rows to {output}")
    else:
        print(data.to_string(index=False))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="japantrade", description="Japanese Customs HS trade data tools")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download official HS archives")
    download.add_argument("--direction", required=True, choices=("import", "export"))
    download.add_argument("--years", required=True, type=_years)
    download.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Normalize one downloaded CSV/ZIP into a dataset")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--direction", required=True, choices=("import", "export"))
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--no-descriptions", action="store_true")
    hs = commands.add_parser("hs", help="Discover HS codes")
    hs_sub = hs.add_subparsers(dest="hs_command", required=True)
    hs_search = hs_sub.add_parser("search", help="Search HS descriptions")
    hs_search.add_argument("query")
    hs_search.add_argument("--level", type=int)
    hs_search.add_argument("--limit", type=int, default=20)
    rank = commands.add_parser("country-rank", help="Rank HS categories for one country")
    rank.add_argument("source", type=Path)
    rank.add_argument("--country", required=True)
    rank.add_argument("--direction", required=True, choices=("import", "export"))
    rank.add_argument("--hs-level", type=int, default=4)
    rank.add_argument("--limit", type=int, default=20)
    rank.add_argument("--date-start")
    rank.add_argument("--date-end")
    rank.add_argument("--output", type=Path)
    compare = commands.add_parser("compare", help="Compare selected HS products across countries")
    compare.add_argument("source", type=Path)
    compare.add_argument("--countries", nargs="+", required=True)
    compare.add_argument("--codes", nargs="+", required=True)
    compare.add_argument("--direction", required=True, choices=("import", "export"))
    compare.add_argument("--date-start")
    compare.add_argument("--date-end")
    compare.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "download":
        start, end = args.years
        args.output.mkdir(parents=True, exist_ok=True)
        paths = CustomsGrabber().grabRange(start, end, direction=args.direction, kind="HS", save_folder=str(args.output), allow_large_download=True)
        (args.output / "manifest.json").write_text(json.dumps({"direction": args.direction, "years": [start, end], "files": paths}, indent=2), encoding="utf-8")
        return 0
    if args.command == "prepare":
        from .tradefile import NormalizationConfig
        trade = TradeFile(args.source, direction=args.direction, kind="HS", normalization_config=NormalizationConfig(include_descriptions=not args.no_descriptions))
        target = trade.save_to_file(args.output, fmt="parquet")
        print(f"Prepared {len(trade.data)} rows at {target}")
        return 0
    if args.command == "hs":
        _write_or_print(search_hs(args.query, args.level, args.limit), None)
        return 0
    data = load_normalized_data(args.source)
    period = _period(args)
    if args.command == "country-rank":
        _write_or_print(country_product_ranking(data, args.country, args.direction, period, args.hs_level, args.limit), args.output)
    elif args.command == "compare":
        _write_or_print(product_country_comparison(data, args.countries, args.codes, args.direction, period), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
