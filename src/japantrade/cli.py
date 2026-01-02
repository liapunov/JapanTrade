from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .analytics import (
    apply_parameterized_filters,
    load_normalized_data,
    month_over_month_trends,
    trailing_12_month_totals,
    year_over_year_trends,
)


def _parse_date_range(start: Optional[str], end: Optional[str]) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    if not start and not end:
        return None
    start_ts = pd.to_datetime(start) if start else pd.Timestamp.min
    end_ts = pd.to_datetime(end) if end else pd.Timestamp.max
    if start_ts > end_ts:
        raise ValueError("Start date must be before end date.")
    return start_ts, end_ts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter normalized Japan trade data and compute aggregates.")
    parser.add_argument("source", help="Path to a normalized CSV file.")
    parser.add_argument("--kind", help="Restrict to a code kind (HS/PC).")
    parser.add_argument("--countries", nargs="+", help="Exact country codes to include.")
    parser.add_argument("--country-prefixes", nargs="+", help="Country code prefixes (e.g. 00).")
    parser.add_argument("--codes", nargs="+", help="Exact HS/PC codes to include.")
    parser.add_argument("--code-prefixes", nargs="+", help="HS/PC code prefixes to include.")
    parser.add_argument("--units", nargs="+", help="Measurement units to include.")
    parser.add_argument("--date-start", help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--date-end", help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument(
        "--aggregate",
        choices=["yoy", "mom", "trailing12"],
        help="Optional aggregate to compute after filtering.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="If provided, write the filtered rows to this CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of rows to display from the filtered/aggregated result.",
    )
    return parser


def _run_aggregate(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name == "yoy":
        return year_over_year_trends(df)
    if name == "mom":
        return month_over_month_trends(df)
    if name == "trailing12":
        return trailing_12_month_totals(df)
    raise ValueError(f"Unknown aggregate {name}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        date_range = _parse_date_range(args.date_start, args.date_end)
    except ValueError as exc:
        parser.error(str(exc))
    df = load_normalized_data(args.source)
    filtered = apply_parameterized_filters(
        df,
        kind=args.kind,
        countries=args.countries,
        country_prefixes=args.country_prefixes,
        codes=args.codes,
        code_prefixes=args.code_prefixes,
        units=args.units,
        date_range=date_range,
    )

    print(f"Rows after filtering: {len(filtered)}")
    if not filtered.empty:
        min_date = pd.to_datetime(filtered['date']).min().date()
        max_date = pd.to_datetime(filtered['date']).max().date()
        print(f"Date span: {min_date} to {max_date}")
        print(filtered.head(args.limit).to_string(index=False))

    if args.aggregate:
        try:
            aggregated = _run_aggregate(args.aggregate, filtered)
        except ValueError as exc:  # pragma: no cover - exercised in tests via CLI call
            print(f"Aggregation failed: {exc}", file=sys.stderr)
            return 1
        print(f"\nAggregate '{args.aggregate}' (first {args.limit} rows):")
        print(aggregated.head(args.limit).to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(args.output, index=False)
        print(f"Wrote filtered data to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
