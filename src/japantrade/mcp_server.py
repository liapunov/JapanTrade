"""Local stdio MCP server for prepared JapanTrade datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from .analytics import (
    apply_parameterized_filters,
    country_product_ranking,
    coverage_report as build_coverage_report,
    load_normalized_data,
    product_country_comparison,
    search_hs,
)

mcp = FastMCP("JapanTrade")


def _load_dataset(dataset_path: str) -> tuple[Path, pd.DataFrame]:
    path = Path(dataset_path).expanduser().resolve()
    if path.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError("dataset_path must point to a .parquet or .csv file.")
    if not path.is_file():
        raise ValueError(f"Dataset does not exist: {path}")
    return path, load_normalized_data(path)


def _write_csv(df: pd.DataFrame, output_csv: str | None) -> str | None:
    if output_csv is None:
        return None
    path = Path(output_csv).expanduser().resolve()
    if path.suffix.lower() != ".csv":
        raise ValueError("output_csv must end in .csv.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _table_response(
    df: pd.DataFrame,
    output_csv: str | None = None,
    max_records: int = 100,
    **metadata: Any,
) -> dict[str, Any]:
    if max_records < 1:
        raise ValueError("max_records must be at least 1.")
    csv_path = _write_csv(df, output_csv)
    returned = df.head(max_records)
    response: dict[str, Any] = {
        "row_count": int(len(df)),
        "returned_record_count": int(len(returned)),
        "truncated": len(returned) < len(df),
        "columns": list(df.columns),
        "records": json.loads(returned.to_json(orient="records", date_format="iso")),
        **metadata,
    }
    if csv_path:
        response["csv_path"] = csv_path
    return response


@mcp.tool()
def inspect_dataset(dataset_path: str) -> dict[str, Any]:
    """Inspect a prepared JapanTrade CSV or Parquet dataset before analysis."""
    path, data = _load_dataset(dataset_path)
    return {
        "dataset_path": str(path), "row_count": int(len(data)), "columns": list(data.columns),
        "directions": sorted(data["direction"].astype(str).unique().tolist()),
        "date_start": str(data["date"].min().date()), "date_end": str(data["date"].max().date()),
        "countries": int(data["country"].nunique()), "codes": int(data["code"].nunique()),
    }


@mcp.tool()
def find_hs_codes(query: str, level: int | None = None, limit: int = 20, max_records: int = 100) -> dict[str, Any]:
    """Find HS codes by keyword, exact code, or code prefix."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    return _table_response(search_hs(query, level=level, limit=limit), max_records=max_records)


@mcp.tool()
def rank_country_products(dataset_path: str, country: str, direction: str, hs_level: int = 4,
                          limit: int = 20, date_start: str | None = None,
                          date_end: str | None = None, output_csv: str | None = None,
                          max_records: int = 100) -> dict[str, Any]:
    """Rank HS products for one country; optionally write the result to CSV."""
    if (date_start is None) != (date_end is None):
        raise ValueError("Provide both date_start and date_end, or neither.")
    _, data = _load_dataset(dataset_path)
    period = (date_start, date_end) if date_start else None
    result = country_product_ranking(data, country, direction, period, hs_level, limit)
    return _table_response(result, output_csv, max_records=max_records, analysis="country_product_ranking")


@mcp.tool()
def compare_product_countries(dataset_path: str, countries: list[str], codes: list[str], direction: str,
                              date_start: str | None = None, date_end: str | None = None,
                              output_csv: str | None = None, max_records: int = 100) -> dict[str, Any]:
    """Compare selected HS products across countries; optionally write CSV."""
    if not countries or not codes:
        raise ValueError("Provide at least one country and one HS code or prefix.")
    if (date_start is None) != (date_end is None):
        raise ValueError("Provide both date_start and date_end, or neither.")
    _, data = _load_dataset(dataset_path)
    period = (date_start, date_end) if date_start else None
    result = product_country_comparison(data, countries, codes, direction, period)
    return _table_response(result, output_csv, max_records=max_records, analysis="product_country_comparison")


@mcp.tool()
def dataset_coverage(dataset_path: str, direction: str | None = None, countries: list[str] | None = None,
                     codes: list[str] | None = None, output_csv: str | None = None,
                     max_records: int = 100, expected_start: str | None = None,
                     expected_end: str | None = None) -> dict[str, Any]:
    """Report month gaps; expected_start and expected_end reveal boundary gaps."""
    if (expected_start is None) != (expected_end is None):
        raise ValueError("Provide both expected_start and expected_end, or neither.")
    _, data = _load_dataset(dataset_path)
    filtered = apply_parameterized_filters(data, direction=direction, countries=countries, codes=codes)
    if filtered.empty:
        raise ValueError("No rows match the requested coverage filters.")
    report = build_coverage_report(
        filtered,
        expected_start=expected_start,
        expected_end=expected_end,
    )
    return _table_response(report, output_csv, max_records=max_records, analysis="coverage_report")


def main() -> None:
    """Run the JapanTrade MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
