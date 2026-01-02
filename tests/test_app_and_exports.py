from pathlib import Path

import pandas as pd

from japantrade.analytics import (
    available_filters,
    country_comparison,
    filter_dataframe,
    load_normalized_data,
    top_products_by_value,
    year_over_year_trends,
)
from japantrade.exporters import EXAMPLE_QUERIES, export_to_duckdb, export_to_sqlite, run_example_query


FIXTURE = Path(__file__).parent / "fixtures" / "normalized_sample.csv"


def test_load_and_filter_fixture():
    df = load_normalized_data(FIXTURE)
    filters = available_filters(df)
    filtered = filter_dataframe(df, kind="HS", countries=["001", "002"])
    assert set(filtered["kind"]) == {"HS"}
    assert set(filtered["country"]).issubset({"001", "002"})
    assert filters.min_date.year == 2022


def test_yoy_trends_and_top_products():
    df = load_normalized_data(FIXTURE)
    trends = year_over_year_trends(df[df["kind"] == "HS"])
    # At least one YoY growth value should be finite thanks to the fixture.
    assert (trends["yoy_value"].notna()).any()
    top_codes = top_products_by_value(df, top_n=3)
    assert not top_codes.empty
    assert set(top_codes.columns) == {"kind", "code", "value"}


def test_country_comparison_and_exports(tmp_path):
    df = load_normalized_data(FIXTURE)
    comparison = country_comparison(df, code="0101")
    assert not comparison.empty

    duckdb_path = tmp_path / "trade.duckdb"
    sqlite_path = tmp_path / "trade.sqlite"
    export_to_duckdb(df, duckdb_path)
    export_to_sqlite(df, sqlite_path)
    assert duckdb_path.exists() and sqlite_path.exists()

    params = {
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "kind": "HS",
        "limit": 3,
        "year": df["date"].max().year,
    }
    for engine, path in (("duckdb", duckdb_path), ("sqlite", sqlite_path)):
        for query in EXAMPLE_QUERIES:
            result = run_example_query(engine, path, query.name, params)
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
