from pathlib import Path

import pandas as pd

from japantrade.analytics import (
    available_filters,
    country_comparison,
    filter_dataframe,
    load_normalized_data,
    month_over_month_trends,
    trailing_12_month_totals,
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
    dates = pd.date_range("2022-01-01", periods=13, freq="MS")
    df = pd.DataFrame(
        {
            "kind": ["HS"] * len(dates),
            "country": ["001"] * len(dates),
            "code": ["0101"] * len(dates),
            "date": dates,
            "unit": ["JPY"] * len(dates),
            "value": [100 + i for i in range(len(dates))],
        }
    )
    trends = year_over_year_trends(df)
    assert (trends["yoy_value"].notna()).any()
    mom = month_over_month_trends(df)
    assert mom["mom_value"].notna().any()
    trailing = trailing_12_month_totals(df)
    assert trailing["trailing_12_value"].notna().any()

    df = load_normalized_data(FIXTURE)
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
