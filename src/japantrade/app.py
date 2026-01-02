from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px

from .analytics import (
    FilterOptions,
    available_filters,
    country_comparison,
    country_comparison_chart,
    filter_dataframe,
    load_normalized_data,
    top_products_by_value,
    top_products_chart,
    year_over_year_trends,
    yoy_chart,
)
from .exporters import EXAMPLE_QUERIES, export_to_duckdb, export_to_sqlite, run_example_query

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "normalized_sample.csv"


def _default_query_params(filters: FilterOptions) -> Dict[str, object]:
    return {
        "start_date": filters.min_date.strftime("%Y-%m-%d"),
        "end_date": filters.max_date.strftime("%Y-%m-%d"),
        "kind": filters.kinds[0],
        "limit": 5,
        "year": filters.max_date.year,
    }


def _build_exports(df: pd.DataFrame) -> Dict[str, bytes]:
    buffers: Dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        duckdb_path = Path(tmpdir) / "trade.duckdb"
        sqlite_path = Path(tmpdir) / "trade.sqlite"
        export_to_duckdb(df, duckdb_path)
        export_to_sqlite(df, sqlite_path)
        buffers["DuckDB"] = duckdb_path.read_bytes()
        buffers["SQLite"] = sqlite_path.read_bytes()
    return buffers


def run(data_path: Path | str = DEFAULT_DATA_PATH) -> None:
    import streamlit as st

    st.set_page_config(page_title="Japan Trade Explorer", layout="wide")
    st.title("Japan Trade Explorer")
    st.caption("Load normalized trade data, explore YoY trends, and export SQL-friendly datasets.")

    uploaded = st.file_uploader("Upload normalized data (CSV)", type=["csv"])
    if uploaded:
        df = load_normalized_data(io.BytesIO(uploaded.getvalue()))
    else:
        st.info(f"Using bundled fixture at {data_path}. Upload a CSV to explore your own data.")
        df = load_normalized_data(data_path)

    filters = available_filters(df)

    with st.sidebar:
        st.header("Filters")
        kind = st.selectbox("Kind", options=filters.kinds, index=0 if filters.kinds else None)
        countries = st.multiselect("Countries", options=filters.countries, default=filters.countries)
        codes = st.multiselect("Codes", options=filters.codes, default=filters.codes)
        units = st.multiselect("Units", options=filters.units, default=filters.units)
        date_range = st.date_input(
            "Date range",
            (filters.min_date, filters.max_date),
            min_value=filters.min_date,
            max_value=filters.max_date,
        )

    parsed_range = None
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        parsed_range = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))

    filtered = filter_dataframe(
        df,
        kind=kind,
        countries=countries,
        codes=codes,
        units=units,
        date_range=parsed_range,
    )
    st.metric("Total value", f"{filtered['value'].sum():,.0f}")

    try:
        trends = year_over_year_trends(filtered)
    except ValueError as exc:
        st.warning(str(exc))
        trends = pd.DataFrame()
    if not trends.empty:
        st.altair_chart(yoy_chart(trends.dropna(subset=["yoy_value"])), use_container_width=True)
    else:
        st.warning("No data available for YoY trend.")

    top_df = top_products_by_value(filtered, top_n=10)
    st.subheader("Top products by value")
    st.dataframe(top_df)
    if not top_df.empty:
        st.altair_chart(top_products_chart(top_df), use_container_width=True)
        treemap = px.treemap(top_df, path=["kind", "code"], values="value", title="Value treemap")
        st.plotly_chart(treemap, use_container_width=True)

    st.subheader("Country comparison")
    comparison = country_comparison(filtered, code=codes[0] if codes else None)
    st.dataframe(comparison)
    if not comparison.empty:
        st.altair_chart(country_comparison_chart(comparison), use_container_width=True)

    st.subheader("Exports")
    buffers = _build_exports(filtered)
    for name, content in buffers.items():
        st.download_button(f"Download {name} database", data=content, file_name=f"trade.{name.lower()}")

    st.subheader("Example parameterized queries")
    params = _default_query_params(filters)
    for query in EXAMPLE_QUERIES:
        st.markdown(f"**{query.name}** — {query.description}")
        st.code(query.sql.strip())
        run_query = st.button(f"Run {query.name} (DuckDB)", key=query.name)
        if run_query:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "trade.duckdb"
                export_to_duckdb(filtered, db_path)
                results = run_example_query("duckdb", db_path, query.name, params)
                st.dataframe(results)


if __name__ == "__main__":
    run()
