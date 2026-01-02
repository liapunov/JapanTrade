from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import duckdb
import pandas as pd

TRADE_TABLE = "trade"


@dataclass(frozen=True)
class ExampleQuery:
    name: str
    description: str
    sql: str
    parameters: Tuple[str, ...]


EXAMPLE_QUERIES: List[ExampleQuery] = [
    ExampleQuery(
        name="top_exporters",
        description="Top exporters by value within a date window.",
        sql=f"""
        SELECT country, SUM(value) AS total_value
        FROM {TRADE_TABLE}
        WHERE date BETWEEN ? AND ? AND kind = ?
        GROUP BY country
        ORDER BY total_value DESC
        LIMIT ?;
        """,
        parameters=("start_date", "end_date", "kind", "limit"),
    ),
    ExampleQuery(
        name="fastest_growing_categories",
        description="Fastest-growing categories by YoY change for a target year.",
        sql=f"""
        WITH by_year AS (
            SELECT
                code,
                year,
                SUM(value) AS total_value
            FROM {TRADE_TABLE}
            WHERE kind = ?
            GROUP BY code, year
        ),
        joined AS (
            SELECT
                curr.code,
                curr.year,
                curr.total_value,
                prev.total_value AS prev_value,
                (curr.total_value - prev.total_value)
                / NULLIF(prev.total_value, 0) AS yoy_growth
            FROM by_year curr
            LEFT JOIN by_year prev
                ON curr.code = prev.code AND curr.year = prev.year + 1
        )
        SELECT code, year, total_value, prev_value, yoy_growth
        FROM joined
        WHERE year = ?
        ORDER BY (yoy_growth IS NULL), yoy_growth DESC
        LIMIT ?;
        """,
        parameters=("kind", "year", "limit"),
    ),
]


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["year"] = prepared["date"].dt.year.astype(int)
    return prepared


def export_to_duckdb(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist normalized data to a DuckDB database."""
    target = Path(path)
    prepared = _prepare_dataframe(df)
    con = duckdb.connect(str(target))
    con.register("trade_df", prepared)
    con.execute(f"CREATE TABLE IF NOT EXISTS {TRADE_TABLE} AS SELECT * FROM trade_df")
    con.execute(f"DELETE FROM {TRADE_TABLE}")
    con.execute(f"INSERT INTO {TRADE_TABLE} SELECT * FROM trade_df")
    con.close()
    return target


def export_to_sqlite(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist normalized data to a SQLite database."""
    target = Path(path)
    prepared = _prepare_dataframe(df)
    with sqlite3.connect(target) as conn:
        prepared.to_sql(TRADE_TABLE, conn, if_exists="replace", index=False)
    return target


def run_query_with_duckdb(db_path: str | Path, sql: str, params: Iterable[Any]) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    result = con.execute(sql, params).df()
    con.close()
    return result


def run_query_with_sqlite(db_path: str | Path, sql: str, params: Iterable[Any]) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def run_example_query(
    engine: str, db_path: str | Path, query_name: str, parameters: Dict[str, Any]
) -> pd.DataFrame:
    query = next(q for q in EXAMPLE_QUERIES if q.name == query_name)
    args = [parameters[name] for name in query.parameters]
    if engine == "duckdb":
        return run_query_with_duckdb(db_path, query.sql, args)
    if engine == "sqlite":
        return run_query_with_sqlite(db_path, query.sql, args)
    raise ValueError(f"Unsupported engine {engine}")
