from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import altair as alt
import pandas as pd

DEFAULT_DATE_FREQ = "MS"


@dataclass
class FilterOptions:
    kinds: Sequence[str]
    countries: Sequence[str]
    codes: Sequence[str]
    units: Sequence[str]
    min_date: pd.Timestamp
    max_date: pd.Timestamp


def load_normalized_data(source: str | io.BytesIO | io.StringIO) -> pd.DataFrame:
    """Load normalized trade data from csv-like sources."""
    df = pd.read_csv(
        source,
        dtype={
            "kind": str,
            "country": str,
            "code": str,
            "unit": str,
            "date": str,
            "value": "float64",
        },
    )
    expected_cols = {"kind", "country", "code", "date", "unit", "value"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])
    return df


def available_filters(df: pd.DataFrame) -> FilterOptions:
    return FilterOptions(
        kinds=sorted(df["kind"].unique()),
        countries=sorted(df["country"].unique()),
        codes=sorted(df["code"].unique()),
        units=sorted(df["unit"].unique()),
        min_date=df["date"].min(),
        max_date=df["date"].max(),
    )


def filter_dataframe(
    df: pd.DataFrame,
    kind: Optional[str] = None,
    countries: Optional[Iterable[str]] = None,
    codes: Optional[Iterable[str]] = None,
    units: Optional[Iterable[str]] = None,
    date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Apply simple filters to the normalized dataframe."""
    filtered = df.copy()
    if kind:
        filtered = filtered[filtered["kind"] == kind]
    if countries:
        filtered = filtered[filtered["country"].isin(set(countries))]
    if codes:
        filtered = filtered[filtered["code"].isin(set(codes))]
    if units:
        filtered = filtered[filtered["unit"].isin(set(units))]
    if date_range:
        start, end = date_range
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
    return filtered


def _resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.set_index("date")
        .groupby(["kind", "country", "code", "unit"])
        .resample(DEFAULT_DATE_FREQ)["value"]
        .sum()
        .fillna(0)
        .reset_index()
    )


def year_over_year_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Return YoY growth by country/code/unit."""
    monthly = _resample_monthly(df)
    monthly["yoy_value"] = (
        monthly.sort_values("date")
        .groupby(["kind", "country", "code", "unit"])["value"]
        .pct_change(periods=12)
    )
    return monthly


def top_products_by_value(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    grouped = (
        df.groupby(["kind", "code"])["value"]
        .sum()
        .reset_index()
        .sort_values("value", ascending=False)
    )
    return grouped.head(top_n)


def country_comparison(df: pd.DataFrame, code: Optional[str] = None) -> pd.DataFrame:
    scoped = df[df["code"] == code] if code else df
    return (
        scoped.groupby(["country", "kind"])["value"]
        .sum()
        .reset_index()
        .sort_values("value", ascending=False)
    )


def yoy_chart(trends: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(trends)
        .mark_line(point=True)
        .encode(
            x="date:T",
            y=alt.Y("yoy_value:Q", title="YoY growth"),
            color="code:N",
            tooltip=["country", "code", "date", "yoy_value"],
        )
        .properties(title="Year-over-year trend")
    )


def top_products_chart(top_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(top_df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="Value"),
            y=alt.Y("code:N", sort="-x", title="Product code"),
            color="kind:N",
            tooltip=["code", "kind", "value"],
        )
        .properties(title="Top products by value")
    )


def country_comparison_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("country:N", title="Country"),
            y=alt.Y("value:Q", title="Value"),
            color="kind:N",
            tooltip=["country", "kind", "value"],
        )
        .properties(title="Country comparison")
    )
