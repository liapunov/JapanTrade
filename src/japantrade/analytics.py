from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import altair as alt
import pandas as pd

DEFAULT_DATE_FREQ = "MS"
DEFAULT_GROUP_KEYS = ["kind", "country", "code", "unit"]


@dataclass
class FilterOptions:
    kinds: Sequence[str]
    countries: Sequence[str]
    codes: Sequence[str]
    units: Sequence[str]
    min_date: pd.Timestamp
    max_date: pd.Timestamp


def load_normalized_data(source: str | io.BytesIO | io.StringIO) -> pd.DataFrame:
    """Load normalized trade data from csv-like sources.

    Examples
    --------
    >>> import io
    >>> sample = io.StringIO("kind,country,code,date,unit,value\\nHS,001,0101,2023-01-01,JPY,100")
    >>> load_normalized_data(sample).iloc[0].date  # doctest: +ELLIPSIS
    Timestamp('2023-01-01 ...')
    """
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
        min_date=pd.to_datetime(df["date"]).min(),
        max_date=pd.to_datetime(df["date"]).max(),
    )


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    return normalized


def _format_group_label(group_keys: Sequence[str], values: tuple[str, ...] | str) -> str:
    if not isinstance(values, tuple):
        values = (values,)
    return ", ".join(f"{key}={value}" for key, value in zip(group_keys, values))


def _ensure_month_window(
    monthly_df: pd.DataFrame,
    group_keys: Sequence[str],
    months_needed: int,
    label: str,
) -> None:
    """Validate that each group has consecutive monthly coverage."""
    if monthly_df.empty:
        raise ValueError(f"Cannot compute {label}: no data provided.")
    monthly_df = monthly_df.copy()
    monthly_df["period"] = pd.to_datetime(monthly_df["date"]).dt.to_period("M")

    for group_values, group in monthly_df.groupby(group_keys):
        unique_periods = set(group["period"].unique())
        if len(unique_periods) < months_needed:
            raise ValueError(
                f"Cannot compute {label} for {_format_group_label(group_keys, group_values)}: "
                f"need at least {months_needed} consecutive months, found {len(unique_periods)}."
            )
        end_period = max(unique_periods)
        start_period = end_period - (months_needed - 1)
        expected = pd.period_range(start=start_period, end=end_period, freq="M")
        missing = [period for period in expected if period not in unique_periods]
        if missing:
            missing_str = ", ".join(period.strftime("%Y-%m") for period in missing)
            raise ValueError(
                f"Cannot compute {label} for {_format_group_label(group_keys, group_values)}: "
                f"missing months {missing_str}."
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
    return apply_parameterized_filters(
        df,
        kind=kind,
        countries=countries,
        codes=codes,
        units=units,
        date_range=date_range,
    )


def apply_parameterized_filters(
    df: pd.DataFrame,
    kind: Optional[str] = None,
    countries: Optional[Iterable[str]] = None,
    country_prefixes: Optional[Iterable[str]] = None,
    codes: Optional[Iterable[str]] = None,
    code_prefixes: Optional[Iterable[str]] = None,
    units: Optional[Iterable[str]] = None,
    date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Apply rich, parameterized filters.

    Parameters
    ----------
    kind : str, optional
        Restrict to a specific code kind (e.g. ``"HS"`` or ``"PC"``).
    countries : iterable of str, optional
        Explicit list of country codes to keep.
    country_prefixes : iterable of str, optional
        Prefixes that should match country codes (e.g. ``["00"]``).
    codes : iterable of str, optional
        Explicit list of HS/PC codes to keep.
    code_prefixes : iterable of str, optional
        Prefix filters for HS/PC codes (e.g. ``["010", "0203"]``).
    units : iterable of str, optional
        Acceptable measurement units.
    date_range : tuple of Timestamp, optional
        Inclusive date range filter.

    Examples
    --------
    >>> data = pd.DataFrame({\n\
    ...     "kind": ["HS"] * 2,\n\
    ...     "country": ["001", "009"],\n\
    ...     "code": ["0101", "0301"],\n\
    ...     "date": ["2023-01-01", "2023-02-01"],\n\
    ...     "unit": ["JPY", "JPY"],\n\
    ...     "value": [1, 2],\n\
    ... })\n\
    >>> apply_parameterized_filters(data, country_prefixes=["00"])["country"].unique()\n\
    array(['001'], dtype=object)\n\
    """
    filtered = _normalize_dates(df)
    if kind:
        filtered = filtered[filtered["kind"] == kind]

    if countries or country_prefixes:
        allowed_countries = set(countries or [])
        if country_prefixes:
            prefixes = tuple(country_prefixes)
            allowed_countries.update(filtered.loc[filtered["country"].str.startswith(prefixes), "country"])
        filtered = filtered[filtered["country"].isin(allowed_countries)]

    if codes or code_prefixes:
        allowed_codes = set(codes or [])
        if code_prefixes:
            prefixes = tuple(code_prefixes)
            allowed_codes.update(filtered.loc[filtered["code"].str.startswith(prefixes), "code"])
        filtered = filtered[filtered["code"].isin(allowed_codes)]

    if units:
        filtered = filtered[filtered["unit"].isin(set(units))]
    if date_range:
        start, end = date_range
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
    return filtered


def _resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_dates(df)
    return (
        normalized.groupby(DEFAULT_GROUP_KEYS + [pd.Grouper(key="date", freq=DEFAULT_DATE_FREQ)])["value"]
        .sum()
        .reset_index()
        .sort_values("date")
    )


def year_over_year_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Return YoY growth by country/code/unit.

    Raises
    ------
    ValueError
        If there are fewer than 13 consecutive months per group.
    """
    monthly = _resample_monthly(df)
    _ensure_month_window(monthly, DEFAULT_GROUP_KEYS, 13, "year-over-year change")
    monthly["yoy_value"] = monthly.groupby(DEFAULT_GROUP_KEYS)["value"].pct_change(periods=12)
    if not monthly["yoy_value"].notna().any():
        raise ValueError("Not enough data to compute year-over-year change for any group.")
    return monthly


def month_over_month_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Return MoM growth by country/code/unit with validation."""
    monthly = _resample_monthly(df)
    _ensure_month_window(monthly, DEFAULT_GROUP_KEYS, 2, "month-over-month change")
    monthly["mom_value"] = monthly.groupby(DEFAULT_GROUP_KEYS)["value"].pct_change(periods=1)
    if not monthly["mom_value"].notna().any():
        raise ValueError("Not enough data to compute month-over-month change for any group.")
    return monthly


def trailing_12_month_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Return trailing 12-month totals by group.

    Examples
    --------
    >>> data = pd.DataFrame({\n\
    ...     "kind": ["HS"] * 12,\n\
    ...     "country": ["001"] * 12,\n\
    ...     "code": ["0101"] * 12,\n\
    ...     "date": pd.date_range(\"2022-01-01\", periods=12, freq=\"MS\"),\n\
    ...     "unit": ["JPY"] * 12,\n\
    ...     "value": [1] * 12,\n\
    ... })\n\
    >>> trailing_12_month_totals(data).iloc[-1].trailing_12_value\n\
    12.0\n\
    """
    monthly = _resample_monthly(df)
    _ensure_month_window(monthly, DEFAULT_GROUP_KEYS, 12, "trailing 12-month totals")
    monthly["trailing_12_value"] = (
        monthly.sort_values("date")
        .groupby(DEFAULT_GROUP_KEYS)["value"]
        .transform(lambda series: series.rolling(window=12, min_periods=12).sum())
    )
    if monthly["trailing_12_value"].isna().all():
        raise ValueError("Not enough data to compute trailing 12-month totals for any group.")
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
