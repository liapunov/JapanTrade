"""Analysis helpers for prepared Japanese Customs HS trade datasets."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from .schema import (
    deduplicate_normalized_data,
    validate_direction_values,
    validate_non_null_values,
    validate_required_columns,
)

VALUE_UNIT = "JPY"


@dataclass
class FilterOptions:
    kinds: Sequence[str]
    countries: Sequence[str]
    codes: Sequence[str]
    units: Sequence[str]
    min_date: pd.Timestamp
    max_date: pd.Timestamp


def load_normalized_data(source: str | Path | io.BytesIO | io.StringIO) -> pd.DataFrame:
    """Load a prepared CSV or Parquet dataset and validate its v1 schema."""
    path = str(source) if isinstance(source, (str, Path)) else None
    df = pd.read_parquet(source) if path and path.endswith(".parquet") else pd.read_csv(source, dtype=str)
    validate_required_columns(df)
    result = df.copy()
    validate_non_null_values(result)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["value"] = pd.to_numeric(result["value"], errors="raise")
    result["country"] = result["country"].astype(str).str.zfill(3)
    result["code"] = result["code"].astype(str).str.replace(r"\s+", "", regex=True)
    validate_direction_values(result["direction"].unique())
    return deduplicate_normalized_data(result)


def available_filters(df: pd.DataFrame):
    """Compatibility helper describing selectable fields."""
    return FilterOptions(sorted(df.kind.unique()), sorted(df.country.unique()), sorted(df.code.unique()),
                         sorted(df.unit.unique()), pd.to_datetime(df.date).min(), pd.to_datetime(df.date).max())


def apply_parameterized_filters(df: pd.DataFrame, kind: Optional[str] = None,
                                countries: Optional[Iterable[str]] = None,
                                country_prefixes: Optional[Iterable[str]] = None,
                                codes: Optional[Iterable[str]] = None,
                                code_prefixes: Optional[Iterable[str]] = None,
                                units: Optional[Iterable[str]] = None,
                                date_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None,
                                direction: Optional[str] = None) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    if kind: result = result[result.kind == kind]
    if direction: result = result[result.direction == direction]
    if countries or country_prefixes:
        allowed = set(str(x).zfill(3) for x in (countries or []))
        if country_prefixes:
            allowed.update(result.loc[result.country.astype(str).str.startswith(tuple(country_prefixes)), "country"])
        result = result[result.country.isin(allowed)]
    if codes or code_prefixes:
        allowed = set(str(x) for x in (codes or []))
        if code_prefixes:
            allowed.update(result.loc[result.code.astype(str).str.startswith(tuple(code_prefixes)), "code"])
        result = result[result.code.isin(allowed)]
    if units: result = result[result.unit.isin(set(units))]
    if date_range: result = result[result.date.between(*date_range)]
    return result


def filter_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return apply_parameterized_filters(df, **kwargs)


def _value_data(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    if direction not in {"import", "export"}:
        raise ValueError("direction must be 'import' or 'export'.")
    validate_required_columns(df)
    validate_non_null_values(df)
    validate_direction_values(df["direction"].unique())
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="raise")
    normalized["country"] = normalized["country"].astype(str).str.zfill(3)
    normalized["code"] = normalized["code"].astype(str).str.replace(r"\s+", "", regex=True)
    normalized = deduplicate_normalized_data(normalized)
    result = normalized[(normalized["direction"] == direction) & (normalized["kind"] == "HS") & (normalized["unit"] == VALUE_UNIT)].copy()
    result["date"] = pd.to_datetime(result["date"])
    if result.empty:
        raise ValueError("No HS JPY-value rows match the requested direction.")
    return result


def _period_bounds(df: pd.DataFrame, period: Optional[tuple[object, object]] = None) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    if period:
        start, end = (pd.Timestamp(value).to_period("M").to_timestamp() for value in period)
        months = (end.year - start.year) * 12 + end.month - start.month + 1
        if months < 1: raise ValueError("period end must be on or after period start.")
    else:
        latest = pd.to_datetime(df.date).max().to_period("M").to_timestamp()
        start, end, months = latest - pd.DateOffset(months=11), latest, 12
    previous_end = start - pd.DateOffset(months=1)
    previous_start = previous_end - pd.DateOffset(months=months - 1)
    return previous_start, previous_end, start, end


def _missing_months(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    expected = pd.period_range(start, end, freq="M")
    observed = set(pd.to_datetime(data["date"]).dt.to_period("M"))
    return [str(month) for month in expected if month not in observed]


def _validate_comparison_coverage(
    data: pd.DataFrame,
    bounds: tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp],
    countries: Sequence[str],
) -> None:
    previous_start, previous_end, start, end = bounds
    failures = []
    for country in countries:
        country_data = data[data["country"] == country]
        for label, window_start, window_end in (
            ("previous", previous_start, previous_end),
            ("current", start, end),
        ):
            missing = _missing_months(country_data, window_start, window_end)
            if missing:
                failures.append(f"{country} {label} window missing {', '.join(missing)}")
    if failures:
        raise ValueError("Incomplete comparison window: " + "; ".join(failures))


def _normalize_prefixes(codes: Iterable[str]) -> tuple[str, ...]:
    cleaned = []
    for code in codes:
        prefix = str(code).replace(" ", "")
        if not prefix or not prefix.isdigit():
            raise ValueError("HS code prefixes must be non-empty numeric strings.")
        if prefix not in cleaned:
            cleaned.append(prefix)
    if not cleaned:
        raise ValueError("Provide at least one HS code or prefix.")
    return tuple(prefix for prefix in cleaned if not any(prefix.startswith(other) for other in cleaned if prefix != other))


def coverage_report(
    df: pd.DataFrame,
    group_by: Sequence[str] = ("direction", "country", "code"),
    *,
    expected_start: object | None = None,
    expected_end: object | None = None,
) -> pd.DataFrame:
    """Report observed months and internal/boundary gaps for every series.

    When explicit expected bounds are omitted, each series is assessed from
    its first observed month through its last observed month.
    """
    if (expected_start is None) != (expected_end is None):
        raise ValueError("Provide both expected_start and expected_end, or neither.")
    requested_start = pd.Timestamp(expected_start).to_period("M") if expected_start is not None else None
    requested_end = pd.Timestamp(expected_end).to_period("M") if expected_end is not None else None
    if requested_start is not None and requested_end < requested_start:
        raise ValueError("expected_end must be on or after expected_start.")
    data = df.copy(); data["date"] = pd.to_datetime(data.date).dt.to_period("M")
    rows = []
    for keys, group in data.groupby(list(group_by)):
        periods = sorted(group.date.unique())
        start = requested_start if requested_start is not None else periods[0]
        end = requested_end if requested_end is not None else periods[-1]
        expected = pd.period_range(start, end, freq="M")
        observed = [period for period in periods if start <= period <= end]
        observed_set = set(observed)
        missing = [period for period in expected if period not in observed_set]
        leading = [period for period in missing if period < periods[0]]
        trailing = [period for period in missing if period > periods[-1]]
        internal = [period for period in missing if periods[0] <= period <= periods[-1]]
        rows.append(dict(zip(group_by, keys if isinstance(keys, tuple) else (keys,)),
                         first_month=str(periods[0]), last_month=str(periods[-1]),
                         expected_start=str(start), expected_end=str(end),
                         expected_months=len(expected), observed_months=len(observed),
                         missing_months=len(missing), internal_missing_months=len(internal),
                         missing_leading_months=len(leading), missing_trailing_months=len(trailing),
                         missing_periods=[str(period) for period in missing]))
    return pd.DataFrame(rows)


def _resolve_country(df: pd.DataFrame, country: str) -> str:
    candidate = str(country).strip()
    if candidate.isdigit():
        candidate = candidate.zfill(3)
        if candidate in set(df.country.astype(str)): return candidate
    if "country_name" in df:
        matches = df.loc[df.country_name.astype(str).str.casefold() == candidate.casefold(), "country"].unique()
        if len(matches) == 1: return str(matches[0])
    raise ValueError(f"Unknown or ambiguous country: {country!r}. Use a Japan Customs code or an exact country name.")


def country_product_ranking(df: pd.DataFrame, country: str, direction: str,
                            period: Optional[tuple[object, object]] = None,
                            hs_level: int = 4, limit: int = 20) -> pd.DataFrame:
    """Rank HS prefixes for one trading partner by current-period JPY value."""
    if hs_level not in {2, 4, 6}: raise ValueError("hs_level must be one of 2, 4, or 6.")
    data = _value_data(df, direction); country_code = _resolve_country(data, country)
    data = data[data.country == country_code].copy()
    bounds = _period_bounds(data, period)
    previous_start, previous_end, start, end = bounds
    _validate_comparison_coverage(data, bounds, [country_code])
    current = data[data.date.between(start, end)]; previous = data[data.date.between(previous_start, previous_end)]
    for frame in (current, previous): frame["hs_code"] = frame.code.str[:hs_level]
    current_totals = current.groupby("hs_code").value.sum().rename("current_value")
    previous_totals = previous.groupby("hs_code").value.sum().rename("prior_value")
    result = pd.concat([current_totals, previous_totals], axis=1).fillna(0).reset_index()
    result["absolute_change"] = result.current_value - result.prior_value
    result["growth"] = result.absolute_change.div(result.prior_value.where(result.prior_value != 0))
    result["share"] = result.current_value / result.current_value.sum()
    result["rank"] = result.current_value.rank(method="first", ascending=False).astype(int)
    description_lookup = enrich_hs_descriptions(
        pd.DataFrame({"code": result["hs_code"]})
    ).rename(columns={"code": "hs_code"})
    result = result.merge(description_lookup[["hs_code", "code_description"]], on="hs_code", how="left")
    result["country"] = country_code; result["direction"] = direction; result["period_start"] = start; result["period_end"] = end
    return result.sort_values("rank").head(limit).reset_index(drop=True)


def product_country_comparison(df: pd.DataFrame, countries: Iterable[str], codes: Iterable[str], direction: str,
                               period: Optional[tuple[object, object]] = None) -> pd.DataFrame:
    """Compare HS prefixes using JPY, treating absent selected rows as zero.

    That zero interpretation assumes the input represents a complete HS
    universe; partial extracts cannot distinguish omission from no trade.
    """
    data = _value_data(df, direction)
    country_codes = list(dict.fromkeys(_resolve_country(data, country) for country in countries))
    prefixes = _normalize_prefixes(codes)
    country_data = data[data.country.isin(country_codes)].copy()
    bounds = _period_bounds(country_data, period)
    previous_start, previous_end, start, end = bounds
    _validate_comparison_coverage(country_data, bounds, country_codes)
    selected = country_data[country_data.code.str.startswith(prefixes)].copy()
    current = selected[selected.date.between(start, end)].groupby("country").value.sum().reindex(country_codes, fill_value=0).rename("current_value")
    previous = selected[selected.date.between(previous_start, previous_end)].groupby("country").value.sum().reindex(country_codes, fill_value=0).rename("prior_value")
    result = pd.concat([current, previous], axis=1).reset_index()
    result["absolute_change"] = result.current_value - result.prior_value
    result["growth"] = result.absolute_change.div(result.prior_value.where(result.prior_value != 0))
    result["selected_market_share"] = result.current_value / result.current_value.sum()
    result["direction"] = direction; result["period_start"] = start; result["period_end"] = end
    if "country_name" in country_data: result = result.merge(country_data[["country", "country_name"]].drop_duplicates(), on="country", how="left")
    return result.sort_values("current_value", ascending=False).reset_index(drop=True)


def trade_timeseries(df: pd.DataFrame, countries: Iterable[str], codes: Iterable[str], direction: str) -> pd.DataFrame:
    data = _value_data(df, direction)
    country_codes = [_resolve_country(data, country) for country in countries]
    prefixes = tuple(str(code) for code in codes)
    return (data[data.country.isin(country_codes) & data.code.str.startswith(prefixes)]
            .groupby(["country", "date"], as_index=False).value.sum().sort_values(["country", "date"]))


def search_hs(query: str, level: Optional[int] = None, limit: int = 20) -> pd.DataFrame:
    """Search packaged HS descriptions by keyword, exact code, or code prefix."""
    path = Path(__file__).with_name("HScodes.csv")
    lookup = pd.read_csv(path, sep=";", dtype=str)
    code_col = "Code.1" if "Code.1" in lookup else "Code"
    lookup = lookup.rename(columns={code_col: "code", "Description": "description", "Level": "level"})
    lookup["code"] = lookup.code.fillna("").str.replace(r"\s+", "", regex=True)
    lookup["level"] = pd.to_numeric(lookup.level, errors="coerce")
    term = str(query).strip()
    matched = lookup.code.str.startswith(term) if term.isdigit() else lookup.description.fillna("").str.contains(term, case=False, regex=False)
    result = lookup.loc[matched, ["code", "level", "description"]].dropna(subset=["code"])
    if level is not None: result = result[result.level == level]
    return result.drop_duplicates("code").head(limit).reset_index(drop=True)


def enrich_hs_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the most specific available HS description to each HS code.

    Japan Customs rows use 9-digit tariff codes while the packaged HS lookup
    primarily contains 2-, 4-, and 6-digit international HS levels.  A row is
    therefore matched first by exact code and then by its longest lookup prefix.
    """
    result = df.copy()
    if "code" not in result:
        raise ValueError("Cannot enrich descriptions: missing 'code' column.")
    lookup_path = Path(__file__).with_name("HScodes.csv")
    lookup = pd.read_csv(lookup_path, sep=";", dtype=str)
    code_column = "Code.1" if "Code.1" in lookup else "Code"
    lookup = lookup.rename(columns={code_column: "code", "Description": "code_description"})
    lookup["code"] = lookup.code.fillna("").str.replace(r"\s+", "", regex=True)
    lookup = lookup[(lookup.code != "") & lookup.code_description.notna()].drop_duplicates("code")

    codes = result.code.astype(str).str.replace(r"\s+", "", regex=True)
    descriptions = result["code_description"].copy() if "code_description" in result else pd.Series(pd.NA, index=result.index, dtype="object")
    lengths = sorted(lookup.code.str.len().unique(), reverse=True)
    for length in lengths:
        missing = descriptions.isna()
        if not missing.any():
            break
        mapping = lookup.loc[lookup.code.str.len() == length].set_index("code")["code_description"]
        descriptions.loc[missing] = codes.loc[missing].str[:length].map(mapping)
    result["code_description"] = descriptions
    return result


# Lightweight backwards-compatible helpers.
def top_products_by_value(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return df.groupby(["kind", "code"], as_index=False).value.sum().sort_values("value", ascending=False).head(top_n)

def country_comparison(df: pd.DataFrame, code: Optional[str] = None) -> pd.DataFrame:
    data = df[df.code == code] if code else df
    return data.groupby(["country", "kind"], as_index=False).value.sum().sort_values("value", ascending=False)


def _monthly(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy(); data["date"] = pd.to_datetime(data.date)
    keys = [key for key in ("direction", "kind", "country", "code", "unit") if key in data]
    return data.groupby(keys + [pd.Grouper(key="date", freq="MS")], as_index=False).value.sum().sort_values("date")


def _check_consecutive(monthly: pd.DataFrame, keys: Sequence[str], needed: int, label: str) -> None:
    for values, group in monthly.groupby(list(keys)):
        periods = set(group.date.dt.to_period("M")); end = max(periods); expected = pd.period_range(end - (needed - 1), end, freq="M")
        if any(period not in periods for period in expected):
            raise ValueError(f"Cannot compute {label}: need {needed} consecutive months for {values}.")


def year_over_year_trends(df: pd.DataFrame) -> pd.DataFrame:
    monthly = _monthly(df); keys = [key for key in ("direction", "kind", "country", "code", "unit") if key in monthly]
    _check_consecutive(monthly, keys, 13, "year-over-year change")
    monthly["yoy_value"] = monthly.groupby(keys).value.pct_change(12)
    return monthly


def month_over_month_trends(df: pd.DataFrame) -> pd.DataFrame:
    monthly = _monthly(df); keys = [key for key in ("direction", "kind", "country", "code", "unit") if key in monthly]
    _check_consecutive(monthly, keys, 2, "month-over-month change")
    monthly["mom_value"] = monthly.groupby(keys).value.pct_change()
    return monthly


def trailing_12_month_totals(df: pd.DataFrame) -> pd.DataFrame:
    monthly = _monthly(df); keys = [key for key in ("direction", "kind", "country", "code", "unit") if key in monthly]
    _check_consecutive(monthly, keys, 12, "trailing 12-month totals")
    monthly["trailing_12_value"] = monthly.groupby(keys).value.transform(lambda values: values.rolling(12, min_periods=12).sum())
    return monthly


def yoy_chart(trends):
    import altair as alt
    return alt.Chart(trends).mark_line(point=True).encode(x="date:T", y="yoy_value:Q", color="code:N")


def top_products_chart(data):
    import altair as alt
    return alt.Chart(data).mark_bar().encode(x="value:Q", y=alt.Y("code:N", sort="-x"))


def country_comparison_chart(data):
    import altair as alt
    return alt.Chart(data).mark_bar().encode(x="country:N", y="value:Q")
