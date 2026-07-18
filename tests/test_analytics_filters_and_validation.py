import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from japantrade.analytics import (
    apply_parameterized_filters,
    coverage_report,
    trailing_12_month_totals,
    year_over_year_trends,
)


def _build_df(dates, unit="JPY"):
    return pd.DataFrame(
        {
            "kind": ["HS"] * len(dates),
            "country": ["001"] * len(dates),
            "code": ["0101"] * len(dates),
            "date": dates,
            "unit": [unit] * len(dates),
            "value": range(1, len(dates) + 1),
        }
    )


def test_apply_filters_supports_prefixes():
    dates = pd.date_range("2023-01-01", periods=3, freq="MS")
    df = pd.DataFrame(
        {
            "kind": ["HS"] * 3,
            "country": ["001", "009", "101"],
            "code": ["0101", "0102", "9900"],
            "date": dates,
            "unit": ["JPY"] * 3,
            "value": [1, 2, 3],
        }
    )
    filtered = apply_parameterized_filters(df, country_prefixes=["00"], code_prefixes=["010"])
    assert set(filtered["country"]) == {"001", "009"}
    assert set(filtered["code"]) == {"0101", "0102"}


def test_yoy_validation_detects_missing_months():
    dates = pd.date_range("2022-01-01", periods=13, freq="MS").delete(5)
    df = _build_df(dates)
    with pytest.raises(ValueError, match="year-over-year change"):
        year_over_year_trends(df)


def test_trailing_totals_guard_against_sparse_units():
    complete = _build_df(pd.date_range("2022-01-01", periods=12, freq="MS"), unit="JPY")
    sparse = _build_df(pd.date_range("2022-01-01", periods=6, freq="MS"), unit="KG")
    df = pd.concat([complete, sparse], ignore_index=True)
    with pytest.raises(ValueError, match="trailing 12-month totals"):
        trailing_12_month_totals(df)


def test_coverage_report_detects_boundary_and_internal_missing_months():
    dates = pd.date_range("2024-03-01", "2024-10-01", freq="MS").delete(3)
    data = pd.DataFrame({"country": "001", "date": dates})

    report = coverage_report(
        data,
        group_by=("country",),
        expected_start="2024-01-01",
        expected_end="2024-12-01",
    )

    row = report.iloc[0]
    assert row["expected_start"] == "2024-01"
    assert row["expected_end"] == "2024-12"
    assert row["expected_months"] == 12
    assert row["observed_months"] == 7
    assert row["missing_months"] == 5
    assert row["missing_leading_months"] == 2
    assert row["internal_missing_months"] == 1
    assert row["missing_trailing_months"] == 2
    assert row["missing_periods"] == ["2024-01", "2024-02", "2024-06", "2024-11", "2024-12"]


def test_coverage_report_preserves_inferred_bounds_by_default():
    dates = pd.to_datetime(["2024-03-01", "2024-05-01"])
    report = coverage_report(
        pd.DataFrame({"country": "001", "date": dates}),
        group_by=("country",),
    )

    row = report.iloc[0]
    assert row["expected_start"] == "2024-03"
    assert row["expected_end"] == "2024-05"
    assert row["missing_periods"] == ["2024-04"]
    assert row["internal_missing_months"] == 1
    assert row["missing_leading_months"] == 0
    assert row["missing_trailing_months"] == 0


def test_coverage_report_validates_expected_bounds():
    data = pd.DataFrame({"country": ["001"], "date": ["2024-01-01"]})
    with pytest.raises(ValueError, match="both expected_start and expected_end"):
        coverage_report(data, group_by=("country",), expected_start="2024-01-01")
    with pytest.raises(ValueError, match="on or after"):
        coverage_report(
            data,
            group_by=("country",),
            expected_start="2024-02-01",
            expected_end="2024-01-01",
        )


def test_cli_hs_search():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "japantrade.cli",
            "hs",
            "search",
            "live animals",
            "--limit",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "code" in result.stdout
