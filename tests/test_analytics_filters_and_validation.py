import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from japantrade.analytics import (
    apply_parameterized_filters,
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


def test_cli_filters_and_mom(tmp_path):
    source = Path(__file__).parent / "fixtures" / "normalized_sample.csv"
    output = tmp_path / "filtered.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "japantrade.cli",
            str(source),
            "--countries",
            "001",
            "--aggregate",
            "mom",
            "--codes",
            "0101",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Rows after filtering" in result.stdout
    assert output.exists()
