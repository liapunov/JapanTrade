import asyncio
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from japantrade import mcp_server


def _dataset(tmp_path):
    rows = []
    for country, name, multiplier in (("220", "Italy", 1), ("302", "United States", 2)):
        for date in pd.date_range("2023-01-01", periods=24, freq="MS"):
            rows.append({
                "direction": "export", "kind": "HS", "country": country,
                "country_name": name, "code": "870321000", "date": date,
                "unit": "JPY", "value": 100 * multiplier,
            })
    path = tmp_path / "trade.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_mcp_tools_are_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {tool.name for tool in tools} == {
        "inspect_dataset", "find_hs_codes", "rank_country_products",
        "compare_product_countries", "dataset_coverage",
    }


def test_mcp_stdio_server_starts_and_waits_for_input():
    process = subprocess.Popen(
        [str(Path(sys.executable).with_name("japantrade-mcp"))],
        cwd=str(Path(__file__).parents[1]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=1)
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_mcp_inspection_ranking_and_csv_export(tmp_path):
    path = _dataset(tmp_path)
    inspection = mcp_server.inspect_dataset(str(path))
    assert inspection["directions"] == ["export"]

    output = tmp_path / "reports" / "italy.csv"
    ranking = mcp_server.rank_country_products(str(path), "Italy", "export", output_csv=str(output))
    assert ranking["row_count"] == 1
    assert ranking["returned_record_count"] == 1
    assert ranking["truncated"] is False
    assert ranking["records"][0]["hs_code"] == "8703"
    assert output.exists()


def test_mcp_comparison_and_coverage(tmp_path):
    path = _dataset(tmp_path)
    comparison = mcp_server.compare_product_countries(
        str(path), ["Italy", "United States"], ["8703"], "export"
    )
    assert comparison["row_count"] == 2
    coverage = mcp_server.dataset_coverage(str(path), direction="export", countries=["220"])
    assert coverage["records"][0]["missing_months"] == 0


def test_table_response_truncates_records_but_writes_complete_csv(tmp_path):
    output = tmp_path / "complete.csv"
    response = mcp_server._table_response(
        pd.DataFrame({"value": range(5)}),
        str(output),
        max_records=2,
    )

    assert response["row_count"] == 5
    assert response["returned_record_count"] == 2
    assert response["truncated"] is True
    assert len(response["records"]) == 2
    assert len(pd.read_csv(output)) == 5


def test_table_response_rejects_invalid_max_records():
    with pytest.raises(ValueError, match="at least 1"):
        mcp_server._table_response(pd.DataFrame({"value": [1]}), max_records=0)
