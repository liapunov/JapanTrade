import asyncio
from pathlib import Path

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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


def test_mcp_stdio_server_lists_tools():
    async def list_tools():
        params = StdioServerParameters(
            command="uv", args=["run", "japantrade-mcp"], cwd=str(Path(__file__).parents[1])
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.list_tools()

    result = asyncio.run(list_tools())
    assert "rank_country_products" in {tool.name for tool in result.tools}


def test_mcp_inspection_ranking_and_csv_export(tmp_path):
    path = _dataset(tmp_path)
    inspection = mcp_server.inspect_dataset(str(path))
    assert inspection["directions"] == ["export"]

    output = tmp_path / "reports" / "italy.csv"
    ranking = mcp_server.rank_country_products(str(path), "Italy", "export", output_csv=str(output))
    assert ranking["row_count"] == 1
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
