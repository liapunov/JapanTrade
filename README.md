# JapanTrade

JapanTrade downloads, prepares, and analyzes Japanese Customs HS import and export data. It supports two analyst workflows: ranking product categories for one trading partner and comparing selected HS categories across countries.

The data comes from the Japanese government e-Stat/Japan Customs releases. JapanTrade is not an official government service. Record release periods and source metadata when publishing results.

## Install

Python 3.10+ is required. Use Parquet for prepared datasets.

```bash
pip install "japantrade[parquet]"
```

For repository development, create a development environment:

```bash
uv sync --extra dev
uv run pytest -q
```

## Quick start

Discover HS categories without downloading any data:

```bash
japantrade hs search "electric vehicle" --level 4
japantrade hs search 8703 --limit 10
```

Download and prepare one direction at a time. e-Stat can be slow; begin with one year and use a longer timeout through `CustomsGrabber` if the CLI download times out.

```bash
japantrade download --direction export --years 2025:2025 --output raw/exports
japantrade prepare raw/exports/export_HS_2025-2025.zip --direction export --output data/exports_2025.parquet
```

For multiple yearly ZIPs, normalize and merge them with the current package API. The notebook [Trade Tools](src/japantrade/Trade%20Tools.ipynb) contains a copyable example.

Run analyses against an explicit prepared dataset:

```bash
japantrade country-rank data/japan_exports_2023_2025.parquet \
  --country Italy --direction export --hs-level 4 --limit 20 \
  --output results/italy_exports_hs4.csv

japantrade compare data/japan_exports_2023_2025.parquet \
  --countries Italy Germany USA --codes 8703 --direction export \
  --output results/hs8703_comparison.csv
```

`--output` writes CSV and creates the parent directory when needed.

## Analyst API

```python
from japantrade import (
    country_product_ranking,
    load_normalized_data,
    product_country_comparison,
    search_hs,
)

data = load_normalized_data("data/japan_exports_2023_2025.parquet")
matches = search_hs("machine tool", level=4)

italy = country_product_ranking(data, country="Italy", direction="export")
cars = product_country_comparison(
    data, countries=["Italy", "Germany", "USA"], codes=["8703"], direction="export"
)
```

Country inputs accept an official Japan Customs code or an exact English country name. HS searches accept text, exact codes, and code prefixes.

## Data contract and interpretation

Prepared datasets require `direction`, `kind`, `country`, `code`, `date`, `unit`, and `value`. They may also contain `country_name` and `code_description`. Direction is part of a record identity, so imports and exports cannot overwrite one another.

V1 comparisons use JPY value only. Quantity rows are preserved but are not comparable across different units. Rankings default to HS-4 and compare the latest 12 available months with the preceding 12 months; pass both `--date-start` and `--date-end` to choose another window.

Japan Customs rows use 9-digit tariff codes while the bundled HS lookup usually has 2-, 4-, and 6-digit entries. JapanTrade uses the most-specific available prefix description. Ranking reports label the requested HS level directly.

## Troubleshooting

- **e-Stat timeout:** Retry a single year first. For a longer timeout, call `CustomsGrabber.grabRange(..., request_timeout=180)` from Python.
- **“Insufficient data” in a ranking:** Run `dataset_coverage` through the MCP plugin or inspect dates/direction in the dataset. The default comparison needs two complete 12-month windows.
- **Direction mismatch:** Do not combine imports and exports under one direction. Re-prepare raw files with the correct explicit direction.
- **Missing descriptions:** Use `enrich_hs_descriptions` to refresh an existing prepared dataset, or re-run the ranking after upgrading JapanTrade.

## Codex MCP plugin

The repository includes a local Codex plugin for prepared-dataset analysis. See the [MCP plugin guide](plugins/japantrade/README.md) for installation, tools, CSV export, examples, and troubleshooting.

## Streamlit explorer

The exploratory Streamlit dashboard in `src/japantrade/app.py` provides filtering, charts, CSV export, and example DuckDB queries. It ships with the bundled fixture at `tests/fixtures/normalized_sample.csv`; you can also upload a normalized CSV.

The legacy notebooks have been rewritten as current tutorials. The supported interfaces are the Python API, CLI, MCP server, and these notebooks; the Streamlit app remains exploratory.
