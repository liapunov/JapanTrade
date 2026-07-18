# JapanTrade Codex MCP plugin

Use this local plugin to analyze a prepared JapanTrade CSV or Parquet dataset from Codex. It does not download or normalize raw Customs archives; use the JapanTrade CLI or the ingestion notebook first.

## Install

From the repository root, install the MCP extra:

```bash
uv sync --extra mcp
```

Add the repository marketplace, then install the plugin:

```bash
codex plugin marketplace add /absolute/path/to/JapanTrade/.agents/plugins
codex plugin add japantrade@personal
```

Start a new Codex thread after installation. When updating the plugin locally, refresh its cachebuster with the plugin-creator helper, reinstall it from the same marketplace, and start another new thread.

## Tools

| Tool | Use it for | Required inputs |
| --- | --- | --- |
| `inspect_dataset` | Verify schema, direction, date range, countries, and codes. | `dataset_path` |
| `find_hs_codes` | Find categories by keyword or HS prefix. | `query` |
| `rank_country_products` | Rank HS categories for one partner country. | `dataset_path`, `country`, `direction` |
| `compare_product_countries` | Compare selected HS prefixes across countries. | `dataset_path`, `countries`, `codes`, `direction` |
| `dataset_coverage` | Diagnose internal and boundary month gaps before or after an analysis. | `dataset_path` |

All dataset paths are explicit and must end in `.parquet` or `.csv`. Analysis tools return at most 100 structured records by default; use `max_records` to request a different response limit. Responses include total and returned row counts plus a `truncated` flag. Add `output_csv` ending in `.csv` when you want the complete table written; the server creates its parent directory and returns `csv_path` even when structured records are truncated.

For a known reporting interval, pass both `expected_start` and `expected_end` to `dataset_coverage`. Its records distinguish internal, leading, and trailing missing months and include the exact missing periods. Without explicit bounds, expected coverage continues to run from each series' first observed month through its last.

## Recommended workflow

1. Ask Codex to inspect the dataset, for example: “Inspect `data/japan_exports_2023_2025.parquet`.”
2. Search unfamiliar products: “Find the HS-4 code for electric vehicles.”
3. Run a country ranking or multi-country comparison.
4. Ask for coverage if a period is incomplete or a comparison fails.
5. Ask for CSV export only when you need a saved table.

Example prompts:

- “Inspect `data/japan_exports_2023_2025.parquet`, then rank Japan’s exports to Italy by HS-4 category.”
- “Compare exports of HS 8703 to Italy, Germany, and the USA, and save the result to `results/hs8703.csv`.”
- “Check monthly coverage for exports to Italy before comparing the latest 12 months with the previous 12.”

## Interpretation and troubleshooting

Values are JPY. The default analysis period is the latest 12 available months versus the preceding 12 months. Country names must be exact English lookup names, or use Japan Customs country codes.

Comparisons treat missing selected-product rows as zero only when the prepared dataset represents the complete HS universe. A partial/custom extract cannot distinguish an omitted product from zero reported trade.

- **Server does not start:** Run `uv sync --extra mcp` at the repository root and confirm the plugin’s MCP configuration still has that repository as its working directory.
- **Dataset error:** Use an existing prepared CSV/Parquet with a `direction` column. The plugin intentionally does not infer a dataset path or a direction.
- **Incomplete comparison window:** Call `dataset_coverage`; the analysis error lists the countries and calendar months missing from either required window.
- **CSV not written:** Supply an explicit `.csv` `output_csv` path and ensure its parent directory is writable.
