---
name: japantrade-analysis
description: Analyze prepared JapanTrade import/export datasets with the JapanTrade MCP tools. Use when an analyst needs HS code discovery, dataset coverage checks, country product rankings, multi-country product comparisons, or a CSV export of one of those analyses.
---

# JapanTrade analysis

Use the JapanTrade MCP tools for prepared local CSV or Parquet data. Require an explicit dataset path in every analysis request.

## Workflow

1. Call `inspect_dataset` before analysis when coverage or direction is unknown; do not infer a dataset path.
2. Call `find_hs_codes` before choosing unfamiliar categories. Prefer HS-2 for sector views, HS-4 for rankings, and HS-6 for detail.
3. Use `rank_country_products` for one country and many product categories. Use `compare_product_countries` for one or more product prefixes across countries.
4. Use `dataset_coverage` when a comparison fails, a direction is uncertain, or the requested period may be incomplete. Pass both `expected_start` and `expected_end` when the user names a reporting interval so boundary gaps are included.
5. State that values are JPY and name the comparison period. The default is the latest 12 available months versus the preceding 12 months.
6. Set `output_csv` only when the user asks to save or export the result; require a `.csv` path and report the returned `csv_path`.

Do not use these tools to download or prepare raw data. Ask the user for a prepared dataset path if none is available.
Treat absent selected-product rows as zero only for complete HS-universe extracts; warn that partial/custom extracts may have omitted the product instead.
