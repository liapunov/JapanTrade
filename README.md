# JapanTrade
Growing repository of tools for analyzing Japanese customs trade data.
Python / Pandas + Seaborn
The biggest problem in working with Japanese Customs data is that, as of April 2020, there does not seem to be an easily accesible API for the trade data - indeed, there is no easy way to slice any piece of information from the website, which only provides csv tables divided by months and including only part of the HS (Harmonized System) codes in each file.
What we are trying to do here is:
- First of all, starting from single csv files, obtain a highly normalized version of the time series
- secondly, extract and normalize the description of each HS code, as the codes descriptions are often distributed hierarchically over many lines of a csv or worse, HTML table.
- finally, create a few methods to extract common queries, such as the comparison year-on-year of a specific HS code for a specific country, or the chart of the time series for a specific HS code and a group of countries.

## Streamlit explorer and SQL exports
- A lightweight Streamlit app (`src/japantrade/app.py`) loads normalized datasets, provides filters for kind/country/code/unit/date, and visualizes YoY trends, top products, and country comparisons. Launch with:
  ```
  streamlit run src/japantrade/app.py
  ```
- DuckDB/SQLite exports and example parameterized queries live in `japantrade.exporters` and are demonstrated in the app as well as `notebooks/query_export_template.ipynb`.
- `tests/fixtures/normalized_sample.csv` is a minimal fixture dataset that powers notebook examples and CI smoke checks.

## Notebook templates
- `notebooks/trend_analysis_template.ipynb` demonstrates common visualizations (line, bar, treemap) using the analytics helpers.
- `notebooks/query_export_template.ipynb` shows how to export to DuckDB/SQLite and run the provided example queries.

## CLI filtering and aggregates
- A lightweight CLI is available via `python -m japantrade.cli <normalized.csv>` with filters for country lists/prefixes, HS/PC code prefixes, units, and optional date bounds.
- Add `--aggregate mom`/`--aggregate yoy`/`--aggregate trailing12` to compute growth metrics once the filtered dataset includes the required monthly coverage.
- Use `--output` to persist filtered rows to CSV while printing a quick preview to stdout.

## Available modules as of March 24, 2021
- Japanese_HS_Codes.ipynb: extract the 4, 6 and 9 lengths HS (harmonized system) codes and descriptions from a static csv table.
- customsgrabber.py: tools for fetching trade data from the customs.or.jp website 
- tradefile.py: tools for extracting, merging and normalising into a pandas DF the trade data csv tables that the customs.or.jp website provides
- tradeanalysis.py: tools for creating useful tables and charts from a preprocessed trade data frame

## Next To Do as of March 24, 2021
- tradeanalysis.py: refactor TradeReport.yoy_country_report; develop TradeReport.comparison_chart_by_country
- Refactor the Japanese_HS_codes.ipynb module
- Package the modules

# Development plan
- Tools for adding HS/PC code descriptions from the HTML tables available in the customs.or.jp website, in particular:
-- The assertions are still missing
-- The textual descriptions have not been included yet
- TradeReport.comparison_chart_by_country, in order to compare and chart the trend of a given code for different countries
- Tools for finding an HS code by textual approximate search (NLP module will be needed)
- Web servicization with Flask
