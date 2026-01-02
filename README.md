# JapanTrade

Tools for acquiring, cleaning, analyzing, and interactively exploring Japanese customs trade data when no official API is available. The library focuses on:

* Downloading monthly CSV releases from the customs repository.
* Normalizing wide monthly tables into long, analysis-ready data.
* Producing quick exploratory reports, notebook examples, and a Streamlit dashboard for exploratory analysis.

> Note: The repository ships Python tools, research notebooks, and a Streamlit app for interactive exploration. A CLI is still on the roadmap.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/japantrade/customsgrabber.py` | Download helper for monthly customs CSV archives (import/export, HS/PC classification). |
| `src/japantrade/tradefile.py` | Normalizes raw CSV/ZIP exports into a tidy pandas DataFrame. |
| `src/japantrade/tradeanalysis.py` | Lightweight reporting utilities on top of normalized data. |
| `src/japantrade/Trade Tools.ipynb` | End-to-end examples: download → normalize → basic visuals. |
| `src/japantrade/Trade data analysis.ipynb` | Exploratory analysis on normalized datasets. |
| `src/japantrade/app.py` | Streamlit "Japan Trade Explorer" dashboard backed by normalized data. |
| `Japanese_HS_Codes.ipynb` | HS code description extraction from static tables. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pandas and BeautifulSoup are required for the core workflow. Jupyter is recommended for running the bundled notebooks.

## Data acquisition: `CustomsGrabber`

Located in `src/japantrade/customsgrabber.py`, `CustomsGrabber` automates downloads from the customs portal.

### Key parameters

* `direction`: `"import"` or `"export"`.
* `kind`: `"HS"` (Harmonized System) or `"PC"` (Japanese product classification).
* `from_year`, `to_year`: inclusive year bounds (≥1988, current year capped).
* `save_folder`: where ZIP files are written (defaults to `../data/` relative to the module).

### Usage examples

```python
from japantrade.customsgrabber import CustomsGrabber

grabber = CustomsGrabber()
# Download a single year of HS import data
grabber.grabRange(from_year=2022, to_year=2022, direction="import", kind="HS", save_folder="./data/")

# Fetch the latest available year
grabber.getLastData(direction="export", kind="PC", save_folder="./data/")
```

The downloader batches requests in chunks of up to 100 files to keep URLs below server limits. It saves ZIP archives you can pass directly to `TradeFile`.

## Data normalization: `TradeFile`

`TradeFile` (in `src/japantrade/tradefile.py`) converts raw customs CSVs or ZIP bundles into long-form pandas DataFrames.

### What it does

1. **Cleaning** – fixes known column typos (e.g., `Apl` → `Apr`), strips code artifacts, drops yearly totals.
2. **Month unpivot** – melts `Quantity1/Quantity2/Value` columns across months into a `date/type/measure` layout.
3. **Unit unpivot + normalization** – consolidates units into a single column, canonizes codes (e.g., `KGS` → `KG`), optionally converts base units (e.g., thousands → absolute counts), and supports keep/exclude lists with warnings on unknowns.
4. **Enrichment** – reuses cached HS/PC and country lookup files (configurable overrides) to attach descriptions during batch processing.
5. **Row reduction** – removes zero or missing values for a compact, analysis-ready table.

### Creating a normalized DataFrame

```python
from japantrade.tradefile import TradeFile

normalized = TradeFile(
    source="./data/import_HS_2022.zip",  # raw CSV or ZIP
    raw=True,                            # set to False if already normalized
    kind="infer"                         # infer HS vs PC from columns
)
df = normalized.data  # columns: kind, country, code, date, unit, value
```

### Merging additional files

`TradeFile` accepts an existing normalized CSV/ZIP (`base_file`) or DataFrame (`base_df`) and will append deduplicated new rows:

```python
base = "./data/normalized_2019_2021.csv"
merged = TradeFile(source="./data/import_HS_2022.zip", base_file=base)
merged.save_to_file(path="./data/normalized_2019_2022.csv", is_zip=False)
```

## Reporting and exploration: `TradeReport`

`TradeReport` (in `src/japantrade/tradeanalysis.py`) provides lightweight analysis helpers on normalized data.

### Current capability

* **YoY country reports**: `yoy_country_report(kind="HS", country="220", method="last_12")` compares the latest 12 months to the prior period, optionally filtering by code length and units.

### Example

```python
from japantrade.tradeanalysis import TradeReport

report = TradeReport(source_file="./data/normalized_2019_2022.csv")
italy_yoy = report.yoy_country_report(
    kind="HS",
    country="220",        # Italy
    code_level=4,         # min code length
    val_only=True,        # value (JPY) only
    method="last_12"      # trailing 12 months vs prior 12
)
```

The returned DataFrame is indexed by HS/PC code and unit, with trailing and prior-period sums ready for visualization in pandas/Seaborn/Matplotlib.

## Streamlit app: Japan Trade Explorer

The Streamlit dashboard (`src/japantrade/app.py`) ships with a bundled fixture (`tests/fixtures/normalized_sample.csv`) so you can explore immediately or upload your own normalized CSV.

### Launch

```bash
streamlit run src/japantrade/app.py
```

### Features

* **Data loading**: Upload normalized CSVs or rely on the bundled sample; data is validated via `load_normalized_data`.
* **Filtering**: Sidebar controls for kind, countries, codes, units, and date range (auto-populated from available data).
* **Summary metrics**: Total value KPI for the active filter set.
* **Trends**: Year-over-year trend chart with validation for minimum monthly coverage.
* **Top products**: Tabular and Altair bar views of top codes by value, plus a Plotly treemap for relative magnitude.
* **Country comparison**: Aggregates by country (optionally scoped to a selected code) with charted results.
* **Exports**: One-click downloads to DuckDB and SQLite files generated from the filtered dataset.
* **Example queries**: Parameterized SQL snippets (DuckDB) such as top exporters in a date window and fastest-growing categories by year, runnable directly in-app.

## Notebooks at a glance

* **`src/japantrade/Trade Tools.ipynb`** – Walks through downloading data, normalizing it with `TradeFile`, and running first-pass plots.
* **`src/japantrade/Trade data analysis.ipynb`** – Deeper exploratory analysis on prepared datasets (country/code slices, trend charts).
* **`Japanese_HS_Codes.ipynb`** – Extracts HS code descriptions from provided tables, demonstrating code-to-description enrichment.

Open notebooks with:

```bash
jupyter notebook src/japantrade/Trade\ Tools.ipynb
```

## Recent updates

* Added the Streamlit-based Japan Trade Explorer with filtering, charts, exports, and runnable example queries.
* Expanded normalization controls (base-unit conversion, keep/exclude filters, and warnings for unknown units) and added cached lookup enrichment to the pipeline.
* Strengthened analysis helpers (`TradeReport`, `analytics.py`) for YoY/MoM trends, trailing totals, and top-product summaries.

## Ideas and roadmap

* CLI tooling: quick filters/aggregations from the terminal, plus CSV/Parquet export commands.
* Performance: chunked CSV ingestion, faster melt paths, and Parquet/Feather outputs for large-scale workflows.
* Automation: scheduled fetch + normalize pipelines with freshness monitoring and artifact publishing.
* Visualization: additional Streamlit views (e.g., MoM trend comparisons, share-of-total charts) and optional alerting for large swings.
* Data quality: richer validation with anomaly detection (e.g., outlier spikes/drops) and clearer warnings in the app/CLI.

Contributions and issue reports are welcome—especially around performance tuning, schema validation, and visualization recipes.
