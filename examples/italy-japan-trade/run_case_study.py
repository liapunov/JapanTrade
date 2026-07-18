"""Reproduce the synthetic Italy–Japan analytical case study."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from japantrade import country_product_ranking, product_country_comparison


def build_dataset(scenario: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in scenario.itertuples(index=False):
        for year in (2023, 2024):
            monthly_value = getattr(item, f"monthly_value_{year}")
            for date in pd.date_range(f"{year}-01-01", periods=12, freq="MS"):
                rows.append({
                    "direction": "export",
                    "kind": "HS",
                    "country": str(item.country).zfill(3),
                    "country_name": item.country_name,
                    "code": str(item.code),
                    "date": date,
                    "unit": "JPY",
                    "value": monthly_value,
                })
    return pd.DataFrame(rows)


def write_svg(ranking: pd.DataFrame, path: Path) -> None:
    rows = ranking.sort_values("rank").head(5)
    maximum = float(rows["current_value"].max())
    bars = []
    for index, row in enumerate(rows.itertuples(index=False)):
        y = 65 + index * 55
        width = 360 * float(row.current_value) / maximum if maximum else 0
        bars.append(
            f'<text x="20" y="{y + 18}" font-size="16">HS {row.hs_code}</text>'
            f'<rect x="100" y="{y}" width="{width:.1f}" height="26" fill="#c62828" rx="3"/>'
            f'<text x="{108 + width:.1f}" y="{y + 19}" font-size="14">JPY {row.current_value:,.0f}</text>'
        )
    height = 95 + len(rows) * 55
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="{height}" viewBox="0 0 620 {height}">'
        '<rect width="100%" height="100%" fill="#fffaf2"/>'
        '<text x="20" y="34" font-family="sans-serif" font-size="22" font-weight="700">'
        'Synthetic exports from Japan to Italy, 2024</text>'
        f'<g font-family="sans-serif" fill="#202124">{"".join(bars)}</g></svg>'
    )
    path.write_text(svg, encoding="utf-8")


def run(output_dir: Path) -> None:
    source_dir = Path(__file__).resolve().parent
    scenario = pd.read_csv(
        source_dir / "synthetic_scenario.csv",
        dtype={"country": str, "code": str},
    )
    data = build_dataset(scenario)
    ranking = country_product_ranking(data, "Italy", "export", hs_level=4, limit=10)
    comparison = product_country_comparison(
        data,
        ["Italy", "Germany", "United States of America"],
        ["8703"],
        "export",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "synthetic_trade.csv", index=False)
    ranking.to_csv(output_dir / "italy_hs4_ranking.csv", index=False)
    comparison.to_csv(output_dir / "hs8703_country_comparison.csv", index=False)
    write_svg(ranking, output_dir / "italy_hs4_ranking.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
