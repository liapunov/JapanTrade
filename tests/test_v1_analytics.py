import pandas as pd
import pytest

from japantrade.analytics import (
    country_product_ranking,
    enrich_hs_descriptions,
    load_normalized_data,
    product_country_comparison,
)


def _dataset():
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    rows = []
    for country, name, multiplier in (("220", "Italy", 1), ("302", "United States", 2)):
        for date in dates:
            rows.extend([
                {"direction": "export", "kind": "HS", "country": country, "country_name": name, "code": "870321", "date": date, "unit": "JPY", "value": 100 * multiplier},
                {"direction": "export", "kind": "HS", "country": country, "country_name": name, "code": "870322", "date": date, "unit": "JPY", "value": 50 * multiplier},
            ])
    return pd.DataFrame(rows)


def test_country_ranking_rolls_up_hs_prefix_and_resolves_name():
    result = country_product_ranking(_dataset(), "Italy", "export", hs_level=4)
    assert result.loc[0, "hs_code"] == "8703"
    assert result.loc[0, "current_value"] == 1800
    assert pd.notna(result.loc[0, "code_description"])


def test_product_comparison_returns_market_share():
    result = product_country_comparison(_dataset(), ["Italy", "302"], ["8703"], "export")
    assert result.loc[0, "country"] == "302"
    assert result["selected_market_share"].sum() == pytest.approx(1)


def test_loader_requires_direction(tmp_path):
    path = tmp_path / "legacy.csv"
    pd.DataFrame({"kind": ["HS"], "country": ["220"], "code": ["8703"], "date": ["2024-01-01"], "unit": ["JPY"], "value": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="direction"):
        load_normalized_data(path)


def test_hs_description_enrichment_uses_six_digit_prefix_for_tariff_codes():
    data = pd.DataFrame({"code": ["010110000"]})
    enriched = enrich_hs_descriptions(data)
    assert pd.notna(enriched.loc[0, "code_description"])
