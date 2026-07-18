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


def test_ranking_rejects_an_incomplete_current_or_previous_window():
    data = _dataset()
    data = data[~((data.country == "220") & (data.date == pd.Timestamp("2024-04-01")))]

    with pytest.raises(ValueError, match=r"220 current window missing 2024-04"):
        country_product_ranking(data, "Italy", "export")


def test_comparison_validates_each_country_before_product_filtering():
    data = _dataset()
    data = data[~((data.country == "302") & (data.date == pd.Timestamp("2023-06-01")))]

    with pytest.raises(ValueError, match=r"302 previous window missing 2023-06"):
        product_country_comparison(data, ["Italy", "302"], ["8703"], "export")


def test_comparison_treats_missing_selected_product_rows_as_zero_trade():
    data = _dataset()
    coverage_rows = data[data.code == "870322"].copy()
    selected_rows = data[(data.code == "870321") & (data.date.dt.year == 2024)].copy()
    result = product_country_comparison(
        pd.concat([coverage_rows, selected_rows], ignore_index=True),
        ["Italy", "302"],
        ["870321"],
        "export",
    )

    assert (result["prior_value"] == 0).all()
    assert result["growth"].isna().all()


def test_comparison_rejects_malformed_prefixes_and_collapses_overlaps():
    data = _dataset()
    broad = product_country_comparison(data, ["Italy"], ["87"], "export")
    overlapping = product_country_comparison(data, ["Italy"], ["87", "8703"], "export")
    assert overlapping.loc[0, "current_value"] == broad.loc[0, "current_value"]
    with pytest.raises(ValueError, match="numeric"):
        product_country_comparison(data, ["Italy"], ["87A"], "export")


def test_analytics_collapses_exact_duplicates_and_rejects_conflicts():
    data = _dataset()
    duplicate = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    assert country_product_ranking(duplicate, "Italy", "export").loc[0, "current_value"] == 1800

    conflict = data.iloc[[0]].copy()
    conflict["value"] = 999
    with pytest.raises(ValueError, match="Conflicting values"):
        country_product_ranking(pd.concat([data, conflict], ignore_index=True), "Italy", "export")


def test_mixed_directions_do_not_leak_into_export_results():
    data = _dataset()
    imports = data.assign(direction="import", value=100000)
    result = country_product_ranking(pd.concat([data, imports]), "Italy", "export")
    assert result.loc[0, "current_value"] == 1800


def test_ambiguous_country_names_are_rejected():
    data = _dataset()
    alias = data[data.country == "302"].copy()
    alias["country"] = "999"
    alias["country_name"] = "Italy"
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        country_product_ranking(pd.concat([data, alias]), "Italy", "export")


def test_loader_requires_direction(tmp_path):
    path = tmp_path / "legacy.csv"
    pd.DataFrame({"kind": ["HS"], "country": ["220"], "code": ["8703"], "date": ["2024-01-01"], "unit": ["JPY"], "value": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="direction"):
        load_normalized_data(path)


def test_hs_description_enrichment_uses_six_digit_prefix_for_tariff_codes():
    data = pd.DataFrame({"code": ["010110000"]})
    enriched = enrich_hs_descriptions(data)
    assert pd.notna(enriched.loc[0, "code_description"])
