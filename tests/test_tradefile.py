import logging

import pandas as pd
import pytest

from japantrade.tradefile import NormalizationConfig, TradeFile


def _build_tradefile(config: NormalizationConfig | None = None):
    tradefile = TradeFile.__new__(TradeFile)
    tradefile.normalization_config = config or NormalizationConfig()
    tradefile._lookup_cache = {}
    tradefile.persist_path = None
    tradefile.persist_format = None
    tradefile.chunk_size = 50000
    tradefile.latest_timings = []
    tradefile.kind = 'HS'
    return tradefile


def _sample_normalized_df(kind='HS'):
    return pd.DataFrame(
        {
            'kind': [kind, kind],
            'country': ['001', '001'],
            'code': ['0101', '0101'],
            'date': ['2021-01-01', '2021-02-01'],
            'unit': ['KG', 'KG'],
            'value': [100, 200],
        }
    )


def test_melt_months_and_units_hs_branch():
    tf = _build_tradefile()
    hs_df = pd.DataFrame(
        {
            'Year': ['2020'],
            'HS': ['0101'],
            'Country': ['001'],
            'Unit1': ['KG'],
            'Unit2': ['NO'],
            'Quantity1-Jan': [10],
            'Quantity2-Jan': [5],
            'Value-Jan': [2],
            'Quantity1-Feb': [20],
            'Quantity2-Feb': [10],
            'Value-Feb': [3],
        }
    )

    melted_months, metrics = tf._meltMonths(hs_df, 'HS')
    assert set(metrics.keys()) == {
        'melt_time', 'split_time', 'clean_time', 'date_merge_time'
    }

    melted_units, unit_metrics = tf._meltUnits(melted_months, 'HS')

    assert set(melted_units['unit']) == {'KG', 'NO', 'JPY'}
    value_rows = melted_units[melted_units['unit'] == 'JPY']['value']
    assert (value_rows.values == pd.Series([2000, 3000]).values).all()
    assert set(unit_metrics.keys()) == {
        'value_cast_time', 'rename_time', 'unit_assignment_time'
    }


def test_melt_months_and_units_pc_branch():
    tf = _build_tradefile()
    pc_df = pd.DataFrame(
        {
            'Year': ['2021'],
            'Commodity': ['1000'],
            'Country': ['002'],
            'Unit': ['TH'],
            'Quantity-Jan': [100],
            'Value-Jan': [7],
        }
    )

    melted_months, _ = tf._meltMonths(pc_df, 'PC')
    melted_units, _ = tf._meltUnits(melted_months, 'PC')

    assert set(melted_units['unit']) == {'TH', 'JPY'}
    assert melted_units.loc[melted_units['unit'] == 'JPY', 'value'].iloc[0] == 7000
    assert set(melted_units.columns) == {'code', 'country', 'unit', 'date', 'value'}


def test_melt_units_missing_unit_error():
    tf = _build_tradefile()
    pc_df = pd.DataFrame(
        {
            'Year': ['2021'],
            'Commodity': ['1000'],
            'Country': ['002'],
            'Unit': [''],
            'Quantity-Jan': [100],
            'Value-Jan': [7],
        }
    )

    melted_months, _ = tf._meltMonths(pc_df, 'PC')
    with pytest.raises(ValueError):
        tf._meltUnits(melted_months, 'PC')


def test_melt_units_mixed_value_rows_raise():
    tf = _build_tradefile()
    pc_df = pd.DataFrame(
        {
            'Year': ['2021'],
            'Commodity': ['1000'],
            'Country': ['002'],
            'Unit': ['TH'],
            'Quantity-Jan': [100],
            'Value-Jan': ['not-a-number'],
        }
    )

    melted_months, _ = tf._meltMonths(pc_df, 'PC')
    with pytest.raises(ValueError):
        tf._meltUnits(melted_months, 'PC')


def test_acquire_new_data_deduplicates_and_enforces_kind():
    tf = _build_tradefile()
    tf.data = _sample_normalized_df(kind='HS')
    tf.data = tf._ensure_kind_column(tf.data, 'HS')

    new_df = pd.DataFrame(
        {
            'kind': ['HS', 'HS'],
            'country': ['001', '001'],
            'code': ['0101', '0101'],
            'date': ['2021-02-01', '2021-03-01'],
            'unit': ['KG', 'KG'],
            'value': [300, 400],
        }
    )

    merged = tf._acquireNewData(new_df=new_df, kind='HS')
    assert len(merged) == 3
    assert merged[merged['date'] == '2021-02-01']['value'].iloc[0] == 200
    with pytest.raises(ValueError):
        tf._acquireNewData(
            new_df=new_df.assign(kind='PC'),
            kind='PC'
        )


def test_acquire_new_data_incremental_date_window(tmp_path):
    tf = _build_tradefile()
    tf.data = _sample_normalized_df(kind='HS')
    tf.data = tf._ensure_kind_column(tf.data, 'HS')

    new_df = pd.DataFrame(
        {
            'kind': ['HS', 'HS'],
            'country': ['001', '001'],
            'code': ['0101', '0101'],
            'date': ['2021-01-01', '2021-02-01'],
            'unit': ['KG', 'KG'],
            'value': [150, 250],
        }
    )

    merged = tf._acquireNewData(new_df=new_df, kind='HS', date_range=('2021-01-01', '2021-02-01'))
    assert len(merged) == 2
    assert set(merged['value']) == {150, 250}


def test_save_to_file_with_custom_filename_and_parquet(tmp_path):
    tf = _build_tradefile()
    tf.data = _sample_normalized_df(kind='HS')
    tf.data = tf._ensure_kind_column(tf.data, 'HS')
    target = tf.save_to_file(path=tmp_path, filename="custom_output", fmt="parquet")
    assert target.exists()
    saved = pd.read_parquet(target)
    assert len(saved) == len(tf.data)


def test_normalize_units_conversion_and_filtering(caplog):
    config = NormalizationConfig(convert_to_base_units=True, keep_units=("KG", "NO"), warn_on_unknown=True)
    tf = _build_tradefile(config)
    df = pd.DataFrame(
        {
            "unit": ["kg", "th", "UNK"],
            "value": [1, 2, 3],
            "code": ["0000"] * 3,
            "country": ["000"] * 3,
            "date": ["2023-01-01"] * 3,
            "kind": ["HS"] * 3,
        }
    )
    with caplog.at_level(logging.WARNING):
        normalized, metrics = tf._normalize_units(df)
    assert set(normalized["unit"]) == {"KG", "NO"}
    assert normalized.loc[normalized["unit"] == "NO", "value"].iloc[0] == 2000
    assert metrics["unit_conversions"] == 1
    assert metrics["filtered_rows"] == 1
    assert any("unknown unit codes" in record.message for record in caplog.records)


def test_enrich_with_lookup_cache(tmp_path):
    config = NormalizationConfig(include_descriptions=True, warn_on_unknown=False)
    tf = _build_tradefile(config)
    df = pd.DataFrame(
        {
            "code": ["0101"],
            "country": ["103"],
            "unit": ["KG"],
            "date": ["2023-01-01"],
            "value": [100],
            "kind": ["HS"],
        }
    )
    enriched, metrics = tf._enrich_with_lookups(df, "HS")
    assert "code_description" in enriched.columns
    assert "country_name" in enriched.columns
    assert "hs" in tf._lookup_cache
    # second call should reuse cache and preserve metrics shape
    enriched_again, metrics_again = tf._enrich_with_lookups(df, "HS")
    assert metrics_again.keys() == metrics.keys()
