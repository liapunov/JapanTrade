import pandas as pd
import pytest

from japantrade.tradefile import TradeFile


def _build_tradefile():
    tradefile = TradeFile.__new__(TradeFile)
    tradefile.persist_path = None
    tradefile.persist_format = None
    tradefile.chunk_size = 50000
    tradefile.latest_timings = []
    return tradefile


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
