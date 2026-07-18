"""Schema and identity rules for normalized JapanTrade datasets."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"direction", "kind", "country", "code", "date", "unit", "value"}
PRIMARY_KEY = ("direction", "kind", "country", "code", "date", "unit")
VALID_DIRECTIONS = {"import", "export"}


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a migration-oriented error when normalized columns are missing."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if not missing:
        return
    if missing == {"direction"}:
        raise ValueError("Dataset has no 'direction' column. Re-run `japantrade prepare` from raw data.")
    raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")


def validate_direction_values(values: Iterable[object]) -> None:
    directions = {str(value) for value in values if pd.notna(value)}
    if not directions.issubset(VALID_DIRECTIONS):
        raise ValueError("direction must contain only 'import' and 'export'.")


def deduplicate_normalized_data(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical records and reject conflicting primary-key values.

    The normalized primary key describes one measure. Multiple distinct values
    for that identity cannot be combined safely without source-specific rules.
    """
    validate_required_columns(df)
    key = list(PRIMARY_KEY)
    conflicts = df.groupby(key, dropna=False)["value"].nunique(dropna=False)
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        sample = conflicts.index[0]
        identity = dict(zip(key, sample if isinstance(sample, tuple) else (sample,)))
        raise ValueError(f"Conflicting values found for normalized primary key: {identity}")
    return df.drop_duplicates(subset=key, keep="first").copy()
