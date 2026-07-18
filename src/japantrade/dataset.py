"""Normalized dataset identity, snapshots, and merge behavior."""
from __future__ import annotations

import pandas as pd

from .schema import PRIMARY_KEY, deduplicate_normalized_data


def snapshot_state(df: pd.DataFrame) -> dict[str, int]:
    key_columns = [column for column in PRIMARY_KEY if column in df.columns]
    ordered = df.sort_values(by=key_columns) if key_columns else df
    checksum = pd.util.hash_pandas_object(ordered.reset_index(drop=True).fillna(""), index=False).sum()
    return {"rows": len(df), "checksum": int(checksum)}


def merge_normalized_data(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    date_range: tuple[object, object] | None = None,
) -> pd.DataFrame:
    """Merge data, allowing conflicts only through explicit window replacement."""
    if date_range:
        start, end = date_range
        incoming = incoming[incoming["date"].between(start, end)]
        existing = existing[~existing["date"].between(start, end)]
    combined = pd.concat([existing, incoming], ignore_index=True)
    return deduplicate_normalized_data(combined)
