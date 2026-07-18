"""Lookup loading helpers for normalized dataset enrichment."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_code_lookup(path: Path) -> pd.DataFrame | None:
    data = pd.read_csv(path, delimiter=";", dtype=str)
    data = data.rename(columns={"Code.1": "code", "Description": "code_description"})
    if "code" not in data.columns or "code_description" not in data.columns:
        return None
    data["code"] = data["code"].astype(str).str.replace(r"\s+", "", regex=True)
    return data[["code", "code_description"]].dropna()


def load_country_lookup(path: Path) -> pd.DataFrame | None:
    data = pd.read_csv(path, dtype=str)
    data = data.rename(columns={column: column.strip().lower().replace(" ", "_") for column in data.columns})
    if "code" not in data.columns:
        return None
    data["code"] = data["code"].astype(str).str.zfill(3)
    name_column = "country" if "country" in data.columns else data.columns[-1]
    zone_column = "geographical_zone" if "geographical_zone" in data.columns else None
    columns = ["code", name_column] + ([zone_column] if zone_column else [])
    return data[columns].rename(columns={name_column: "country_name", "code": "country"})
