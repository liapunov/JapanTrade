"""Output path and metadata helpers for normalized datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def normalize_compression(compression: str | None) -> str | None:
    aliases = {"gz": "gzip", "bz": "bz2"}
    compression = aliases.get(compression, compression)
    if compression not in {None, "zip", "gzip", "bz2"}:
        raise ValueError("Unsupported compression. Use 'zip', 'gzip', or 'bz2'.")
    return compression


def compression_from_suffix(suffix: str) -> str | None:
    return {".gz": "gzip", ".gzip": "gzip", ".zip": "zip", ".bz2": "bz2"}.get(suffix)


def extension_for(fmt: str, compression: str | None) -> str:
    compression = normalize_compression(compression)
    if fmt.lower() == "csv":
        return {"zip": ".csv.zip", "gzip": ".csv.gz", "bz2": ".csv.bz2", None: ".csv"}[compression]
    if fmt.lower() == "parquet":
        return ".parquet"
    raise ValueError("Unsupported format. Use 'csv' or 'parquet'.")


def build_output_path(data: pd.DataFrame, kind: str, path, filename, fmt, compression):
    base_path = Path(path)
    target_dir = base_path.parent if base_path.suffix else base_path
    if base_path.suffix and filename is None:
        filename = base_path.name
    resolved_fmt = fmt.lower() if fmt else None
    resolved_compression = normalize_compression(compression)
    if filename:
        target_path = target_dir / filename
        suffixes = target_path.suffixes
        if suffixes:
            detected = compression_from_suffix(suffixes[-1])
            if detected:
                resolved_compression = resolved_compression or detected
                if len(suffixes) > 1:
                    resolved_fmt = resolved_fmt or suffixes[-2].lstrip(".")
            else:
                resolved_fmt = resolved_fmt or suffixes[-1].lstrip(".")
        else:
            resolved_fmt = resolved_fmt or "csv"
            target_path = target_path.with_suffix(extension_for(resolved_fmt, resolved_compression))
    else:
        resolved_fmt = resolved_fmt or "csv"
        dates = data["date"].sort_values()
        filename = f"{kind}_{dates.iloc[0]}_{dates.iloc[-1]}{extension_for(resolved_fmt, resolved_compression)}"
        target_path = target_dir / filename
    if not target_path.suffix:
        resolved_fmt = resolved_fmt or "csv"
        target_path = target_path.with_suffix(extension_for(resolved_fmt, resolved_compression))
    return target_path, resolved_fmt, resolved_compression


def load_saved_file(path: Path, fmt: str, compression: str | None) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path, compression=compression)
    raise ValueError("Unsupported format for loading. Use 'csv' or 'parquet'.")


def build_metadata(data: pd.DataFrame, kind: str, source: str | None) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "directions": sorted(data["direction"].astype(str).unique()),
        "source_files": [source] if source else [],
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "min_date": str(pd.to_datetime(data["date"]).min().date()),
        "max_date": str(pd.to_datetime(data["date"]).max().date()),
        "rows": int(len(data)),
    }
