"""CSV and ZIP ingestion helpers for Japan Customs source files."""
from __future__ import annotations

import logging
import zipfile
from collections.abc import Callable, Iterator

import pandas as pd

log = logging.getLogger(__name__)


def open_csv(filename: str, dtypes: dict, clean: Callable | None = None) -> pd.DataFrame:
    data = pd.read_csv(filename, dtype=dtypes)
    return clean(data) if clean else data


def open_zip(filename: str, dtypes: dict, clean: Callable | None = None) -> pd.DataFrame:
    pieces = []
    with zipfile.ZipFile(filename) as archive:
        for member in archive.namelist():
            if member.endswith(".csv"):
                with archive.open(member) as source:
                    data = pd.read_csv(source, dtype=dtypes)
                    pieces.append(clean(data) if clean else data)
    if not pieces:
        raise ValueError("ZIP archive contains no CSV files.")
    return pd.concat(pieces, axis=0)


def stream_raw_chunks(filename: str, dtypes: dict, chunk_size: int) -> Iterator[pd.DataFrame]:
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename) as archive:
            for member in archive.namelist():
                if not member.endswith(".csv"):
                    continue
                with archive.open(member) as source:
                    total_rows = 0
                    for index, chunk in enumerate(pd.read_csv(source, dtype=dtypes, chunksize=chunk_size), start=1):
                        total_rows += len(chunk)
                        log.info("Loaded chunk %s from %s with %s rows (total %s).",
                                 index, member, len(chunk), total_rows)
                        yield chunk
    else:
        total_rows = 0
        for index, chunk in enumerate(pd.read_csv(filename, dtype=dtypes, chunksize=chunk_size), start=1):
            total_rows += len(chunk)
            log.info("Loaded chunk %s from %s with %s rows (total %s).",
                     index, filename, len(chunk), total_rows)
            yield chunk
