"""Japan Customs trade file processor.

This module provides functions to handle raw file data from the Japanese \
Customs, transforming raw data into long form, normalized data that can be \
easily used to create custom reports and predictions in Pandas or other data \
processing tools.

The tool accepts files in .csv form (the native format of the data provided \
by the Japan Customs website), possibly archived into a single .zip file.
The output will be a Pandas DataFrame or a .csv file.

This module contains a single class, TradeFile.
"""
# coding: utf-8
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, Tuple
import pandas as pd
import zipfile as zfile
from time import time

MONTH_DICT = {'Jan': "01", 'Feb': "02", 'Mar': "03", 'Apr': "04",
              'May': "05", 'Jun': "06", 'Jul': "07", 'Aug': "08",
              'Sep': "09", 'Oct': "10", 'Nov': "11", 'Dec': "12"}

# logging settings for file and console
# thanks to @Escualo, https://stackoverflow.com/a/9321890
logging.basicConfig(level=logging.DEBUG,
                    filename='develop-logging.log',
                    format='%(asctime)s %(levelname)s:%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
log = logging.getLogger("tradefile")


@dataclass
class NormalizationConfig:
    """Configuration for the trade file normalization pipeline."""

    chunk_size: int = 50000
    use_tqdm: bool = False
    parallel_chunks: bool = False
    max_workers: Optional[int] = None
    output_formats: Tuple[str, ...] = ("csv", "parquet")
    include_descriptions: bool = True
    convert_to_base_units: bool = False
    keep_units: Optional[Tuple[str, ...]] = None
    exclude_units: Optional[Tuple[str, ...]] = None
    warn_on_unknown: bool = True
    lookup_paths: Optional[dict] = None

    def __post_init__(self):
        self.output_formats = tuple(fmt.lower() for fmt in self.output_formats)
        if self.keep_units is not None:
            self.keep_units = tuple(unit.upper() for unit in self.keep_units)
        if self.exclude_units is not None:
            self.exclude_units = tuple(unit.upper() for unit in self.exclude_units)
        if self.lookup_paths is not None:
            self.lookup_paths = {key.lower(): Path(value) for key, value in self.lookup_paths.items()}


@dataclass
class NormalizationContext:
    """Context passed to each normalization stage."""

    kind: str
    chunk_index: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class NormalizationPipeline:
    """Composable pipeline to normalize raw trade data chunks.

    The pipeline wires together the canonical normalization steps (clean,
    melt months, melt units, reduce rows) while allowing callers to inject
    additional steps via :meth:`add_step`.
    """

    def __init__(self, processor: "TradeFile", config: NormalizationConfig, hooks: Optional[List[Tuple[str, Callable]]] = None):
        self.processor = processor
        self.config = config
        self.logger = logging.getLogger("tradefile.pipeline")
        self._custom_steps: List[dict] = []
        self._tqdm = getattr(processor, "_tqdm", None)
        if hooks:
            for name, func in hooks:
                self.add_step(name, func)

    def add_step(self, name: str, func: Callable, position: Optional[int] = None,
                 before: Optional[str] = None, after: Optional[str] = None):
        """Register a custom step.

        Parameters
        ----------
        name : str
            Display name for the step.
        func : Callable
            Callable with signature ``func(df, context) -> (df, metrics_dict)``.
        position : int, optional
            Absolute index to insert the step in the pipeline order.
        before : str, optional
            Insert before the named step.
        after : str, optional
            Insert after the named step.
        """
        self._custom_steps.append({
            "name": name,
            "func": func,
            "position": position,
            "before": before,
            "after": after
        })

    def _default_steps(self) -> List[Tuple[str, Callable]]:
        def clean(df, context):
            cleaned = self.processor._cleanDataFile(df)
            self.processor._validate_raw_schema(cleaned.columns, context.kind)
            return cleaned, {}

        def melt_months(df, context):
            return self.processor._meltMonths(df, context.kind)

        def melt_units(df, context):
            return self.processor._meltUnits(df, context.kind)

        def normalize_units(df, context):
            return self.processor._normalize_units(df)

        def reduce_rows(df, context):
            return self.processor._reduce_rows(df), {}

        def enrich(df, context):
            return self.processor._enrich_with_lookups(df, context.kind)

        return [
            ("clean", clean),
            ("melt_months", melt_months),
            ("melt_units", melt_units),
            ("normalize_units", normalize_units),
            ("reduce", reduce_rows),
            ("enrich", enrich),
        ]

    def _compose_steps(self) -> List[Tuple[str, Callable]]:
        steps = list(self._default_steps())
        for custom in self._custom_steps:
            insertion = (custom["name"], custom["func"])
            if custom.get("position") is not None:
                steps.insert(custom["position"], insertion)
                continue
            if custom.get("before"):
                try:
                    idx = next(i for i, (name, _) in enumerate(steps) if name == custom["before"])
                except StopIteration:
                    idx = len(steps)
                steps.insert(idx, insertion)
                continue
            if custom.get("after"):
                try:
                    idx = next(i for i, (name, _) in enumerate(steps) if name == custom["after"])
                    steps.insert(idx + 1, insertion)
                except StopIteration:
                    steps.append(insertion)
                continue
            steps.append(insertion)
        return steps

    def _step_iterator(self, steps: List[Tuple[str, Callable]]):
        if self._tqdm:
            return self._tqdm(steps, desc="Normalizing chunk", unit="step")
        return steps

    def run(self, df: pd.DataFrame, kind: str, chunk_index: Optional[int] = None):
        context = NormalizationContext(kind=kind, chunk_index=chunk_index)
        composed_steps = self._compose_steps()
        metrics: List[dict] = []
        for name, func in self._step_iterator(composed_steps):
            before_rows = len(df)
            start = time()
            result = func(df, context)
            if isinstance(result, tuple) and len(result) == 2:
                df, step_metrics = result
            else:
                df, step_metrics = result, {}
            duration = time() - start
            step_metrics = step_metrics or {}
            step_metrics.update({
                "duration": duration,
                "before_rows": before_rows,
                "after_rows": len(df)
            })
            metrics.append({"step": name, **step_metrics})
            self.logger.info(
                "Pipeline step '%s' completed for chunk %s | rows %s -> %s | %.2fs",
                name,
                chunk_index,
                before_rows,
                len(df),
                duration
            )
        return df, metrics


class TradeFile():
    PRIMARY_KEY = ['kind', 'country', 'code', 'date', 'unit']
    """
    Trade data processing tool for the Japanese Customs database.

    Attributes
    ----------
    data : pandas DataFrame
        Trade data in normalized form.

    direction : string
        'import' (towards Japan) or 'export'. Default is 'import'.

    kind : string
        'HS', 'PC' or 'infer'. Default is 'infer'.
        Whether the data are categorized by HS code or PC code.
        If 'infer' is selected, TradeFile will attempt to infer the kind from \
the data.

    Example
    -------
    tool = TradeFile("PC_Trade_2019_2020.zip")
    tool.data[tool.data['date'] == "2020-07"]['measure'].mean()
    tool.data.to_csv("2019-2020jul_PC_code_all_countries.csv", index=False)

    """

    def __init__(self, source=None, raw=True, direction='import',
                 kind='infer', base_file=None, base_df=None,
                 chunk_size=50000, persist_path=None, persist_format=None,
                 normalization_config: Optional[NormalizationConfig] = None,
                 use_tqdm: bool = False, parallel_chunks: bool = False,
                 max_workers: Optional[int] = None, output_formats: Optional[Iterable[str]] = None):
        """
        Initialize a dataframe with a source trade file.

        Parameters
        ----------
        source : string
            the path to the source file.

        raw : boolean

        direction : string

        kind : string

        base_file : string
            (optional) path to an existing csv the source will be merged to

        base_df : string
            (optional) an existing DataFrame the source will be merged to
        chunk_size : int
            number of rows to load per chunk when streaming raw data
        persist_path : str
            optional directory where intermediate cleaned chunks will be saved
        persist_format : str
            optional format for persistence. Supported: 'parquet', 'feather'.
            Only used when persist_path is provided.
        normalization_config : NormalizationConfig, optional
            Pipeline settings. If omitted, one is created from other params.
        use_tqdm : bool
            Enable tqdm progress bars when available.
        parallel_chunks : bool
            Process normalization steps for chunks in parallel when safe.
        max_workers : int, optional
            Maximum workers for parallel chunk processing.
        output_formats : Iterable[str], optional
            Allowed output formats for saving normalized data.
        """
        if source is None:
            raise ValueError("You must specify the source file. Use raw=False\
                             if the file is already in normal form.")

        base_config = normalization_config or NormalizationConfig(
            chunk_size=chunk_size,
            use_tqdm=use_tqdm,
            parallel_chunks=parallel_chunks,
            max_workers=max_workers,
            output_formats=tuple(output_formats) if output_formats else ("csv", "parquet")
        )
        self.normalization_config = base_config
        self.chunk_size = self.normalization_config.chunk_size
        self.persist_path = persist_path
        self.persist_format = persist_format
        self.latest_timings = []
        self.kind = kind
        self._tqdm = self._load_tqdm() if self.normalization_config.use_tqdm else None
        self._lookup_cache: dict = {}

        if base_file is not None:
            self.data, self.kind = self._openNormalFile(base_file, kind)
            self.data = self._ensure_kind_column(self.data, self.kind)
            self.data = self._acquireNewData(new_file=source, kind=kind)
        elif base_df is not None:
            self.data, self.kind = self._ensure_kind_column(base_df, kind), kind
            self.data = self._acquireNewData(new_file=source, kind=kind)
        elif raw:
            log.warning("Warning: this operation might take more than one \
 minute if the file spans over more years.")
            self.data, self.kind = self._dfFromRaw(
                source, kind, chunk_size=self.chunk_size)
            self.data = self._ensure_kind_column(self.data, self.kind)
        else:
            self.data, self.kind = self._openNormalFile(source, kind)
            self.data = self._ensure_kind_column(self.data, self.kind)

    def _ensure_kind_column(self, df, kind, update_attr: bool = True):
        """Ensure the dataframe contains a consistent 'kind' column."""
        inferred_kind = kind
        if 'kind' not in df.columns:
            if kind == 'infer':
                raise ValueError(
                    "Cannot infer kind without a 'kind' column. "
                    "Please specify kind='HS' or kind='PC'."
                )
            inferred_kind = kind
            df = df.copy()
            df['kind'] = inferred_kind
        else:
            if kind == 'infer':
                inferred_kind = self._infer_kind(df, raw=False)
            existing_kinds = set(df['kind'].astype(str).unique())
            if len(existing_kinds) > 1:
                raise ValueError(
                    "Multiple kinds detected in provided dataframe: "
                    f"{existing_kinds}"
                )
            if inferred_kind != 'infer' and inferred_kind not in existing_kinds:
                raise ValueError(
                    f"Inconsistent kind detected. Expected {inferred_kind}, "
                    f"found {existing_kinds}"
                )
            inferred_kind = existing_kinds.pop()
        if update_attr:
            self.kind = inferred_kind
        return df

    def _load_tqdm(self):
        try:
            from tqdm import tqdm  # type: ignore
            return tqdm
        except ImportError:
            log.warning("tqdm requested but not installed. Proceeding without progress bars.")
            return None

    def _primary_key(self):
        return self.PRIMARY_KEY

    def _deduplicate_by_key(self, df):
        missing_key_cols = [col for col in self._primary_key() if col not in df.columns]
        if missing_key_cols:
            raise ValueError(
                f"Cannot deduplicate dataframe. Missing primary key columns: {missing_key_cols}"
            )
        before = len(df)
        deduped = df.drop_duplicates(subset=self._primary_key())
        after = len(deduped)
        if after != before:
            log.info(
                "Deduplicated dataframe on primary key.",
                extra={"deduplication": {"before": before, "after": after}}
            )
        return deduped

    def _snapshot_state(self, df):
        key_columns = [col for col in self._primary_key() if col in df.columns]
        ordered = df
        if key_columns:
            ordered = df.sort_values(by=key_columns)
        ordered = ordered.reset_index(drop=True)
        checksum = pd.util.hash_pandas_object(
            ordered.fillna(""), index=False
        ).sum()
        return {"rows": len(df), "checksum": int(checksum)}

    def _log_validation(self, stage, before_snapshot, after_snapshot, extra=None):
        payload = {
            "stage": stage,
            "before": before_snapshot,
            "after": after_snapshot
        }
        if extra:
            payload["meta"] = extra
        log.info("Data validation", extra={"validation": payload})

    def _normalize_compression(self, compression):
        if compression in (None, 'zip', 'gzip', 'bz2'):
            return compression
        if compression in ('gz',):
            return 'gzip'
        if compression in ('bz',):
            return 'bz2'
        raise ValueError("Unsupported compression. Use 'zip', 'gzip', or 'bz2'.")

    def _compression_from_suffix(self, suffix):
        mapping = {
            '.gz': 'gzip',
            '.gzip': 'gzip',
            '.zip': 'zip',
            '.bz2': 'bz2'
        }
        return mapping.get(suffix)

    def _extension_for(self, fmt, compression):
        fmt = fmt.lower()
        normalized_compression = self._normalize_compression(compression)
        compression_extension = {
            'zip': 'zip',
            'gzip': 'gz',
            'bz2': 'bz2',
            None: None
        }
        if fmt == 'csv':
            if normalized_compression in ('zip', 'gzip', 'bz2'):
                return f".csv.{compression_extension[normalized_compression]}"
            return ".csv"
        if fmt == 'parquet':
            return ".parquet"
        raise ValueError("Unsupported format. Use 'csv' or 'parquet'.")

    def _default_filename(self, fmt, compression):
        ordered_timerange = self.data['date'].sort_values()
        first_date = ordered_timerange.iloc[0]
        last_date = ordered_timerange.iloc[-1]
        extension = self._extension_for(fmt, compression)
        return f"{self.kind}_{first_date}_{last_date}{extension}"

    def _build_output_path(self, path, filename, fmt, compression):
        base_path = Path(path)
        if base_path.suffix and filename is None:
            target_dir = base_path.parent
            filename = base_path.name
        elif base_path.suffix:
            target_dir = base_path.parent
        else:
            target_dir = base_path

        resolved_fmt = fmt.lower() if fmt else None
        resolved_compression = self._normalize_compression(compression)
        if filename:
            target_path = target_dir / filename
            suffixes = target_path.suffixes
            if suffixes:
                detected_compression = self._compression_from_suffix(suffixes[-1])
                if detected_compression:
                    resolved_compression = resolved_compression or detected_compression
                    if len(suffixes) > 1:
                        resolved_fmt = resolved_fmt or suffixes[-2].lstrip('.')
                else:
                    resolved_fmt = resolved_fmt or suffixes[-1].lstrip('.')
            else:
                resolved_fmt = resolved_fmt or 'csv'
                target_path = target_path.with_suffix(self._extension_for(resolved_fmt, resolved_compression))
        else:
            resolved_fmt = resolved_fmt or 'csv'
            target_path = target_dir / self._default_filename(resolved_fmt, resolved_compression)

        if not target_path.suffix:
            resolved_fmt = resolved_fmt or 'csv'
            target_path = target_path.with_suffix(self._extension_for(resolved_fmt, resolved_compression))

        return target_path, resolved_fmt, resolved_compression

    def _load_saved_file(self, target_path, fmt, compression):
        if fmt == 'parquet':
            return pd.read_parquet(target_path)
        if fmt == 'csv':
            return pd.read_csv(target_path, compression=compression)
        raise ValueError("Unsupported format for loading. Use 'csv' or 'parquet'.")


    def _infer_kind(self, df, raw=True):
        """
        Infer the kind of data (HS or PC) from the columns.

        Parameters
        ----------
        df : pd.DataFrame
            a DataFrame to be inspected.

        Returns
        -------
        string
            'HS' or 'PC'

        """
        if raw is True:
            # kind can be inferred from the columns names
            if 'HS' in df.columns:
                return 'HS'
            elif 'Commodity' in df.columns:
                return 'PC'
            else:
                raise ValueError("Cannot infer the type of data: HS or PC?")
        else:
            # we need to infer the kind by examining the kind column
            kinds = df['kind'].unique()
            if 'PC' in kinds and 'HS' in kinds:
                return 'mixed'
            elif 'PC' in kinds:
                return 'PC'
            elif 'HS' in kinds:
                return 'HS'
            else:
                raise ValueError(f"_infer_kind: the 'kind' column is dirty.\n\
                                 Values in the column: {kinds}")

    def _dfFromRaw(self, file, kind, chunk_size=None):
        """
        Normalize raw trade data into a dataframe.

        Parameters
        ----------
        file : string
            the path of the raw trade file.
        chunk_size : int
            number of rows to process per chunk

        Returns
        ----------
        normalized : pd.DataFrame
            a Pandas Dataframe with normalized columns.
        """
        print(f"Opening {file}...")
        raw_stream = self._openRawData(file, chunk_size=chunk_size)
        raw_iter = iter(raw_stream)
        try:
            first_chunk = next(raw_iter)
        except StopIteration:
            raise ValueError("No data was loaded from the provided file.")

        inferred_kind = kind if kind != 'infer' else self._infer_kind(first_chunk)

        pipeline = NormalizationPipeline(self, self.normalization_config)
        processed_chunks = []
        self.latest_timings = []

        def process(idx, chunk):
            reduced, metrics = pipeline.run(chunk, inferred_kind, chunk_index=idx)
            reduced['kind'] = inferred_kind
            self._persist_chunk(reduced, idx)
            return idx, reduced, metrics

        # process first chunk synchronously to establish schema confidence
        first_idx, first_reduced, first_metrics = process(0, first_chunk)
        processed_chunks.append((first_idx, first_reduced))
        self.latest_timings.append({'chunk_index': first_idx, 'metrics': first_metrics})
        chunk_count = 1

        if self.normalization_config.parallel_chunks:
            futures = []
            with ThreadPoolExecutor(max_workers=self.normalization_config.max_workers) as executor:
                for idx, chunk in enumerate(raw_iter, start=1):
                    futures.append(executor.submit(process, idx, chunk))
                for future in as_completed(futures):
                    idx, reduced, metrics = future.result()
                    processed_chunks.append((idx, reduced))
                    self.latest_timings.append({'chunk_index': idx, 'metrics': metrics})
                    log.info("Completed chunk %s in parallel with %s rows", idx, len(reduced))
                    chunk_count += 1
        else:
            chunk_iterator = enumerate(raw_iter, start=1)
            if self._tqdm:
                chunk_iterator = self._tqdm(chunk_iterator, desc="Processing chunks", unit="chunk")
            for idx, chunk in chunk_iterator:
                _, reduced, metrics = process(idx, chunk)
                processed_chunks.append((idx, reduced))
                self.latest_timings.append({'chunk_index': idx, 'metrics': metrics})
                log.info("Completed chunk %s with %s rows", idx, len(reduced))
                chunk_count += 1

        processed_chunks = sorted(processed_chunks, key=lambda t: t[0])
        combined = pd.concat([df for _, df in processed_chunks], axis=0, ignore_index=True)
        combined = self._deduplicate_by_key(combined)
        log.info("Combined %s chunks into %s rows after deduplication", chunk_count, len(combined))
        return combined, inferred_kind

    def _opener(self):
        # identifying the right function to call the right function
        # in order to open the file according to extension.
        return {'zip': self._openzip, 'csv': self._opencsv}

    def _openzip(self, filename, trade_types, raw=True):
        pieces = []
        with zfile.ZipFile(filename) as z:
            for f in z.namelist():
                if f[-4:] == ".csv":
                    with z.open(f) as piece:
                        df = pd.read_csv(piece, dtype=trade_types)
                        if raw:
                            df = self._cleanDataFile(df)
                        pieces.append(df)
            merged = pd.concat(pieces, axis=0)
            return merged

    def _opencsv(self, filename, trade_types, raw=True):
        df = pd.read_csv(filename, dtype=trade_types)
        if raw:
            df = self._cleanDataFile(df)
        return df

    def _validate_raw_schema(self, columns, kind):
        base_columns = {'Year', 'Country'}
        if kind == 'HS':
            base_columns.update({'HS', 'Unit1', 'Unit2'})
            expected_types = ['Quantity1', 'Quantity2', 'Value']
        else:
            base_columns.update({'Commodity', 'Unit'})
            expected_types = ['Quantity', 'Value']

        missing_base = base_columns.difference(columns)
        if missing_base:
            raise ValueError(f"Missing required column(s): {sorted(missing_base)}")

        missing_months = []
        for measure_type in expected_types:
            for month_name in MONTH_DICT.keys():
                column_name = f"{measure_type}-{month_name}"
                if column_name not in columns:
                    missing_months.append(column_name)

        if missing_months:
            raise ValueError(
                "The file is missing expected monthly measure columns. "
                f"Missing: {missing_months}"
            )

        return True

    def _stream_raw_chunks(self, filename, trade_types, chunk_size):
        if filename.endswith('.zip'):
            with zfile.ZipFile(filename) as z:
                for f in z.namelist():
                    if f.endswith(".csv"):
                        with z.open(f) as piece:
                            total_rows = 0
                            for i, chunk in enumerate(pd.read_csv(
                                    piece, dtype=trade_types,
                                    chunksize=chunk_size)):
                                total_rows += len(chunk)
                                log.info(f"Loaded chunk {i+1} from {f} with "
                                         f"{len(chunk)} rows (total {total_rows}).")
                                yield chunk
        else:
            total_rows = 0
            for i, chunk in enumerate(pd.read_csv(
                    filename, dtype=trade_types, chunksize=chunk_size)):
                total_rows += len(chunk)
                log.info(f"Loaded chunk {i+1} from {filename} with "
                         f"{len(chunk)} rows (total {total_rows}).")
                yield chunk

    def _persist_chunk(self, df, chunk_index):
        if not self.persist_path:
            return

        os.makedirs(self.persist_path, exist_ok=True)
        persist_format = (self.persist_format or 'parquet').lower()

        if persist_format == 'parquet':
            extension = 'parquet'
            df.to_parquet(
                os.path.join(
                    self.persist_path, f"clean_chunk_{chunk_index}.{extension}"
                ),
                index=False
            )
        elif persist_format == 'feather':
            extension = 'feather'
            df.reset_index(drop=True).to_feather(
                os.path.join(
                    self.persist_path, f"clean_chunk_{chunk_index}.{extension}"
                )
            )
        else:
            raise ValueError("Unsupported persist_format. Use 'parquet' or 'feather'.")

    def _openRawData(self, filename, chunk_size=None):
        """
        Open a zip or a csv file containing raw trade data from customs.go.jp.

        The following columns are expected in the file:
        - "Year", containing a 4 digit year number
        - "Country, a 3-digit code for each country
        - "HS" or "Commodity", a code used to identify the category of the\
 imported/exported product.
        Also, we expect 36 columns created as a couple "Type-Mon" where "type"
        is one among
        "Quantity1", "Quantity2" or "Value", while "Mon" represents the first
        three letters
        (capitalized) of a specific month, e.g. "Jan", "Aug" etc.
        In case the file is a zip with multiple csv files, the function opens
        and streams each file in chunks. Cleaning and normalization are applied
        later by the pipeline, so this generator yields raw chunks.
        If the file is a single csv, it is streamed in chunks as well.

        """
        trade_types = {'Year': 'string', 'HS': 'category',
                       'Country': 'category', 'Commodity': 'category',
                       'Unit1': 'category', 'Unit2': 'category',
                       'Unit': 'category'}

        print("Loading the file in streaming mode...")
        chunksize = chunk_size or self.chunk_size

        for chunk in self._stream_raw_chunks(filename, trade_types, chunksize):
            yield chunk

    def _cleanDataFile(self, df):
        """Clean the dataframe acquired from the raw data."""
        columns = df.columns.to_list()
        # check on the columns
        if ('Unit2' not in columns) and 'Unit' not in columns:
            raise ValueError("This is not a standard raw file.\
No data were acquired.")
        # some files have "Aplil" instead of "April"...
        if "Quantity1-Apl" in columns:
            df.rename(
                columns={"Quantity1-Apl": "Quantity1-Apr",
                         "Quantity2-Apl": "Quantity2-Apr",
                         "Value-Apl": "Value-Apr"},
                inplace=True)
        elif "Quantity-Apl" in columns:
            df.rename(columns={
                "Quantity-Apl": "Quantity-Apr",
                "Value-Apl": "Value-Apr"
            },
                inplace=True)
        # not interested in yearly values - can recover them later anyway
        df = df.drop(columns=[c for c in df.columns.tolist() if "-Year" in c])
        # data are either all import or all export
        log.debug(f"The columns before dropping the exp or imp column are \
{df.columns}")
        df = df.drop(columns=["Exp or Imp"], errors="ignore")
        # remove the "'" from the code column
        if 'HS' in df.columns:
            df['HS'] = df['HS'].str.strip("' ")
        elif 'Commodity' in df.columns:
            df['Commodity'] = df['Commodity'].str.strip("' ")

        return df

    def _openNormalFile(self, filename, kind='infer'):
        """
        Open a zip or a csv file containing normalized trade data.

        The following columns are expected in the file:
        - "Country", a 3-digit code for each country
        - "HS", a code used to identify the category of the imported/exported
          product.
        - "unit": Kgs, Tons, Number of, 000 JPYs etc.
        - "date": a string (not a date time) of the form AAAA-MM-01
        - "value": integer with the value (in "unit") for the tuple (country,
           HS, date)

        In case the file is a zip with multiple csv files, the function opens
        and merges
        the files into a single DataFrame object, which the function returns.
        If the file is a single csv, return the DataFrame built from it.

        Parameters
        ----------
        filename : string
            the name of the file containing an already processed trade file.

        Returns
        ----------
        pd.DataFrame
            a Pandas DataFrame with the trade data already normalized
        """
        trade_types = {'HS': 'category', 'Country': 'category',
                       'code': 'category',
                       'date': object, 'unit': 'category', 'value': 'Int64'}
        df = self._opener()[filename[-3:]](filename, trade_types, raw=False)
        if kind == 'infer':
            kind = self._infer_kind(df, raw=False)
        return df, kind

    def _meltMonths(self, df, kind):
        """
        Melt all the "unit-month" columns into a long format.

        - "date", a Y-M-D date format with the last day of the month
        - "type", with one of the three possible values "Quantity1",
          "Quantity2", "Value"
        - "measure", with the value associated to the type
        """
        # a very lousy check but a check nontheless...
        print("Unpivoting the monthly columns. This might take a minute...")

        if kind == 'HS':
            ids = ['Year', 'HS', 'Country', 'Unit1', 'Unit2']
        else:
            ids = ['Year', 'Commodity', 'Country', 'Unit']
        start_melt = time()
        melted = pd.melt(df, id_vars=ids, value_name='measure')
        end_melt = time()
        melt_time = end_melt-start_melt

        # splitting the month and the unit in two different columns
        # note: the StringIO method is the fastest around for now (performance
        # x3 vs list comprehension)
        # https://stackoverflow.com/posts/30113715/revisions
        start_split = time()
        split_columns = melted['variable'].str.split('-', n=1, expand=True)
        if split_columns.shape[1] != 2 or split_columns.isnull().any().any():
            raise ValueError("Found malformed monthly column identifiers during split.")
        melted[['type', 'month']] = split_columns
        melted = melted.drop(columns=['variable'])
        end_split = time()
        split_time = end_split - start_split

        # we can start reducing the size of the DF by erasing the rows
        # that have Quantity 1 or Quantity 2 without a measure.
        log.debug(f"columns of the df before melting months: {melted.columns}")
        start_clean = time()
        if kind == 'HS':
            has_qty1 = (melted["type"] != "Quantity1") |\
                (melted["Unit1"] != "  ")
            has_qty2 = (melted["type"] != "Quantity2") |\
                (melted["Unit2"] != "  ")
            melted = melted[has_qty1 & has_qty2]
        else:
            has_qty = (melted["type"] != "Quantity") | (melted["Unit"] != "  ")
            melted = melted[has_qty]
        end_clean = time()
        clean_time = end_clean - start_clean

        # now, merging creating a datetime column from the year and the month
        start_date = time()
        melted['month'] = melted['month'].replace(MONTH_DICT)
        melted['date'] = melted['Year'] + "-" + melted['month'] + "-01"
        end_date = time()
        date_time = end_date - start_date

        # we can now get rid of year, day and month:
        melted = melted.drop(columns=['Year', 'month'])

        metrics = {
            'melt_time': melt_time,
            'split_time': split_time,
            'clean_time': clean_time,
            'date_merge_time': date_time
        }
        log.info(
            "Timing metrics for _meltMonths | melt: %s | split: %s | clean: %s | date_merge: %s",
            melt_time,
            split_time,
            clean_time,
            date_time
        )

        return melted, metrics

    def _meltUnits(self, df, kind):
        """
        Melt the units columns.

        This function executes an essential transformation for trade files, as
        in the original file up to three units (unit1, unit2, and the implicit
        "yen" value) are present for each row.
        We want to have a table that is  normalized  with respect to units,
        that is, a row will have a column "unit" that will contain the exact
        unit the measure is in (e.g. Kg, Ton, No. of items, 1000 yen...) and a
        column for the numeric value ("measure").
        """
        # a very lousy check but a check nontheless...
        if 'Unit2' not in df.columns and 'Unit' not in df.columns:
            log.error("meltUnits: input dataframe has no valid units.")
            log.debug(f"input dataframe columns: {df.columns}")
            return df

        log.info("Unpivoting the metrics...")

        melted = df.copy()

        value_cast_start = time()

        val_index = melted[melted["type"] == 'Value'].index
        if kind == 'HS':
            qty1_index = melted[melted["type"] == 'Quantity1'].index
            qty2_index = melted[melted["type"] == 'Quantity2'].index
            melted.loc[qty1_index, 'unit'] = melted.loc[qty1_index, 'Unit1']
            melted.loc[qty2_index, 'unit'] = melted.loc[qty2_index, 'Unit2']
        else:
            qty_index = melted[melted["type"] == 'Quantity'].index
            melted.loc[qty_index, 'unit'] = melted.loc[qty_index, 'Unit']

        melted.loc[val_index, 'unit'] = 'JPY'
        if val_index.size:
            casted_values = pd.to_numeric(
                melted.loc[val_index, 'measure'], errors='coerce'
            )
            if casted_values.isna().any():
                raise ValueError("Value rows must contain numeric data before scaling.")
            melted.loc[val_index, 'measure'] = casted_values.multiply(1000)
        value_cast_end = time()
        value_cast_time = value_cast_end - value_cast_start

        unit_assignment_time = value_cast_time
        missing_units = melted['unit'].isna() | (
            melted['unit'].astype(str).str.strip() == ''
        )
        if missing_units.any():
            sample = melted.loc[missing_units, ['type', 'date']].head().to_dict('records')
            raise ValueError(
                "Encountered rows with missing units after melting: "
                f"{sample}"
            )
        melted['unit'] = melted['unit'].astype(str).str.strip()

        # these columns have been replaced by "unit", won't be useful anymore
        rename_start = time()
        if kind == 'HS':
            melted.drop(columns=['Unit1', 'Unit2', 'type'], inplace=True)
            melted.rename(columns={'HS': 'code',
                                   'Country': 'country',
                                   'measure': 'value'}, inplace=True)
        else:
            melted.drop(columns=['Unit', 'type'], inplace=True)
            melted.rename(columns={'Commodity': 'code',
                                   'Country': 'country',
                                   'measure': 'value'}, inplace=True)
        rename_end = time()
        metrics = {
            'value_cast_time': value_cast_time,
            'rename_time': rename_end - rename_start,
            'unit_assignment_time': unit_assignment_time
        }
        log.info(
            "Timing metrics for _meltUnits | value_cast: %s | rename: %s | unit_assignment: %s",
            metrics['value_cast_time'],
            metrics['rename_time'],
            metrics['unit_assignment_time']
        )
        return melted, metrics

    def _normalize_units(self, df: pd.DataFrame):
        canonical_map = {
            "KG": "KG",
            "KGS": "KG",
            "KILOGRAM": "KG",
            "KILOGRAMS": "KG",
            "NO": "NO",
            "NUMBER": "NO",
            "TH": "TH",
            "THOUSAND": "TH",
            "JPY": "JPY",
        }
        conversion_factors = {
            "TH": ("NO", 1000),
        }

        normalized = df.copy()
        normalized["unit"] = normalized["unit"].astype(str).str.upper()
        normalized["unit"] = normalized["unit"].map(lambda u: canonical_map.get(u, u))

        conversions = 0
        if self.normalization_config.convert_to_base_units:
            for source_unit, (target_unit, factor) in conversion_factors.items():
                mask = normalized["unit"] == source_unit
                if mask.any():
                    normalized.loc[mask, "value"] = normalized.loc[mask, "value"] * factor
                    normalized.loc[mask, "unit"] = target_unit
                    conversions += int(mask.sum())

        known_units = set(canonical_map.values())
        unknown_units_mask = ~normalized["unit"].isin(known_units)
        unknown_units = normalized.loc[unknown_units_mask, "unit"].unique().tolist()
        if unknown_units and self.normalization_config.warn_on_unknown:
            log.warning("Encountered unknown unit codes: %s", unknown_units)

        filtered_out = 0
        keep_units = set(self.normalization_config.keep_units or [])
        exclude_units = set(self.normalization_config.exclude_units or [])
        if keep_units:
            keep_mask = normalized["unit"].isin(keep_units)
            filtered_out += int((~keep_mask).sum())
            normalized = normalized[keep_mask]
        if exclude_units:
            exclude_mask = normalized["unit"].isin(exclude_units)
            filtered_out += int(exclude_mask.sum())
            normalized = normalized[~exclude_mask]

        metrics = {
            "unit_conversions": conversions,
            "filtered_rows": filtered_out,
            "unknown_unit_count": int(unknown_units_mask.sum()),
        }
        return normalized, metrics

    def _reduce_rows(self, df):
        """
        Reduce the number of rows by deleting those that are zero.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to reduce.

        Returns
        reduced : pd.DataFrame
        A reduced DataFrame.

        """
        reduced = df[df['value'] != 0].dropna()
        return reduced

    def _lookup_path(self, key: str) -> Optional[Path]:
        base_dir = Path(__file__).resolve().parent
        overrides = self.normalization_config.lookup_paths or {}
        if key in overrides:
            candidate = overrides[key]
            return candidate if candidate.exists() else None
        default_map = {
            "hs": base_dir / "HScodes.csv",
            "pc": base_dir / "PCcodes.csv",
            "country": base_dir / "Countries.csv",
        }
        candidate = default_map.get(key)
        return candidate if candidate and candidate.exists() else None

    def _load_lookup(self, key: str, loader: Callable[[Path], pd.DataFrame]) -> Optional[pd.DataFrame]:
        if key in self._lookup_cache:
            return self._lookup_cache[key]
        path = self._lookup_path(key)
        if not path:
            self._lookup_cache[key] = None
            if self.normalization_config.warn_on_unknown:
                log.warning("Lookup file for '%s' not found. Skipping enrichment.", key)
            return None
        loaded = loader(path)
        self._lookup_cache[key] = loaded
        return loaded

    def _code_lookup(self, kind: str) -> Optional[pd.DataFrame]:
        key = kind.lower()

        def _loader(path: Path) -> pd.DataFrame:
            df = pd.read_csv(path, delimiter=';', dtype=str)
            df = df.rename(columns={"Code.1": "code", "Description": "code_description"})
            if "code" not in df.columns or "code_description" not in df.columns:
                return None
            df["code"] = df["code"].astype(str).str.replace(r"\s+", "", regex=True)
            df = df[["code", "code_description"]].dropna()
            return df

        return self._load_lookup(key, _loader)

    def _country_lookup(self) -> Optional[pd.DataFrame]:
        def _loader(path: Path) -> pd.DataFrame:
            df = pd.read_csv(path, dtype=str)
            rename_map = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
            df = df.rename(columns=rename_map)
            if "code" not in df.columns:
                return None
            df["code"] = df["code"].astype(str).str.zfill(3)
            name_col = "country" if "country" in df.columns else df.columns[-1]
            zone_col = "geographical_zone" if "geographical_zone" in df.columns else None
            keep_cols = ["code", name_col] + ([zone_col] if zone_col else [])
            return df[keep_cols].rename(columns={name_col: "country_name", "code": "country"})

        return self._load_lookup("country", _loader)

    def _enrich_with_lookups(self, df: pd.DataFrame, kind: str):
        if not self.normalization_config.include_descriptions:
            return df, {}

        enriched = df.copy()
        metrics = {}

        code_lookup = self._code_lookup(kind)
        if code_lookup is not None and not code_lookup.empty:
            before = len(enriched)
            enriched = enriched.merge(code_lookup, how="left", on="code")
            missing_codes = enriched["code_description"].isna().sum()
            metrics["code_enriched"] = int(before - missing_codes)
            metrics["missing_code_descriptions"] = int(missing_codes)
        elif self.normalization_config.warn_on_unknown:
            log.warning("Code lookup for kind '%s' unavailable; skipping code enrichment.", kind)

        country_lookup = self._country_lookup()
        if country_lookup is not None and not country_lookup.empty:
            before = len(enriched)
            enriched = enriched.merge(country_lookup, how="left", on="country")
            missing_countries = enriched["country_name"].isna().sum()
            metrics["country_enriched"] = int(before - missing_countries)
            metrics["missing_country_descriptions"] = int(missing_countries)
        elif self.normalization_config.warn_on_unknown:
            log.warning("Country lookup unavailable; skipping country enrichment.")

        if "code_description" in enriched:
            unknown_codes = enriched[enriched["code_description"].isna()]["code"].unique().tolist()
        else:
            unknown_codes = []
        if unknown_codes and self.normalization_config.warn_on_unknown:
            log.warning("Unknown codes encountered during enrichment: %s", unknown_codes[:5])

        return enriched, metrics

    def _acquireNewData(self, new_file=None, new_df=None, kind='infer', date_range=None):
        """
        Add extra data from file (csv, zip) to the normalized trade dataframe.

        Parameters
        -------
        new_file : string
            a file with the new, raw data

        new_df : pd.DataFrame
            an existing in-memory dataframe with new, raw data

        date_range : tuple, optional
            (start_date, end_date) inclusive window to merge incrementally.
            Rows in the existing dataframe within the window are replaced.

        Returns
        -------
        pd.DataFrame
            a dataframe with the new data merged with the existing data.

        """
        if date_range is not None and (not isinstance(date_range, (list, tuple)) or len(date_range) != 2):
            raise ValueError("date_range must be a tuple or list of two date strings (start, end).")

        base_snapshot = self._snapshot_state(self.data)
        existing_kind = getattr(self, "kind", None)
        if new_file is not None:
            # we are giving priority to files.
            # if both a DF and a file are specified,
            # only the file will be considered.
            log.info(f"Merging {new_file} to the existing database.")
            if new_df is not None:
                log.warning("TradeFile.acquireNewData: both df and csv \
files were provided as merge parameters. The dataframe will be ignored.")
            # normalize the new file
            new_data, kind = self._dfFromRaw(
                new_file, kind, chunk_size=self.chunk_size)
            new_data = self._ensure_kind_column(new_data, kind)
        elif new_df is not None:
            new_data = self._ensure_kind_column(new_df, kind, update_attr=False)
        else:
            # no file or DataFrame to merge with
            return self.data
        incoming_kind = self._infer_kind(new_data, raw=False)
        if existing_kind not in ['infer', None] and existing_kind != incoming_kind:
            raise ValueError(
                f"Inconsistent kind when merging. Existing kind: {existing_kind}, "
                f"new data kind: {incoming_kind}"
            )
        self.kind = incoming_kind

        if date_range:
            start_date, end_date = date_range
            new_data = new_data[new_data['date'].between(start_date, end_date)]
            base_subset = self.data[~self.data['date'].between(start_date, end_date)]
        else:
            base_subset = self.data

        combined = pd.concat([base_subset, new_data], ignore_index=True)
        combined = self._deduplicate_by_key(combined)
        after_snapshot = self._snapshot_state(combined)
        self._log_validation(
            "merge",
            before_snapshot=base_snapshot,
            after_snapshot=after_snapshot,
            extra={"date_range": date_range}
        )
        self.data = combined
        return combined

    def save_to_file(self, path="./data/", filename=None, fmt="csv", compression=None):
        """
        Save TradeFile.data to disk with validation.

        Parameters
        ----------
        path : str
            Directory or file path.
        filename : str, optional
            Optional file name (with or without extension).
        fmt : str, optional
            'csv' or 'parquet'. Defaults to 'csv'.
        compression : str, optional
            Compression codec. For CSV: 'zip', 'gzip', 'bz2'. For Parquet: codec string accepted by pandas.

        Returns
        -------
        Path
            The path where the file was saved.
        """
        if not hasattr(self, "normalization_config") or self.normalization_config is None:
            self.normalization_config = NormalizationConfig()
        resolved_fmt = fmt.lower() if fmt else None
        if resolved_fmt and resolved_fmt not in self.normalization_config.output_formats:
            raise ValueError(
                f"Format '{fmt}' is not allowed. "
                f"Supported formats: {self.normalization_config.output_formats}"
            )
        target_path, resolved_fmt, resolved_compression = self._build_output_path(path, filename, fmt, compression)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        before_snapshot = self._snapshot_state(self.data)
        if resolved_fmt == 'parquet':
            self.data.to_parquet(
                target_path,
                index=False,
                compression=resolved_compression
            )
        elif resolved_fmt == 'csv':
            self.data.to_csv(
                target_path,
                index=False,
                compression=resolved_compression
            )
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'parquet'.")

        saved_df = self._load_saved_file(target_path, resolved_fmt, resolved_compression)
        after_snapshot = self._snapshot_state(saved_df)
        self._log_validation(
            "save",
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            extra={
                "path": str(target_path),
                "format": resolved_fmt,
                "compression": resolved_compression
            }
        )
        log.info(f"TradeFile.save_to_file: saved data to {target_path}.")
        return target_path
