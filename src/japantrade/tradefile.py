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


class TradeFile():
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
                 chunk_size=50000, persist_path=None, persist_format=None):
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
        """
        if source is None:
            raise ValueError("You must specify the source file. Use raw=False\
                             if the file is already in normal form.")

        self.chunk_size = chunk_size
        self.persist_path = persist_path
        self.persist_format = persist_format
        self.latest_timings = []

        if base_file is not None:
            self.data, self.kind = self._openNormalFile(base_file, kind)
            self.data = self._acquireNewData(new_file=source, kind=kind)
        elif base_df is not None:
            self.data, self.kind = base_df, kind
            self.data = self._acquireNewData(new_file=source, kind=kind)
        elif raw:
            log.warning("Warning: this operation might take more than one \
 minute if the file spans over more years.")
            self.data, self.kind = self._dfFromRaw(
                source, kind, chunk_size=self.chunk_size)
        else:
            self.data, self.kind = self._openNormalFile(source, kind)


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

    def _dfFromRaw(self, file, kind, chunk_size=50000):
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
        processed_chunks = []
        inferred_kind = None
        self.latest_timings = []
        for idx, chunk in enumerate(raw_stream):
            if inferred_kind is None:
                inferred_kind = self._infer_kind(chunk)
                self._validate_raw_schema(chunk.columns, inferred_kind)
            else:
                self._validate_raw_schema(chunk.columns, inferred_kind)

            month_melted, month_timings = self._meltMonths(chunk, inferred_kind)
            normalized, unit_timings = self._meltUnits(month_melted, inferred_kind)
            reduced = self._reduce_rows(normalized)
            reduced['kind'] = inferred_kind
            self._persist_chunk(reduced, idx)
            processed_chunks.append(reduced)
            self.latest_timings.append({
                'chunk_index': idx,
                'melt_months': month_timings,
                'melt_units': unit_timings
            })

        if not processed_chunks:
            raise ValueError("No data was loaded from the provided file.")

        combined = pd.concat(processed_chunks, axis=0, ignore_index=True)
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
        and streams each file in chunks, yielding cleaned DataFrames one by one.
        If the file is a single csv, it is streamed in chunks as well.

        """
        trade_types = {'Year': 'string', 'HS': 'category',
                       'Country': 'category', 'Commodity': 'category',
                       'Unit1': 'category', 'Unit2': 'category',
                       'Unit': 'category'}

        print("Loading the file in streaming mode...")
        chunksize = chunk_size or self.chunk_size

        for chunk in self._stream_raw_chunks(filename, trade_types, chunksize):
            cleaned = self._cleanDataFile(chunk)
            yield cleaned

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
        df = df.drop(columns=["Exp or Imp"])
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
        if len(df.columns) < 20:
            log.error("meltMonths: the dataframe provided does not have \
monthly columns!")
            log.error("meltMonths: input dataframe has no monthly columns.")
            log.debug(f"columns of the input dataframe: {df.columns}")
            return df, {}

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

        melted = df

        value_cast_start = time()

        val_index = melted[melted.type == 'Value'].index
        if kind == 'HS':
            qty1_index = melted[melted.type == 'Quantity1'].index
            qty2_index = melted[melted.type == 'Quantity2'].index
            melted.loc[qty1_index, 'unit'] = melted.loc[qty1_index, 'Unit1']
            melted.loc[qty2_index, 'unit'] = melted.loc[qty2_index, 'Unit2']
        else:
            qty_index = melted[melted.type == 'Quantity'].index
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

    def _acquireNewData(self, new_file=None, new_df=None, kind='infer'):
        """
        Add extra data from file (csv, zip) to the normalized trade dataframe.

        Parameters
        -------
        new_file : string
            a file with the new, raw data

        new_df : pd.DataFrame
            an existing in-memory dataframe with new, raw data

        Returns
        -------
        pd.DataFrame
            a dataframe with the new data merged with the existing data.

        """
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
            # merge to the existing dataframe
            updated_df = pd.concat([self.data, new_data], ignore_index=True)\
                .drop_duplicates()
        elif new_df is not None:
            # merge to the existing file
            updated_df = pd.concat([self.data, new_df], ignore_index=True)\
                .drop_duplicates()
        else:
            # no file or DataFrame to merge with
            return self.data
        return updated_df

    def save_to_file(self, path="./data/", is_zip=False):
        """
        Save TradeFile.data onto file.

        Parameters
        ----------
        is_zip : TYPE, optional
            whether to compress or not. The default is False.

        Returns
        -------
        None.

        """
        ordered_timerange = self.data['date'].sort_values()
        first_date = ordered_timerange.iloc[0]
        last_date = ordered_timerange.iloc[-1]
        kind = self.kind
        # create the complete path. Find a name if not provided.
        if path[-4:] == '.csv':
            is_zip = False
            complete_path = path
            path = complete_path.rpartition()('/')[1]
        elif path[-4:] == '.zip':
            is_zip = True
            complete_path = path
            path = complete_path.rpartition()('/')[1]
        else:
            filename = '_'.join([kind, first_date, last_date])
            extension = '.zip' if is_zip else '.csv'
            # works even for mistakes in the file name
            # e.g. if the path in input is "/data/2020.xls"
            # (i.e. not a csv or zip extension)
            # it becomes "/data/2020.xls/newfilename.csv".
            if path[-1] != '/':
                path += '/'
            complete_path = "".join([path, filename, extension])

        if not os.path.exists(path):
            os.mkdir(path)
        self.data.to_csv(complete_path, index=False)
        log.info(f"TradeFile.save_to_file: saved data to {complete_path}.")
