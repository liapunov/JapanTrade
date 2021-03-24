#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import zipfile as zfile
from time import time
from io import StringIO


# thanks to MarcH https://stackoverflow.com/a/14981125
# for this function to print to stderr:
def errprint(*args, **kwargs):
    """Print messages on stderr."""
    print(*args, file=sys.stderr, **kwargs)


class TradeFile():
    """
    Trade data from the Japanese Customs database.

    Attributes
    ----------
    data : pandas DataFrame
        Trade data in narrow form.

    Example
    -------
    tool = TradeFile("PC_Trade_2019_2020.zip")
    tool.data[tool.data['date'] == "2020-07"]['measure'].mean()
    tool.data.to_csv("2019-2020jul_PC_code_all_countries.csv", index=False)

    """

    def __init__(self, filename=None, raw=True, direction='import'):

        if filename is None:
            raise ValueError("You must specify the source file. Use raw=False\
                             if the file is already in normal form.")

        if raw:
            print("Warning: this operation might take more than one minute if\
                  the file spans over more years.")
            self.data = self._dfFromRaw(filename)
        else:
            self.data = self._openNormalFile(filename)

    def _openRawData(self, filename):
        """
        Open a zip or a csv file containing raw trade data from customs.go.jp.

        The following columns are expected in the file:
        - "Year", containing a 4 digit year number
        - "Country, a 3-digit code for each country
        - "HS", a code used to identify the category of the imported/exported
        product.
        Also, we expect 36 columns created as a couple "Type-Mon" where "type"
        is one among
        "Quantity1", "Quantity2" or "Value", while "Mon" represents the first
        three letters
        (capitalized) of a specific month, e.g. "Jan", "Aug" etc.
        In case the file is a zip with multiple csv files, the function opens
        and merges
        the files into a single DataFrame object, which the function returns.
        If the file is a single csv, return the DataFrame built from it.

        """
        trade_types = {'Year': 'string', 'HS': 'category',
                       'Country': 'category', 'Commodity': 'category',
                       'Unit1': 'category', 'Unit2': 'category',
                       'Unit': 'category'}

        def merge(frame1, frame2):
            return frame1.append(frame2, ignore_index=True)

        print("Loading the file...")

        pieces = []

        if filename[-4:] == ".zip":
            with zfile.ZipFile(filename) as z:
                for f in z.namelist():
                    if f[-4:] == ".csv":
                        with z.open(f) as piece:
                            df = pd.read_csv(piece, dtype=trade_types)
                            columns = df.columns.to_list()
                            if ('Unit2' not in columns)\
                               and ('Unit' not in columns):
                                errprint("This is not a standard raw file.\
                                         No data were acquired.")
                                return None
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
                            pieces.append(df)
                merged = pd.concat(pieces, axis=0)
                return merged

        elif filename[-4:] == ".csv":
            single_file = pd.read_csv(filename, dtype=trade_types)
            return single_file

    def _cleanDataFile(self, df):
        """Clean the dataframe acquired from the raw data."""
        print("Cleaning the raw file...")
        # not interested in yearly values - can recover them later anyway
        df = df.drop(columns=[c for c in df.columns.tolist() if "-Year" in c])
        # they are all either import or export data so far.
        df = df.drop(columns=["Exp or Imp"])
        # remove the "'" from the HS column
        if 'HS' in df.columns:
            df['HS'] = df['HS'].str.strip("' ")
        elif 'Commodity' in df.columns:
            df['Commodity'] = df['Commodity'].str.strip("' ")

        return df

    def _openzip(self, f, trade_types):
        with zfile.ZipFile(f) as z:
            n = z.namelist()[0]
            with z.open(n) as file:
                normal = pd.read_csv(file, dtype=trade_types)
                return normal

    def _opencsv(self, filename, trade_types):
        data = pd.read_csv(filename, dtype=trade_types)
        return data

    def _opener(self):
        # identifying the right function to call the right function
        # in order to open the file according to extension.
        return {'zip': self._openzip, 'csv': self._opencsv}

    def _openNormalFile(self, filename):
        """
        Open a zip or a csv file containing normalized trade data.

        The following columns are expected in the file:
        - "Country", a 3-digit code for each country
        - "HS", a code used to identify the category of the imported/exported
          product.
        - "unit": Kgs, Tons, Number of, 000 JPYs etc.
        - "date": a string (not a date time) of the form AAAA-MM
        - "measure": integer with the value (in "unit") for the tuple (country,
           HS, date)

        In case the file is a zip with multiple csv files, the function opens
        and merges
        the files into a single DataFrame object, which the function returns.
        If the file is a single csv, return the DataFrame built from it.
        """
        trade_types = {'HS': 'category', 'Country': 'category',
                       'Commodity': 'category',
                       'date': object, 'unit': 'category', 'measure': 'Int64'}
        return self._opener[filename.split(".")[-1]](filename, trade_types)

    def meltMonths(self, df):
        """
        Melt all the "unit-month" columns into a narrow form.

        - "date", a Y-M-D date format with the last day of the month
        - "type", with one of the three possible values "Quantity1",
          "Quantity2", "Value"
        - "measure", with the value associated to the type
        """
        # this dictionary will be useful to convert the months from words
        # to digits
        month_dict = {'Jan': "-01", 'Feb': "-02", 'Mar': "-03", 'Apr': "-04",
                      'May': "-05", 'Jun': "-06", 'Jul': "-07", 'Aug': "-08",
                      'Sep': "-09", 'Oct': "-10", 'Nov': "-11", 'Dec': "-12"}

        # a very lousy check but a check nontheless...
        if len(df.columns) < 20:
            errprint("meltMonths: the dataframe provided does not have\
                     monthly columns!")
            return df

        print("Unpivoting the monthly columns. This might take a minute...")

        if "Unit1" in df.columns:
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
        melted[['type', 'month']] = pd.read_table(
            StringIO(melted['variable'].to_csv(None, index=None, header=None)
                     ), sep='-', header=None)
        melted = melted.drop(columns=['variable'])
        end_split = time()
        split_time = end_split - start_split

        # we can start reducing the size of the DF by erasing the rows
        # that have Quantity 1
        # or Quantity 2 without a measure.
        start_clean = time()
        if "Quantity1" in melted.columns:
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
        melted['month'] = melted['month'].replace(month_dict)
        melted['date'] = melted.Year + melted.month
        end_date = time()
        date_time = end_date - start_date

        # we can now get rid of year, day and month:
        melted = melted.drop(columns=['Year', 'month'])

        print(f"The melting took {melt_time}, the splitting {split_time},\
              the cleaning {clean_time} and the date merging time is\
                  {date_time}.")

        return melted

    def meltUnits(self, df):
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
            errprint("meltUnits: the dataframe provided does not have units!")
            return df

        print("Unpivoting the metrics...")

        melted = df.copy()

        val_index = melted[melted.type == 'Value'].index
        if 'Quantity1' in melted.type.unique():
            qty1_index = melted[melted.type == 'Quantity1'].index
            qty2_index = melted[melted.type == 'Quantity2'].index
            melted.loc[qty1_index, 'unit'] = melted.loc[qty1_index, 'Unit1']
            melted.loc[qty2_index, 'unit'] = melted.loc[qty2_index, 'Unit2']
        elif 'Quantity' in melted.type.unique():
            qty_index = melted[melted.type == 'Quantity'].index
            melted.loc[qty_index, 'unit'] = melted.loc[qty_index, 'Unit']

        melted.loc[val_index, 'unit'] = '000s JPY'

        # these columns have been replaced by "unit", won't be useful anymore
        if "Unit1" in df.columns:
            melted.drop(columns=['Unit1', 'Unit2', 'type'], inplace=True)
        else:
            melted.drop(columns=['Unit', 'type'], inplace=True)

        return melted

    def _dfFromRaw(self, file):

        raw = self._openRawData(file)
        clean = self._cleanDataFile(raw)
        month_melted = self.meltMonths(clean)
        normalized = self.meltUnits(month_melted)

        return normalized

    def acquireNeWData(self, file, to_db_file=None, to_dataframe=None):
        """
        Add extra data from file (csv, zip) to the normalized trade dataframe.

        Returns
        -------
        unit_melted : pd.DataFrame

        """
        raw = self._openRawData(file)
        month_melted = self.meltMonths(raw)
        new_data = self.meltUnits(month_melted)

        if to_dataframe is not None:
            # we are giving priority to existing dataframes.
            # if both a DF and a file are specified,
            # only the DF will be considered.
            if to_db_file is not None:
                errprint("warning: both DataFrame and csv file\
                         were provided to normalizeData, but only\
                             the DataFrame will be used.")
            # merge to the existing dataframe
            updated_df = pd.merge(to_dataframe, new_data, axis=0).\
                drop_duplicates()
            return updated_df
        elif to_db_file is not None:
            base = self.openNormalFile(to_db_file)
            # merge to the existing file
            pass
        else:
            # no file or DataFrame to merge with
            return self.data
