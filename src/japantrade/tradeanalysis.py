"""Japan Customs trade report tool.

This module provides functions to create reports from trade data files as \
processed by the tradefile module.

It is possible to create reports such as YoY trends for a country, or \
timeseries for a set of countries and a specific good.

The tool accepts files in .csv form (it uses the tools in the tradefile \
module), possibly archived into a single .zip file.

This module contains a single class, TradeFile.
"""
# !/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import numpy as np
import pandas as pd
import logging
from tradefile import TradeFile as trf


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
log = logging.getLogger("tradeanalysis")


class TradeReport():
    """Reporting tools for normalized trade data."""

    # Column names dictionary, before we decide to implement a full fledged
    # trade dataframe class.
    trade_df_col_name = {"kind", "country", "code", "date", "unit", "value"}
    trade_df_keys = ["kind", "date", "country", "code", "unit"]
    trade_df_units = {'NO', 'MT', 'KG', 'KL', 'SM', 'GR', 'TH', 'DZ', 'JPY'}

    def __init__(self, source_file=None, source_df=None, kind='infer'):
        """
        Initialize the TradeReport object from a source database.

        Parameters
        ----------
        source_file : string, optional
            Path to a (possibly archived) csv file. The default is None.
        source_df : pd.DataFrame, optional
            A preexisting DataFrame source. The default is None.

        Returns
        -------
        None.

        """
        if source_file is not None:
            self.trade_df = self._tradeDFFromFile(source_file)
        elif source_df is not None:
            self.trade_df = self._tradeDFFromDF(source_df)
        self.first_date = self.trade_df.index.get_level_values('date')[0]
        self.last_date = self.trade_df.index.get_level_values('date')[-1]
        self.codes = self.trade_df.index.get_level_values('code').unique()
        print(f"TradeReport: the uploaded data start at date {self.first_date}\
 and end at date {self.last_date}.")

    def _check_source_integrity(self, df_to_check):
        """
        Verify that the source DataFrame has the correct columns and data.

        Parameters
        ----------
        df_to_check : pd.DataFrame
            A DataFrame to check-up.

        Returns
        -------
        True id the DataFrame is valid, False otherways.

        """
        right_columns = self.trade_df_col_name
        these_columns = set(df_to_check.columns)
        check_columns = (these_columns == right_columns)
        log.debug(f"{these_columns} == {right_columns}?")
        right_units = self.trade_df_units
        these_units = set(df_to_check['unit'])
        check_units = these_units.issubset(right_units)
        log.debug(f"{these_units} < {right_units}?")
        return check_columns and check_units

    def _trendSeries(self, df, curr, prev):
        return (df[curr] / df[prev]) - 1

    def _tradeDFFromFile(self, filename):
        """
        Open a zip or a csv file containing normalized trade data.

        The following columns are expected in the file:
        - kind: 'PC' code or 'HS' code type
        - "country", a 3-digit code for each country
        - "code", a code used to identify the category of the imported/exported
        product.
        - "unit": Kgs, Tons, Number of, JPY etc.
        - "date": a string of the form AAAA-MM
        - "value": integer with the value (in "unit") for the tuple
        (kind, country, code, date)

        In case the file is a zip with multiple csv files, the function opens
        and merges the files into a single DataFrame object,
        which the function returns.
        If the file is a single csv, return the DataFrame built from it.
        """
        source = trf(filename, raw=False).data
        if self._check_source_integrity(source):
            keys = self.trade_df_keys
            source['date'] = pd.to_datetime(source['date'])
            indexed_df = source.dropna().set_index(keys).sort_index()
            return indexed_df
        else:
            raise ValueError("The provided source data are not valid.")

    def _tradeDFFromDF(self, df, kind):
        if self._check_source_integrity(df):
            keys = self.trade_df_keys
            df['date'] = pd.to_datetime(df['date'])
            sorted_df = df.dropna().set_index(keys).sort_index()
            return sorted_df

    def yoy_country_report(self, kind="HS", country="220", value_cut=10**6,
                           val_only=True, method="last_12"):
        """
        Create a year-on-year report for a single country given a country code.

        Parameters
        ----------
        kind : {"HS", "PC"}, optional
            Whether to use the HS codes or the PC codes.
            Default is "HS".
            If there are no codes of that type, the report fails with a \
ValueError.

        country : string, optional
            Country code for the report. The default is "220" \
(country code of Italy). Use the method _country_code(country) to retrieve \
the code of a country.

        value_cut : Int64, optional
            Minimum threshold of reporting for the monetary value of a trade. \
The default is 10 million.

        val_only : boolean, optional
            Whether to report only the monetary value of trade. \
The default is True.

        method : {"last_year", "last_12"}, optional
            The report can account for the reporting period in two ways:
        - "last_year": the cur_period will be yyyy_1-12 if the last year has \
all the months, yyyy_1-xx instead
        - "last_12": takes the last 12 months. cur_period will be \
yyyy-mm_zzzz-nn.
The report will fail if self.trade_df does not have enough data in the past.
The default is "last_12".

        Returns
        -------
        hs_grouped : pd.DataFrame
            The report will have a variable number of columns.
        The fixed columns are:
        - code: the HS or PC code. This is going to be the key.
        - item: text description of the HS code
        - value_{cur_period}: the value in thousand yen for the cur_period
        - value_{last_period}: the value in thousand yen for the last_period
        - yoy_value_trend: percentage of increase (decrease) of value from \
the last year.
        The variable "quantity" columns (likely at least four) are:
        - a column for each different unit different from the yen
        - a column for the yoy trend of each of the unit above
        Many quantity columns will be empty as each HS code can have at most \
two quantities.

        """
        # do we have all that we need?
        # check the periods first.
        pass

        # now let's check the HS codes. It might be that not all the chapters
        # were present.
        # since there may be HS codes that do not appear yoy, we will accept
        # with a 90% overlapping.
        pass

        # we only need the selected country. The Country column also can go.
        country_slice = self.trade_df[self.trade_df["Country"] == country]\
            .drop(columns=["Country"])

        # now let's filter by minimum value
        if value_cut is not None:
            country_slice = country_slice[country_slice["measure"] > value_cut]

        # now let's aggregate by period
        period_map = self._perioder()
        date_filtered_df = period_map[method](country_slice)
        print(date_filtered_df)

        # these are the units that will be used to calculate the trends
        units = date_filtered_df.unit.unique()
        print(units)

        # pivot by hs code index & unit column
        hs_grouped = pd.pivot_table(date_filtered_df, index=["HS"],
                                    columns=["unit", "period"],
                                    aggfunc={"measure": np.sum}).reset_index()
        hs_grouped.columns = [' '.join(col).strip()
                              for col in hs_grouped.columns.values]
        print(hs_grouped.head())
        print(hs_grouped.columns.tolist())

        # adding the trend columns for each unit
        for u in units:
            if (not val_only) or ("JPY" in u):
                try:
                    hs_grouped[f"YoY Trend: {u}"] =\
                        self._trendSeries(hs_grouped, f"measure {u} curr",
                                          f"measure {u} prev")
                except KeyError:
                    pass
        if val_only:
            cols = hs_grouped.columns.tolist()
            not_val = [c for c in cols
                       if "JPY" not in c and "measure" in c]
            hs_grouped.drop(columns=not_val, inplace=True)

        return hs_grouped
