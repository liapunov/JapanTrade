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

default_pc_dict = "../data/PC_codes.csv"
default_hs_dict = "../data/HS_codes.csv"
default_country_dict = "../data/country_codes.csv"

class TradeReport():
    """Reporting tools for normalized trade data."""

    # Column names dictionary, before we decide to implement a full fledged
    # trade dataframe class.
    trade_df_col_name = {"kind", "country", "code", "date", "unit", "value"}
    trade_df_keys = ["kind", "date", "country", "code", "unit"]
    trade_df_units = {' L', 'PR', 'JPY', 'SM', 'DZ', 'KL', 'TH', 'KG', 'ST', 
                      'GT', 'MT', 'CT', 'NO', 'GR', ' M', 'CM'}

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

    def _dates_exist(self, start_date, end_date):
        return start_date >= self.first_date and end_date <= self.last_date

    def _country_exists(self, country):
        return int(country) in self.trade_df.index.get_level_values('country')

    def yoy_country_report(self, kind="HS", country="220", value_cut=10**6,
                           code_level=3, val_only=True, method="last_12"):
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
        
        code_level : Int64, optional
            The minimum length of the codes to be reported. \
The default is 3.

        val_only : boolean, optional
            Whether to report only the monetary value of trade. \
The default is True.

        method : {"last_year", "last_12"}, optional
            The report can account for the reporting period in two ways:
        - "curr_year": the cur_period will be yyyy_1-12 if the current year \
has all the months, yyyy_1-xx instead
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
        if method == 'curr_year':
            end_date = self.last_date
            start_date = pd.datetime(f"{self.last_date.year()}-1-1")
        elif method == 'last_year':
            end_date = pd.datetime(f"{self.last_date.year-1}-12-1")
            start_date = pd.datetime(f"{self.last_date.year-1}-1-1")
        elif method == 'last_12':
            end_date = self.last_date
            start_date = end_date - pd.DateOffset(months=12)
        else:
            raise ValueError(f"Unknown method: {method}")
        if not self._dates_exist(start_date, end_date):
            raise ValueError("Cannot use this method: not enough past data.")
        # check the country
        if not self._country_exists(country):
            raise ValueError(f"Country code {country} does not exist")

        # time to drill down the dataframe.
        # make sure to select only the codes with length>=code_level
        code_mask = self.trade_df.index.get_level_values('code').str.len()\
            >= code_level

        # restrict the unit to 'JPY' if val_only.
        if val_only:
            drilled_df = self.trade_df.loc[kind,
                                           start_date:end_date,
                                           country,
                                           code_mask,
                                           'JPY'].copy()
        else:
            drilled_df = self.trade_df.loc[kind,
                                           start_date:end_date,
                                           country,
                                           code_mask,
                                           :].copy()
            log.debug(drilled_df.head(5))
        # pivot by hs code index & unit column.
        # Grouper is needed in order to properly group by month.
        year_sums = drilled_df.droplevel(['kind', 'country']).\
            groupby(['code', 'unit'] +
                    [pd.Grouper(freq=pd.offsets.MonthBegin(12),
                                level='date')])\
            .sum()
        log.debug(year_sums.head(5))
        year_compared = year_sums.reset_index().pivot(index='code',
                                                      columns=['unit', 'date'],
                                                      values='value')
        log.debug(year_compared)
        hs_grouped = year_compared .\
            sort_values(by=('JPY', end_date), ascending=False)

        return hs_grouped
