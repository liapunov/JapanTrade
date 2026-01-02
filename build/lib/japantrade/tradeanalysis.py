#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import zipfile as zfile


# thanks to MarcH https://stackoverflow.com/a/14981125 for this function
# to print to stderr:
def errprint(*args, **kwargs):
    """Print a message to stderr."""
    print(*args, file=sys.stderr, **kwargs)


class TradeReport():
    """Reporting tools for normalized trade data."""

    # Column names dictionary, before we decide to implement a full fledged
    # trade dataframe class.
    trade_df_col_name = {"country": "Country", "code": "HS", "date": "date",
                         "unit": "unit", "value": "measure"}

    def __init__(self, filename):
        self.trade_df = self.openNormalFile(filename)
        self.dates = sorted(self.trade_df["date"].unique())
        self.first_date = self.dates[0]
        self.last_date = self.dates[-1]
        print(f"TradeReport: the uploaded data start at date {self.first_date}\
              and end at date {self.last_date}.")
        self.hs_codes = self.trade_df["HS"].unique()

    def year(self, date_str):
        if isinstance(date_str, str):
            return int(date_str[:4])
        else:
            return date_str[:4].astype(int)

    def month(self, date_str):
        if isinstance(date_str, str):
            return int(date_str[-2:])
        else:
            return date_str[-2:].astype(int)

    def _last_24_months(self, date_str):
        """Create a list of the last 24 year-date strings."""
        last_month = self.month(date_str)
        first_month = (last_month % 12) + 1
        first_year = self.year(date_str)
        last_year = first_year - 1 if first_month == 1 else first_year - 2

        months = []
        y = first_year
        m = first_month
        for i in range(12):
            months.append(f"{y}-{m.zfill(2)}")
            m = m % 12 + 1
            y = y if y == 1 else y+1

        return months[:12], months[12:]

    def _trendSeries(self, df, curr, prev):
        return (df[curr] / df[prev]) - 1

    def _openzip(f, trade_types):
        with zfile.ZipFile(f) as z:
            n = z.namelist()[0]
            with z.open(n) as file:
                normal = pd.read_csv(file, dtype=trade_types)
                return normal

    def _opencsv(self, filename, trade_types):
        data = pd.read_csv(filename, dtype=trade_types)
        return data

    # identifying the right function to call the right function
    # in order to open the file according to extension.
    opener = {"zip": _openzip, "csv": _opencsv}

    def openNormalFile(self, filename):
        """
        Open a zip or a csv file containing normalized trade data.

        The following columns are expected in the file:
        - "Country", a 3-digit code for each country
        - "HS", a code used to identify the category of the imported/exported
        product.
        - "unit": Kgs, Tons, Number of, 000 JPYs etc.
        - "date": a string of the form AAAA-MM
        - "measure": integer with the value (in "unit") for the tuple
        (country, HS, date)

        In case the file is a zip with multiple csv files, the function opens
        and merges the files into a single DataFrame object,
        which the function returns.
        If the file is a single csv, return the DataFrame built from it.
        """
        pieces = []
        trade_types = {"HS": object, "Country": object, "date": pd.to_datetime,
                       "unit": object, "measure": "Int64"}
        return self.opener[filename.split(".")[-1]](filename, trade_types)

    def _solar_periods(self, df):
        # take only the last two solar years
        this_year = self.year(self.last_date)
        prev_year = int(this_year)-1
        df["period"] = np.where((self.year(df["date"].str) == this_year),
                                "curr", "")
        df["period"] = np.where((self.year(df["date"].str) == prev_year),
                                "prev", df["period"])
        df = df[df["period"] != ""]

        return df

    def _12_month_periods(self, df):

        first_months, last_months = self.one_two_years_back(self.last_date)
        df["period"] = np.where(df["date"].isin(last_months),
                                "curr", "")
        df["period"] = np.where(df["date"].isin(first_months),
                                "prev", "")
        df = df[df["period"] != ""]

        return df

    # identifying the right function to call to mark the current and
    # the previous reporting periods
    def _perioder(self):
        return {"solar": self._solar_periods, "last_12": self._12_month_periods}

    def yoy_country_report(self, country="220", value_cut=10000,
                           val_only=True, method="solar"):
        """
        Create a year-on-year report for a single country given a country code.

        The report will have a variable number of columns.
        The fixed columns are:
        - HS: the HS code. This is going to be the key.
        - item: text description of the HS code
        - value_{cur_period}: the value in thousand yen for the cur_period
        - value_{last_period}: the value in thousand yen for the last_period
        - yoy_value_trend: percentage of increase (decrease) of value from the
        last year

        The variable "quantity" columns (likely at least four) are:
        - a column for each different unit different from the yen
        - a column for the yoy trend of each of the unit above
        Many quantity columns will be empty as each HS code can have at most
        two quantities.
        It is possible to have a value-only report, by passing qty_only=True.

        The rows will be as many as the HS codes, filtered for value.
        The default value under which the HS will not be reported is 10000,
        that is, 10 million yen.

        The country, by default, is "220" (country code of Italy).
        Use the method country_code(country) to retrieve the code of a country.

        The report can account for the reporting period in two ways:
        - "solar": the cur_period will be yyyy_1-12 if the last year has
        all the months, yyyy_1-xx instead
        - "year_ago": takes the last 12 months. cur_period will be
        yyyy-mm_zzzz-nn
        the report will fail if self.trade_df does not have enough data
        in the past, or enough chapters for the two last years.

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
            if (not val_only) or ("000s JPY" in u):
                try:
                    hs_grouped[f"YoY Trend: {u}"] =\
                        self._trendSeries(hs_grouped, f"measure {u} curr",
                                          f"measure {u} prev")
                except KeyError:
                    pass
        if val_only:
            cols = hs_grouped.columns.tolist()
            not_val = [c for c in cols
                       if "000s JPY" not in c and "measure" in c]
            hs_grouped.drop(columns=not_val, inplace=True)

        return hs_grouped
