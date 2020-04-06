# JapanTrade
Growing repository of tools for analyzing Japanese customs trade data.
Python / Pandas + Seaborn
The biggest problem in working with Japanese Customs data is that, as of April 2020, there does not seem to be an easily accesible API for the trade data - indeed, there is no easy way to slice any piece of information from the website, which only provides csv tables divided by months and including only part of the HS (Harmonized System) codes in each file.
What we are trying to do here is:
- First of all, starting from single csv files, obtain a highly normalized version of the time series
- secondly, extract and normalize the description of each HS code, as the codes descriptions are often distributed hierarchically over many lines of a csv or worse, HTML table.
- finally, create a few methods to extract common queries, such as the comparison year-on-year of a specific HS code for a specific country, or the chart of the time series for a specific HS code and a group of countries.

## Available modules as of April 6, 2020
- Japanese_HS_Codes.ipynb: extract the 4, 6 and 9 lengths HS (harmonized system) codes and descriptions from a static csv table.
- Trade Tools: tools for extracting, merging and normalising into a pandas DF the trade data csv tables that the customs.or.jp website provides
- (in progress) Trade data analysis: tools for creating useful tables and charts from a normalised trade data frame.

## Urgent To Do as of April 6, 2020
- Complete the TradeReport class in the Trade data analysis module
- Clean and add documentation to the Japanese_HS_codes module

# To appear next
- Tools for extracting HS code descriptions from the HTML tables available in the customs.or.jp website, in particular:
-- The assertions are still missing
-- The textual descriptions have not been included yet
- Tools for downloading/updating all the trade data CSV tables from the customs.or.jp website
- A TimeseriesTrendByCountry class int he Trade data analysis module, in order to compare and chart the trend of a given HS code
- Tools for finding an HS code by textual approximate search (NLP module will be needed)
