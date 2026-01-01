#!/usr/bin/env python
# coding: utf-8
import os
import requests
from bs4 import BeautifulSoup
import datetime as dt


class CustomsGrabber():
    """
    Tool to extract and save trade data from the Japanese Customs website.

    Attributes
    ----------
    first_year : integer
        the first available year in the repository.
    page_address : string
        the base url to retrieve the files for a given year
    file_address : string
        the base url to download multiple files in zip format
    dir_params : Dict
        the parameters used to specify whether we want import or export data
    common_params : string
        the parameters that specify the trade data repository within
        the Customs database
    save_folder : string
        the default saving folder

    """

    first_year = 1988
    page_address = "https://www.e-stat.go.jp/en/stat-search/files?"
    file_address = "https://www.e-stat.go.jp/en/stat-search/files/data?page=1&files="
    dir_params = {'HS': {'import': 'tclass2=000001013182',
                         'export': 'tclass2=000001013181'},
                  'PC': {'import': 'tclass2=000001013197',
                         'export': 'tclass2=000001013196'}}
    common_params = "layout=dataset&toukei=00350300&tstat=000001013141"
    save_folder = "../data/"

    def __init__(self):
        """
        Inizializes the data grabber.

        Returns:
            None.

        """
        self.first = self.first_year
        self.last = dt.datetime.now().year
        # year_param = "year=" + str(self.last) + "0"
        # self.query = "&".join([self.common_params, year_param, dir_params])

    def _grabYear(self, year, direction='import', kind='HS', save_folder=None,
                  allow_large_download=False, request_timeout=30):
        """
        Download all the files of a given year into a zip file.

        Parameters
        ----------
        year : integer
            the requested year of data.
        direction : ['import', 'export']
            whether we need the data of trade to Japan (import) or
            from Japan (export).
        kind : ['HS', 'PC']
            whether we need the data classified under HS or PC.
        save_folder : string
            the path to the saving folder.
        allow_large_download : bool, optional
            Set to True to allow downloads that require multiple requests
            (more than 100 files). Defaults to False.
        request_timeout : int, optional
            Timeout (seconds) for HTTP requests. Defaults to 30 seconds.

        Raises
        ------
        ValueError
            If the year is not in the range defined by self.first
            and self.last.

        """
        return self.grabRange(year, year, direction=direction, kind=kind,
                              save_folder=save_folder,
                              allow_large_download=allow_large_download,
                              request_timeout=request_timeout)

    def grabRange(self, from_year, to_year,
                  direction='import', kind='HS', save_folder=None,
                  allow_large_download=False, request_timeout=30):
        """
        Download all the files in a given range of years.

        Parameters
        ----------
        from_year : integer
            the first requested year of data.
        to_year : integer
            the last requested year of data.
        direction : ['import', 'export']
            whether we need the data of trade to Japan (import) or
            from Japan (export).
        kind : ['HS', 'PC']
            whether we need the data classified under HS or PC.
        save_folder : string
            the path to the saving folder.
        allow_large_download : bool, optional
            Set to True to allow downloads that require multiple requests
            (more than 100 files). Defaults to False.
        request_timeout : int, optional
            Timeout (seconds) for HTTP requests. Defaults to 30 seconds.

        Raises
        ------
        ValueError
            If the range is not within the range defined by self.first
                and self.last.

        """
        if to_year > self.last or from_year < self.first:
            raise ValueError(f"The range {from_year}-{to_year} is out of\
                             bound.\
            The year must be within the range {self.first}-{self.last}.")

        files = []

        for y in range(from_year, to_year + 1):
            year_param = "year=" + str(y) + "0"
            query = self.page_address + \
                "&".join([self.common_params,
                          year_param,
                          self.dir_params[kind][direction]])

            year_response = requests.get(query, timeout=request_timeout)
            year_response.raise_for_status()
            year_html = BeautifulSoup(year_response.content, "html.parser")
            file_links = year_html.find_all("a",
                                            attrs={"data-file_type": "CSV"})

            year_files = [":".join([link['data-file_id'],
                                    link['data-release_count']])
                          for link in file_links]
            files.extend(year_files)

        if save_folder is None:
            save_folder = self.save_folder
        os.makedirs(save_folder, exist_ok=True)

        # if there are too many files the GET request becomes too long
        # (the limit is 2048 characters) and we need to split it.
        num_splits = (len(files) // 100) + 1
        if num_splits > 1 and not allow_large_download:
            raise ValueError("Download exceeds 100 files. Set "
                             "allow_large_download=True to proceed.")

        for i in range(num_splits):
            # create chunk of max 100 files
            files_chunk = files[100 * i: min(len(files), 100 * (i + 1))]

            # compose and send get request
            file_request_url = self.file_address + ",".join(files_chunk)
            print(f"Downloading chunk {i + 1}/{num_splits} with"
                  f" {len(files_chunk)} files (years {from_year}-{to_year}).")
            zipped_year_response = requests.get(file_request_url,
                                                stream=True,
                                                timeout=request_timeout)
            zipped_year_response.raise_for_status()

            # save the zip file
            if num_splits == 1:
                file_name = f"{direction}_{kind}_{from_year}-{to_year}.zip"
            else:
                file_name = f"{direction}_{kind}_{from_year}-{to_year}_{i}.zip"
            file_path = os.path.join(save_folder, file_name)
            with open(file_path, 'wb') as f:
                for chunk in zipped_year_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Saved the data as {file_path}.")

    def grabAll(self, direction='import', kind='HS', save_folder=None,
                allow_large_download=False, request_timeout=30):
        """
        Download all the files from all available years.

        Parameters
        ----------
        direction : ['import', 'export']
            whether we need the data of trade to Japan (import) or
            from Japan (export).
        kind : ['HS', 'PC']
            whether we need the data classified under HS or PC.
        save_folder : string
            the path to the saving folder.
        allow_large_download : bool, optional
            Set to True to allow downloads that require multiple requests
            (more than 100 files). Defaults to False.
        request_timeout : int, optional
            Timeout (seconds) for HTTP requests. Defaults to 30 seconds.

        """
        self.grabRange(self.first, self.last, direction=direction, kind=kind,
                       save_folder=save_folder,
                       allow_large_download=allow_large_download,
                       request_timeout=request_timeout)

    def getLastData(self, direction='import', kind='HS', save_folder=None,
                    allow_large_download=False, request_timeout=30):
        """
        Download the file relative to the current year.

        Parameters
        ----------
        direction : ['import', 'export']
            whether we need the data of trade to Japan (import) or
            from Japan (export).
        kind : ['HS', 'PC']
            whether we need the data classified under HS or PC.
        save_folder : string
            the path to the saving folder.
        allow_large_download : bool, optional
            Set to True to allow downloads that require multiple requests
            (more than 100 files). Defaults to False.
        request_timeout : int, optional
            Timeout (seconds) for HTTP requests. Defaults to 30 seconds.

        """
        self._grabYear(self.last, direction=direction, kind=kind,
                       save_folder=save_folder,
                       allow_large_download=allow_large_download,
                       request_timeout=request_timeout)
