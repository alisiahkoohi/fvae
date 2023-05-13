"""Functions for loading Mars raw data.
"""

import datetime
import os
from typing import List, Optional, Tuple
import numpy as np
import obspy
import json

from srcsep.utils import windows
from facvae.utils import configsdir


class MarsquakeSeparationSetup(object):
    """
    A class for setting up marsquake source separation.
    """
    def __init__(self, mars_raw_path: str, json_file: str) -> None:
        """Initializes the class.
        """
        self.mars_raw_path = mars_raw_path
        self.json_path = os.path.join(configsdir('marsquakes'), json_file)
        self.read_json()

    def read_json(self) -> None:
        """
        Reads the JSON file and stores the data in the class.
        """
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        self.background_data_path = [
            os.path.join(self.mars_raw_path, date)
            for date in data['background']['date']
        ]

        self.background_data_range = [
            idx_range for idx_range in data['background']['idx_range']
        ]

        self.marsquake_data_path = os.path.join(self.mars_raw_path,
                                                data['marsquake']['date'])
        self.marsquake_data_range = data['marsquake']['idx_range']

        self.arrival_time = {
            "P":
            data['marsquake']['arrival_time']['P'],
            "S":
            data['marsquake']['arrival_time']['S'],
            "PP":
            data['marsquake']['arrival_time']['PP']
            if 'PP' in data['marsquake']['arrival_time'] else None,
            "SS":
            data['marsquake']['arrival_time']['SS']
            if 'SS' in data['marsquake']['arrival_time'] else None,
        }

    def get_windowed_marsquake_data(self, window_size: int) -> np.ndarray:
        """
        Extracts and windows seismic data from a specific file on Mars.

        Args:
            window_size: the size of each window.

        Returns:
            a numpy array containing the extracted and windowed data
        """
        stream = obspy.read(self.marsquake_data_path).detrend(type='spline',
                                                              order=2,
                                                              dspline=2000,
                                                              plot=False)
        trace = stream[0][self.marsquake_data_range[0]:self.
                          marsquake_data_range[1]].astype(np.float64)

        # Apply windowing to the extracted data and return the resulting numpy
        # array.
        windowed_trace = np.expand_dims(
            windows(trace, window_size, window_size, 0), 1)
        return windowed_trace

    def get_windowed_background_data(self, window_size: int,
                                     stride: int) -> np.ndarray:
        """
        Extracts and windows data from the given files based on the specified
        index ranges, window size, and stride.

        Args:
            file_paths: a list of file paths to the data files
            idx_ranges: a list of index ranges to be extracted from each data
            file
            window_size: the size of each window
            stride: the stride of the window

        Returns:
            a numpy array containing the extracted and windowed data from all
            the files
        """
        # Initialize an empty list to store the extracted data.
        background_data = []

        # Loop over each pair of `(idx_range, file_path)` in the input
        # `self.background_data_range` and `self.background_data_path`.
        for idx_range, file_path in zip(self.background_data_range,
                                        self.background_data_path):

            # Read the data from the file and apply detrending.
            stream = obspy.read(file_path).detrend(type='spline',
                                                   order=2,
                                                   dspline=2000,
                                                   plot=False)

            # Extract the data within the specified index range and convert it to a
            # float numpy array.
            trace = stream[0][range(*idx_range)].astype(np.float64)

            # Apply windowing to the extracted data and add it to the
            # `background_data` list.
            windowed_trace = np.expand_dims(
                windows(trace, window_size, stride, 0), 1)
            background_data.append(windowed_trace)

        # Concatenate the extracted and windowed data from all the files and
        # return the resulting numpy array.
        return np.concatenate(background_data, axis=0)

    def get_time_axis(
        self,
        time_offset: int,
    ) -> np.ndarray:
        """
        Gets the time axis of the Marsquake at the given index for plotting.

        Args:
            time_offset (int): the number of time samples to extract

        Returns:
            A tuple containing the time axis of the extracted data, and the start
            times of the P and S waves.
        """
        # If no index is specified, use a default file path on Mars and extract a
        # specific portion of the data.
        stream = obspy.read(self.marsquake_data_path)

        # Extract the time axis of the specified portion of the data.
        time_axis = stream[0].times(
            type="utcdatetime"
        )[self.marsquake_data_range[0]:self.marsquake_data_range[0] +
          time_offset]

        # Return the time axis and start times of the P and S waves.
        return time_axis

    def get_arrival_times(
        self, ) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Gets the P and S wave arrivals.

        Returns:
            A tuple containing the time axis of the extracted data, and the start
            times of the P and S waves.
        """
        # Set the start times of the P and S waves.
        p_start = datetime.datetime(*self.arrival_time['P'])
        s_start = datetime.datetime(*self.arrival_time['S'])

        pp_start = datetime.datetime(
            *self.arrival_time['PP']
        ) if self.arrival_time['PP'] is not None else None
        ss_start = datetime.datetime(
            *self.arrival_time['SS']
        ) if self.arrival_time['SS'] is not None else None

        # Return the time axis and start times of the P and S waves.
        return p_start, s_start, pp_start, ss_start


class GlitchSeparationSetup(object):
    """
    A class for setting up glitch separation.
    """
    def __init__(self, mars_raw_path: str, json_file: str) -> None:
        """Initializes the class.
        """
        self.mars_raw_path = mars_raw_path
        self.json_path = os.path.join(configsdir('glitch'), json_file)
        self.read_json()

    def read_json(self) -> None:
        """
        Reads the JSON file and stores the data in the class.
        """
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        self.background_data_path = [
            os.path.join(self.mars_raw_path, date)
            for date in data['background']['date']
        ]

        self.background_data_range = [
            idx_range for idx_range in data['background']['idx_range']
        ]

        self.glitch_data_path = os.path.join(self.mars_raw_path,
                                             data['glitch']['date'])
        self.x_baseline_data_path = os.path.join(self.mars_raw_path,
                                                 data['glitch']['x_baseline'])
        self.g_baseline_data_path = os.path.join(self.mars_raw_path,
                                                 data['glitch']['g_baseline'])
        self.glitch_data_range = data['glitch']['idx_range']

    def get_windowed_glitch_data(self, window_size: int) -> np.ndarray:
        """
        Extracts and windows seismic data from a specific file on Mars.

        Args:
            window_size: the size of each window.

        Returns:
            a numpy array containing the extracted and windowed data
        """
        stream = obspy.read(self.glitch_data_path).detrend(type='spline',
                                                           order=2,
                                                           dspline=2000,
                                                           plot=False)

        trace = stream[0][self.glitch_data_range[0]:self.
                          glitch_data_range[1]].astype(np.float64)

        # Apply windowing to the extracted data and return the resulting numpy
        # array.
        windowed_trace = np.expand_dims(
            windows(trace, window_size, window_size, 0), 1)
        return windowed_trace

    def get_windowed_baseline(self, window_size: int) -> np.ndarray:
        """
        Extracts and windows baseline result.

        Args:
            window_size: the size of each window.

        Returns:
            a numpy array containing the extracted and windowed data
        """
        x_stream = obspy.read(self.x_baseline_data_path).detrend(type='spline',
                                                                 order=2,
                                                                 dspline=2000,
                                                                 plot=False)
        g_stream = obspy.read(self.g_baseline_data_path).detrend(type='spline',
                                                                 order=2,
                                                                 dspline=2000,
                                                                 plot=False)

        x_trace = x_stream[0][self.glitch_data_range[0]:self.
                              glitch_data_range[1]].astype(np.float64)
        g_trace = g_stream[0][self.glitch_data_range[0]:self.
                              glitch_data_range[1]].astype(np.float64)

        # Apply windowing to the extracted data and return the resulting numpy
        # array.
        x_windowed_trace = np.expand_dims(
            windows(x_trace, window_size, window_size, 0), 1)
        g_windowed_trace = np.expand_dims(
            windows(g_trace, window_size, window_size, 0), 1)
        return x_windowed_trace, g_windowed_trace

    def get_windowed_background_data(self, window_size: int,
                                     stride: int) -> np.ndarray:
        """
        Extracts and windows data from the given files based on the specified
        index ranges, window size, and stride.

        Args:
            file_paths: a list of file paths to the data files
            idx_ranges: a list of index ranges to be extracted from each data
            file
            window_size: the size of each window
            stride: the stride of the window

        Returns:
            a numpy array containing the extracted and windowed data from all
            the files
        """
        # Initialize an empty list to store the extracted data.
        background_data = []

        # Loop over each pair of `(idx_range, file_path)` in the input
        # `self.background_data_range` and `self.background_data_path`.
        for idx_range, file_path in zip(self.background_data_range,
                                        self.background_data_path):

            # Read the data from the file and apply detrending.
            stream = obspy.read(file_path).detrend(type='spline',
                                                   order=2,
                                                   dspline=2000,
                                                   plot=False)

            # Extract the data within the specified index range and convert it to a
            # float numpy array.
            trace = stream[0][range(*idx_range)].astype(np.float64)

            # Apply windowing to the extracted data and add it to the
            # `background_data` list.
            windowed_trace = np.expand_dims(
                windows(trace, window_size, stride, 0), 1)
            background_data.append(windowed_trace)

        # Concatenate the extracted and windowed data from all the files and
        # return the resulting numpy array.
        return np.concatenate(background_data, axis=0)

    def get_time_axis(
        self,
        time_offset: int,
    ) -> np.ndarray:
        """
        Gets the time axis of the glitch at the given index for plotting.

        Args:
            time_offset (int): the number of time samples to extract

        Returns:
            A tuple containing the time axis of the extracted data, and the start
            times of the P and S waves.
        """
        # If no index is specified, use a default file path on Mars and extract a
        # specific portion of the data.
        stream = obspy.read(self.glitch_data_path)

        # Extract the time axis of the specified portion of the data.
        time_axis = stream[0].times(
            type="utcdatetime"
        )[self.glitch_data_range[0]:self.glitch_data_range[0] + time_offset]

        # Return the time axis and start times of the P and S waves.
        return time_axis