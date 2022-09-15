"""Probing the Mars raw data.
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md

from facvae.utils import datadir, date_conv_stand_to_mars

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_RAW_WAV_PATH = datadir(os.path.join(MARS_PATH, 'waveforms_h5'))
RAW_WAV_FILENAME = 'day-long-raw-waveforms.h5'


def probe_mars_data(query):
    filename = date_conv_stand_to_mars(query)

    file_path = os.path.join(MARS_RAW_WAV_PATH, RAW_WAV_FILENAME)
    # HDF5 file.
    file = h5py.File(file_path, 'r')

    daylong_event = file[filename]
    trace = daylong_event['waveform'][...]
    start_time = str(daylong_event['start-time'][...].astype(str))
    end_time = str(daylong_event['end-time'][...].astype(str))

    file.close()
    return trace, start_time, end_time


if __name__ == '__main__':

    plt.figure()
    for date in ['2019-11-04', '2019-11-05', '2019-11-06']:
        trace, start_time, end_time = probe_mars_data(date)

        times = np.arange(
            np.datetime64(start_time[:-1]),
            np.datetime64(end_time[:-1]) + np.timedelta64(50, 'ms'),
            np.timedelta64(50, 'ms')).astype('datetime64[s]')
        plt.plot_date(times, trace, xdate=True, fmt='')

    ax = plt.gca()

    ax.xaxis.set_major_locator(md.HourLocator(interval=9))
    ax.xaxis.set_major_formatter(md.DateFormatter('%D-%H:%M'))
    # ax.set_xlim([
    #     np.datetime64(start_time[:-1]),
    #     np.datetime64(end_time[:-1]) + np.timedelta64(50, 'ms')
    # ])
    # ax.set_title(start_time.split('T')[0])
    plt.show()

