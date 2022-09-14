"""Probing the Mars raw data.
"""

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as md

from facvae.utils import datadir

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_RAW_WAV_PATH = datadir(os.path.join(MARS_PATH, 'waveforms_h5'))
RAW_WAV_FILENAME = 'day-long-raw-waveforms.h5'

STAND_TO_MARS_MONTH_CONVERSION = {
    'Jan': 'JAN',
    'Feb': 'FEB',
    'Mar': 'MARCH',
    'Apr': 'APRIL',
    'May': 'MAY',
    'Jun': 'JUN',
    'Jul': 'JUL',
    'Aug': 'AUG',
    'Sep': 'SEPT',
    'Oct': 'OCT',
    'Nov': 'NOV',
    'Dec': 'DEC'
}

MARS_TO_STAND_MONTH_CONVERSION = {}
for key, value in STAND_TO_MARS_MONTH_CONVERSION.items():
    MARS_TO_STAND_MONTH_CONVERSION[value] = key


def date_conv_mars_to_stand(filename):
    date_only = filename.split('.')[0][:-3]
    year, month, day = date_only.split('-')
    month = MARS_TO_STAND_MONTH_CONVERSION[month]
    return year + '-' + month + '-' + day


def date_conv_stand_to_mars(date):
    year, month, day = date.split('-')
    month = STAND_TO_MARS_MONTH_CONVERSION[month]
    return year + '-' + month + '-' + day + '.UVW_calib_ACC.mseed'


def yyyy_mm_dd_to_datetime(yyyy_mm_dd):
    return datetime.strptime(yyyy_mm_dd, '%Y-%m-%d').strftime("%Y-%b-%d")


def probe_mars_data(query):
    query = yyyy_mm_dd_to_datetime(query)
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
    for date in ['2019-10-04', '2019-10-05', '2019-10-06']:
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

