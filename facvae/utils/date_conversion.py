import datetime
from obspy.core.utcdatetime import UTCDateTime
import numpy as np
import os

from facvae.marsconverter import MarsConverter

from .project_path import datadir

# Path to Mars data directory.
MARS_PATH = datadir('mars')

MARS_TO_MONTH_INT_CONVERSION = {
    'JAN': '1',
    'FEB': '2',
    'MARCH': '3',
    'APRIL': '4',
    'MAY': '5',
    'JUN': '6',
    'JUL': '7',
    'AUG': '8',
    'SEPT': '9',
    'OCT': '10',
    'NOV': '11',
    'DEC': '12',
}

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


def date_conv_stand_to_mars(date, suffix='.UVW_calib_ACC.mseed'):
    date = yyyy_mm_dd_to_datetime(date)
    year, month, day = date.split('-')
    month = STAND_TO_MARS_MONTH_CONVERSION[month]
    return year + '-' + month + '-' + day + suffix


def yyyy_mm_dd_to_datetime(yyyy_mm_dd):
    return datetime.datetime.strptime(yyyy_mm_dd,
                                      '%Y-%m-%d').strftime("%Y-%b-%d")


def get_waveform_path_from_time_interval(start_time, end_time):

    # Path to raw data and directory for creating scattering dataset.
    waveform_path = datadir(os.path.join(MARS_PATH, 'waveforms_UVW_raw'))
    raw_data_files = os.listdir(waveform_path)

    # By design this should not trigger (as windows are seleted within one day)
    # but a sanity check that the start and end times have the same year,
    # month, and day.
    if start_time.year != end_time.year:
        raise ValueError("Start and end times have different years")

    if start_time.month != end_time.month:
        raise ValueError("Start and end times have different months")

    if start_time.day != end_time.day:
        raise ValueError("Start and end times have different days")

    # Construct the file name.
    file_name = (str(start_time.year) + '-' +
                 STAND_TO_MARS_MONTH_CONVERSION[datetime.date(
                     1900, start_time.month, 1).strftime('%b')] + '-' +
                 str(start_time.day).zfill(2) + '.UVW.mseed')

    filepath = os.path.join(waveform_path, file_name)

    return filepath


def get_time_interval(window_key,
                      window_size=2**17,
                      frequency=20.0,
                      time_zone='UTC'):
    batch = window_key.split('_')[-1]
    year, month, day = window_key.split('-')

    day = day.split('.')[0]
    month = MARS_TO_MONTH_INT_CONVERSION[month]

    batch = int(batch)

    dt = 1 / frequency
    start_time = (batch / 2) * dt * window_size
    end_time = ((batch / 2) + 1) * dt * (window_size - 1)

    str_start_time = UTCDateTime(year + '-' + str(month) + '-' + day)
    str_start_time = str_start_time.__add__(start_time)

    str_end_time = UTCDateTime(year + '-' + str(month) + '-' + day)
    str_end_time = str_end_time.__add__(end_time)

    if time_zone == 'UTC':
        return str_start_time, str_end_time
    elif time_zone == 'LMST':
        mars_date = MarsConverter()
        str_start_time = mars_date.get_utc_2_lmst(utc_date=str_start_time)
        str_end_time = mars_date.get_utc_2_lmst(utc_date=str_end_time)
        return str_start_time, str_end_time
    else:
        raise NotImplementedError('Time zone not implemented')


def is_night_time_event(event_start, event_end):
    mars_date = MarsConverter()

    event_start = mars_date.get_utc_2_lmst(utc_date=event_start)
    event_end = mars_date.get_utc_2_lmst(utc_date=event_end)

    day_start_time = 'T05:00:00.000000'
    day_end_time = 'T19:00:00.000000'

    event_start_day = event_start.split('T')[0]
    event_end_day = event_end.split('T')[0]

    if event_start_day == event_end_day:
        same_day_start = event_start.split('T')[0] + day_start_time
        same_day_end = event_end_day.split('T')[0] + day_end_time
        if event_start > same_day_end or event_end < same_day_start:
            return True
    else:
        next_day_start = event_end_day.split('T')[0] + day_start_time
        same_day_end = event_start_day.split('T')[0] + day_end_time
        if event_start > same_day_end and event_end < next_day_start:
            return True
    return False


def create_lmst_xticks(start_time: str,
                       end_time: str,
                       window_size: int = 2**17,
                       frequency: float = 20.0,
                       time_zone: str = 'LMST') -> np.ndarray:
    """
    Create an array of datetime objects at regular intervals.

    Args:
        start_time: A string representing the start time in the format
            'YYYY-MM-DDTHH:MM:SS.ssssss'.
        end_time: A string representing the end time in the format
            'YYYY-MM-DDTHH:MM:SS.ssssss'.
        window_size: An integer specifying the number of datetime objects to
            create.
        time_zone: A string specifying the time zone of the input times. Can be
            either 'LMST' or 'UTC'.

    Returns:
        An array of datetime objects between start_time and end_time with a
        regular interval.

    Raises:
        NotImplementedError: If the time_zone is not 'LMST' or 'UTC'.
    """
    if time_zone == 'LMST':
        # Convert start_time and end_time from UTC to Local Mean Solar Time
        # (LMST) of Mars.
        mars_date = MarsConverter()
        start_time = mars_date.get_utc_2_lmst(utc_date=start_time)
        end_time = mars_date.get_utc_2_lmst(utc_date=end_time)
    elif time_zone == 'UTC':
        pass  # No conversion needed.
    else:
        raise NotImplementedError('Time zone not implemented')

    # Extract the day portion of start_time and end_time.
    start_day = start_time.split('T')[0]
    end_day = end_time.split('T')[0]

    # Extract the time portion of start_time and end_time and convert them to
    # datetime objects.
    start_time = start_time.split('T')[1]
    end_time = end_time.split('T')[1]
    start_time = datetime.datetime.strptime(start_time, '%H:%M:%S.%f')
    end_time = datetime.datetime.strptime(end_time, '%H:%M:%S.%f')

    # If end_day is greater than start_day, add a day to end_time to include
    # the entire day.
    if int(end_day) > int(start_day):
        end_time = end_time + datetime.timedelta(days=1)

    # Create an array of timestamp values between start_time and end_time with
    # a regular interval.
    times = np.linspace(start_time.timestamp(),
                        end_time.timestamp(),
                        num=window_size)
    # Convert the timestamp values to datetime objects and return the array.
    times = np.array([datetime.datetime.fromtimestamp(ts) for ts in times])
    return times


def lmst_xtick(input_utc):
    mars_date = MarsConverter()
    start_time = mars_date.get_utc_2_lmst(utc_date=input_utc)
    start_time = start_time.split('T')[1]
    start_time = datetime.datetime.strptime(start_time, '%H:%M:%S.%f')
    return start_time
