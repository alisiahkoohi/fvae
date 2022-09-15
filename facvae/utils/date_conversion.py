from datetime import datetime

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
    date = yyyy_mm_dd_to_datetime(date)
    year, month, day = date.split('-')
    month = STAND_TO_MARS_MONTH_CONVERSION[month]
    return year + '-' + month + '-' + day + '.UVW_calib_ACC.mseed'


def yyyy_mm_dd_to_datetime(yyyy_mm_dd):
    return datetime.strptime(yyyy_mm_dd, '%Y-%m-%d').strftime("%Y-%b-%d")
