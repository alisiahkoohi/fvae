# Author: Greg

import os
import pandas as pd
from datetime import datetime
from obspy import UTCDateTime

# Mars time library
from facvae.marsconverter import MarsConverter
from facvae.utils import catalogsdir

mDate = MarsConverter()

bound = 1000


def jd_to_date(jdate):
    """
   Plain function to convert Julian format to standard date eg: 2021-074 =>
   2021-03-15


   """
    fmt = '%Y-%j'
    datestd = datetime.strptime(jdate, fmt).date()
    return (datestd)


path_2_data = catalogsdir()

file_pres_drop = os.path.join(path_2_data, "alldrop_ordered.txt")
if os.path.isfile(file_pres_drop):
    df_presdrop = pd.read_csv(file_pres_drop,
                              delimiter=";",
                              skiprows=[0],
                              names=["drop", "ltst", "sol", "time", "ratio"])
else:
    print("File does not exist")

df_presdrop["eventTime"] = df_presdrop["time"]
for i, row in df_presdrop.iterrows():
    df_presdrop["eventTime"][i] = UTCDateTime(
        str(jd_to_date(row["time"][1:].split("T")[0])) + "T" +
        row["time"].split("T")[1])

df_presdrop["lowerbound"] = df_presdrop["eventTime"] - bound
df_presdrop["upperbound"] = df_presdrop["eventTime"] + bound

# Estimate LMST time in another column
# get_utc_2_lmst(self, utc_date=None, output='date')
df_presdrop['lmst'] = df_presdrop.apply(
    lambda x: mDate.get_utc_2_lmst(utc_date=x["eventTime"], output='decimal'),
    axis=1)

df_presdrop.to_pickle(os.path.join(path_2_data, 'pressure_drops_InSIght.pkl'))
