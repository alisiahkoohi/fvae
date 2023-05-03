# Author: Greg

import os
import pandas as pd
from datetime import datetime
from obspy import UTCDateTime

# Mars time library
from facvae.marsconverter import MarsConverter
from facvae.utils import catalogsdir

mDate = MarsConverter()


path_2_data = catalogsdir()

file_pres_drop = os.path.join(path_2_data, "Johns_glitch_Sol184_.txt")
if os.path.isfile(file_pres_drop):
    df_presdrop = pd.read_csv(file_pres_drop,
                              delimiter="\t",
                              skiprows=[0],
                              names=["Time start detection time UTC", "Time end detection time UTC"])
else:
    print("File does not exist")

df_presdrop["start_time"] = df_presdrop["Time start detection time UTC"]
df_presdrop["end_time"] = df_presdrop["Time end detection time UTC"]
# All ones.
df_presdrop["glitch"] = 1.0

df_presdrop.to_pickle(os.path.join(path_2_data, 'Salma_glitches_InSIght.pkl'))
