import sys
from pathlib import Path
from MarsConverter import MarsConverter


landerconfigfile = './landerconfig.xml'
my_file = Path(landerconfigfile)
mDate = MarsConverter()

mDate.get_lmst_to_utc('1359T05:00:00.000000')

mDate.get_lmst_to_utc('1359T19:00:00.000000')
