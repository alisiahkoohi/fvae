"""Reading a subset of the Mars InSight dataset.

The HDFS file contains windowed raw waveform and corresponding scattering
covariances for July 01, 2019. The window size used here is 2**17 samples.

The code that generated the scattering covariances is linked below:
https://github.com/alisiahkoohi/factorialVAE/blob/a67c6d1b4a3cfd2fbef6e60b74204663902474ef/scripts/generate_scattering_cov.py
"""

import h5py

# Adjust the path to the data directory.
SCAT_COV_FILE_PATH = '2019-JUL-01-scat_covs.h5'

if __name__ == '__main__':
    # Read the HDF5 file, containing the windowed waveform and corresponding
    # scattering covariances. The data structure is as follows:

    # file
    #    ├── '2019-JUL-01.UVW_calib_ACC.mseed_0'    # First window.
    #        ├── 'waveform'                         # The raw windowed waveform.
    #        ├── 'scat_cov'                         # Scattering covariances.
    #    ├── '2019-JUL-01.UVW_calib_ACC.mseed_1'    # Second window.
    #        ├── 'waveform'                         # The raw windowed waveform.
    #        ├── 'scat_cov'                         # Scattering covariances.
    #    ├── '2019-JUL-01.UVW_calib_ACC.mseed_2'    # Third window.
    #        ├── 'waveform'                         # The raw windowed waveform.
    #        ├── 'scat_cov'                         # Scattering covariances.
    #    ⋮
    file = h5py.File(SCAT_COV_FILE_PATH, 'r')

    # Extract a data window, e.g., the first one here. There are 25 windows in
    # total: '2019-JUL-01.UVW_calib_ACC.mseed_0',
    # '2019-JUL-01.UVW_calib_ACC.mseed_1', ...,
    # '2019-JUL-01.UVW_calib_ACC.mseed_24'
    windowed_data = file['2019-JUL-01.UVW_calib_ACC.mseed_0']

    # Extract waveform and scattering covariances.
    # The shape of `waveform` is (131072,).
    waveform = windowed_data['waveform'][...]
    # The shape of `scat_cov` is (348,).
    scat_cov = windowed_data['scat_cov'][...]

    # Close file.
    file.close()
