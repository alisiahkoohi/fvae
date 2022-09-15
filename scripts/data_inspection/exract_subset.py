import h5py
import numpy as np
import os

from facvae.utils import datadir

MARS_PATH = datadir('mars')
SCAT_COV_FILENAME = 'scat_covs.h5'
SUBSET_SCAT_COV_FILENAME = '2019-JUL-01-scat_covs.h5'

if __name__ == '__main__':
    # Path to the file.
    path = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))
    file_path = os.path.join(path, SCAT_COV_FILENAME)

    # HDF5 file.
    file = h5py.File(file_path, 'r')

    subset_path = datadir(os.path.join(MARS_PATH, 'scat_covs_subset_h5'))
    subset_file_path = os.path.join(subset_path, SUBSET_SCAT_COV_FILENAME)
    subset_file = h5py.File(subset_file_path, 'w')

    for filename in file.keys():
        if '2019-JUL-01.UVW_calib_ACC.mseed' in filename:
            print(filename)
            subset_group = subset_file.create_group(filename)
            subset_group.create_dataset('waveform',
                                        data=file[filename]['waveform'][...],
                                        dtype=np.float32)
            subset_group.create_dataset('scat_cov',
                                        data=file[filename]['scat_cov'][...],
                                        dtype=np.float32)

    file.close()
    subset_file.close()
