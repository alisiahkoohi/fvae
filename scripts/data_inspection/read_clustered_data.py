"""Reading a subset of the clustered Mars InSight dataset.

The HDFS file contains windowed raw waveform, corresponding scattering
covariances, file name (with window number) for 10 samples of the five clusters obtained via a GMVAE.

The code that generated the scattering covariances is linked below:
https://github.com/alisiahkoohi/factorialVAE/blob/a67c6d1b4a3cfd2fbef6e60b74204663902474ef/scripts/generate_scattering_cov.py
"""

import h5py

# Adjust the path to the data directory.
FILE_PATH = 'clustered_data.h5'

if __name__ == '__main__':
    # Read the HDF5 file, containing the windowed waveform, corresponding
    # scattering covariances, and file names for five clusters of the Mars
    # dataset. The data structure is as follows:

    # file
    #    ├── '0'                        # Cluster 0.
    #        ├── 'waveform'             # 10 waveforms samples.
    #        ├── 'scat_cov'             # 10 Scattering covariances samples.
    #        ├── 'filename'             # The associated filenames (for ref).
    #    ├── '1'                        # Cluster 1.
    #        ├── 'waveform'             # 10 waveforms samples.
    #        ├── 'scat_cov'             # 10 Scattering covariances samples.
    #        ├── 'filename'             # The associated filenames (for ref).
    #    ⋮
    file = h5py.File(FILE_PATH, 'r')

    # Extract a cluster, e.g., the first cluster here. There are 5 clusters in
    # total.
    clustered_data = file['0']

    # Extract waveform and scattering covariances.
    # The shape of `waveform` is (10, 131072).
    waveform = clustered_data['waveform'][...]
    # The shape of `scat_cov` is (10, 348).
    scat_cov = clustered_data['scat_cov'][...]

    # Close file.
    file.close()
