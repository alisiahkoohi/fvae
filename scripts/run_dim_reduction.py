import os
import sys
from facvae.utils import datadir, MarsDataset

MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))
H5_FILE_PATH = os.path.join(MARS_SCAT_COV_PATH, sys.argv[1])

if __name__ == "__main__":
    dataset = MarsDataset(H5_FILE_PATH,
                          0.99,
                          data_types=['scat_cov'],
                          load_to_memory=False,
                          normalize_data=False)
    dataset.pca_dim_reduction()
