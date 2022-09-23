import os

from facvae.utils import CatalogReader, datadir

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))
SCAT_COV_FILENAME = 'scat_covs_UVW_raw_q1-2_q2-4_semi-labeled.h5'

if __name__ == "__main__":
    catalog = CatalogReader()
    catalog.add_labels_to_h5_file(
        os.path.join(MARS_SCAT_COV_PATH, SCAT_COV_FILENAME))
