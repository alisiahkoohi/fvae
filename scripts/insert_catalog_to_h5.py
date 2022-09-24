import argparse
import os

from facvae.utils import CatalogReader, datadir

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))
SCAT_COV_FILENAME = 'scat_covs_UVW_raw_q1-2_q2-4_nightime.h5'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--h5_filename',
                        dest='h5_filename',
                        type=str,
                        default='scat_covs_UVW_raw_q1-2_q2-4_nightime.h5',
                        help='h5 file to add events to')
    args = parser.parse_args()

    catalog = CatalogReader()
    catalog.add_labels_to_h5_file(
        os.path.join(MARS_SCAT_COV_PATH, SCAT_COV_FILENAME))
