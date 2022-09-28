import argparse
import os

from facvae.utils import CatalogReader, datadir

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--h5_filename',
                        dest='h5_filename',
                        type=str,
                        default='scat_covs_w-size-2e15_q1-2_q2-4_nighttime.h5',
                        help='h5 file to add events to')
    parser.add_argument('--window_size',
                        dest='window_size',
                        type=int,
                        default=2**15,
                        help='Window size of raw waveforms')
    args = parser.parse_args()

    catalog = CatalogReader(window_size=args.window_size)
    catalog.add_labels_to_h5_file(
        os.path.join(MARS_SCAT_COV_PATH, args.h5_filename))
