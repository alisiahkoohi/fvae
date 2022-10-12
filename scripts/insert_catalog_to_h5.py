import argparse
import os

from facvae.utils import CatalogReader, datadir, catalogsdir

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
    parser.add_argument('--h5_dataset_name',
                        dest='h5_dataset_name',
                        type=str,
                        default='label',
                        help='dataset name to be created for h5 file')
    parser.add_argument('--catalog_filename',
                        dest='catalog_filename',
                        type=str,
                        default='events_InSIght.pkl',
                        help='catalog file to be written to h5 file')
    parser.add_argument('--target_column_name',
                        dest='target_column_name',
                        type=str,
                        default='type',
                        help='target catalog dataframe column name')
    parser.add_argument('--window_size',
                        dest='window_size',
                        type=int,
                        default=2**15,
                        help='Window size of raw waveforms')
    parser.add_argument('--n_workers',
                        dest='n_workers',
                        type=int,
                        default=16,
                        help='Number of workers to use for multiprocessing')
    args = parser.parse_args()

    catalog = CatalogReader(os.path.join(catalogsdir(), args.catalog_filename),
                            args.window_size)
    catalog.add_events_to_h5_file(
        os.path.join(MARS_SCAT_COV_PATH, args.h5_filename),
        args.h5_dataset_name, args.target_column_name, args.n_workers)
