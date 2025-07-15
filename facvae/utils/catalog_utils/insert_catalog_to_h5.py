import argparse
import os

from facvae.utils import CatalogReader, datadir, catalogsdir

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--h5_filename',
        dest='h5_filename',
        type=str,
        default=
        'pyramid_full-mission_window_size-65536_q-1-1_j-8-8_use_day_data-1_avgpool_base-4_avgpool_exp-5-6-7-8_model_type-scat+cov_filter_key-true_backup.h5',
        help='h5 file to add events to')
    parser.add_argument(
        '--h5_dataset_name',
        dest='h5_dataset_name',
        type=str,
        default='labels',
        help='dataset name to be created for h5 file, labels or pressure')
    parser.add_argument(
        '--catalog_filename',
        dest='catalog_filename',
        type=str,
        default='events_InSIght_v14.pkl',
        help='events_InSIght_v14.pkl or pressure_drops_InSIght.pkl')
    parser.add_argument(
        '--target_column_name',
        dest='target_column_name',
        type=str,
        default='type',
        help='target catalog dataframe column name, type or drop')
    parser.add_argument('--window_size',
                        dest='window_size',
                        type=int,
                        default=1024,
                        help='Window size of raw waveforms')
    parser.add_argument('--n_workers',
                        dest='n_workers',
                        type=int,
                        default=40,
                        help='Number of workers to use for multiprocessing')
    args = parser.parse_args()

    catalog = CatalogReader(os.path.join(catalogsdir(), args.catalog_filename),
                            args.window_size)
    catalog.add_events_to_h5_file(
        os.path.join(MARS_SCAT_COV_PATH, args.h5_filename),
        args.h5_dataset_name, args.target_column_name, args.n_workers)
