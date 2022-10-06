""" Generate scattering covariance dataset from Mars waveforms. """
import argparse
import h5py
import obspy
import numpy as np
import os
from tqdm import tqdm

from scatcov.frontend import analyze
from facvae.utils import datadir, is_night_time_event, make_h5_file_name

# Path to Mars data directory.
MARS_PATH = datadir('mars')

# Datastream merge method.
MERGE_METHOD = 1
FILL_VALUE = 'interpolate'

# Windowing parameters.
STRIDE = 1 / 2
OFFSET = 0


def setup_hdf5_file(path, scat_cov_filename, window_size, max_win_num,
                    num_components, scat_cov_size):
    """
    Setting up an HDF5 file to write scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, scat_cov_filename)

    # HDF5 File.
    file = h5py.File(file_path, 'a')

    # Scattering covariance dataset of size `max_win_num x num_components x
    # scat_cov_size x 2`. The dataset will be resized at the end to reflect
    # the actual number of windows in the data. Chunks are chosen to be
    # efficient for extracting single-component scattering covariances for each
    # window.
    file.require_dataset('scat_cov',
                         (max_win_num, num_components, scat_cov_size, 2),
                         chunks=(1, 1, scat_cov_size, 2),
                         dtype=np.float32)
    # Raw waveforms dataset of size `max_win_num x window_size`. The dataset
    # will be resized at the end to reflect the actual number of windows in the
    # data. Chunks are chosen to be efficient for extracting a single-component
    # windowed waveform.
    file.require_dataset('waveform',
                         (max_win_num, num_components, window_size),
                         chunks=(1, 1, window_size),
                         dtype=np.float32)
    # Time interval dataset for recording the start and end time of each
    # windowed waveform.
    file.require_dataset('time_interval', (max_win_num, 2),
                         dtype=h5py.string_dtype())

    file.close()


def update_hdf5_file(path, scat_cov_filename, filename, file_idx, waveform,
                     scat_covariances, time_intervals):
    """
    Update the HDF5 file by writing new scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, scat_cov_filename)

    # HDF5 file.
    file = h5py.File(file_path, 'r+')

    # Write `scat_covariances` to the HDF5 file.
    file['scat_cov'][file_idx, ...] = scat_covariances
    # Write `waveform` to the HDF5 file.
    file['waveform'][file_idx, ...] = waveform
    # Write `time_intervals` to the HDF5 file.
    file['time_interval'][file_idx, ...] = time_intervals

    file.close()


def compute_scat_cov(args):

    # Path to raw data and directory for creating scattering dataset.
    waveform_path = datadir(os.path.join(MARS_PATH, 'waveforms'))
    scat_cov_path = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))
    raw_data_files = os.listdir(waveform_path)[:50]

    # Extract some properties of the data to setup HDF5 file.
    data_stream = obspy.read(os.path.join(waveform_path, raw_data_files[0]))
    # Number of components.
    num_components = len(data_stream)
    # Data sampling rate.
    sampling_rate = data_stream[0].meta.sampling_rate
    # Shape of the scattering covariance for one window size.
    scat_cov_size = analyze(data_stream[0][:args.window_size],
                            Q1=args.q1,
                            Q2=args.q2,
                            model_type=args.model_type,
                            cuda=args.cuda,
                            normalize=True,
                            nchunks=1).y.shape[1]
    # Max window number.
    max_win_num = int(
        len(raw_data_files) * 24 * 3600 * sampling_rate / args.window_size /
        STRIDE)

    # Setup HDF5 file.
    setup_hdf5_file(scat_cov_path, args.scat_cov_filename, args.window_size,
                    max_win_num, num_components, scat_cov_size)

    discarded_files = 0
    with tqdm(raw_data_files,
              unit='file',
              colour='#B5F2A9',
              dynamic_ncols=True) as pb:
        for file_idx, file in enumerate(pb):
            if os.stat(os.path.join(waveform_path, file)).st_size > 0:
                # Read data into a stream format.
                data_stream = obspy.read(os.path.join(waveform_path, file))

                # Only keep files that do not have gaps.
                if len(data_stream.get_gaps()) == 0:

                    # The following line although will not do interpolation —
                    # because there are not gaps — but will combine different
                    # streams into one.
                    data_stream = data_stream.merge(method=MERGE_METHOD,
                                                    fill_value=FILL_VALUE)

                    # Trimming all the components to the same length.
                    data_stream = data_stream.trim(
                        starttime=max(
                            [tr.meta.starttime for tr in data_stream]),
                        endtime=min([tr.meta.endtime for tr in data_stream]))

                    # Turn the trace to a batch of windowed data with size
                    # `window_size`.
                    windowed_data = list(
                        data_stream.slide(
                            (args.window_size - 1) /
                            data_stream[0].meta.sampling_rate,
                            args.window_size /
                            data_stream[0].meta.sampling_rate // STRIDE,
                            offset=OFFSET))

                    time_intervals = []
                    batched_window = []
                    for window in windowed_data:
                        time_intervals.append(
                            (window[0].meta.starttime, window[0].meta.endtime))
                        batched_window.append(
                            np.stack([tr.data for tr in window]))
                    batched_window = np.stack(batched_window)
                    batched_window = batched_window.reshape(
                        len(windowed_data) * num_components, args.window_size)
                    from IPython import embed; embed()
                    # Compute scattering covariance.
                    y = analyze(batched_window,
                                Q1=args.q1,
                                Q2=args.q2,
                                model_type=args.model_type,
                                cuda=args.cuda,
                                normalize=True,
                                nchunks=args.nchunks).y

                    batched_window = batched_window.reshape(
                        len(windowed_data), num_components, args.window_size)
                    y = y.reshape(len(windowed_data), num_components,
                                  scat_cov_size).numpy()

                    # Compute the average of the scattering covariance
                    # over the time axis.
                    for b in range(len(windowed_data)):

                        # CASE 1: keep real and imag parts by considering
                        # it as different real coefficients
                        scat_covariances = np.stack(
                            [y[b, ...].real, y[b, ...].imag],
                            axis=-1).astype(np.float32)

                        # CASE 2: only keeps the modulus of the scattering
                        # covariance, hence discarding time asymmetry info
                        # scat_covariances = np.abs(cplx.to_np(y))

                        # CASE 3: only keep the phase, which looks at time
                        # asymmetry in the data y =
                        # RX.reduce(m_type=['m01', 'm11'], re=False).y[0,
                        # :, 0].numpy() y_phase = np.angle(y)
                        # y_phase[np.abs(y) < 1e-2] = 0.0  # rules phase
                        # instability scat_covariances = y_phase

                        filename = file + '_' + str(b)
                        event_start, event_end = time_intervals[b]
                        if args.use_day_data or is_night_time_event(
                                event_start, event_end):
                            update_hdf5_file(scat_cov_path,
                                             args.scat_cov_filename, filename,
                                             file_idx, batched_window[b, ...],
                                             scat_covariances,
                                             time_intervals[b])
                        pb.set_postfix({
                            'shape':
                            scat_covariances.shape,
                            'discarded':
                            f'{discarded_files/(file_idx + 1):.4f}'
                        })

                else:
                    discarded_files += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--window_size',
                        dest='window_size',
                        type=int,
                        default=2**12,
                        help='Window size of raw waveforms')
    parser.add_argument('--q1', dest='q1', type=int, default=2)
    parser.add_argument('--q2', dest='q2', type=int, default=4)
    parser.add_argument('--cuda',
                        dest='cuda',
                        type=int,
                        default=1,
                        help='set to 1 for running on GPU, 0 for CPU')
    parser.add_argument('--nchunks',
                        dest='nchunks',
                        type=int,
                        default=16,
                        help='set higher for less memory usage.')
    parser.add_argument('--use_day_data',
                        dest='use_day_data',
                        type=int,
                        default=1,
                        help='set to 0 for extracting only night time data')
    parser.add_argument('--model_type',
                        dest='model_type',
                        type=str,
                        default='cov',
                        help='model types to be extracted')
    parser.add_argument('--filename',
                        dest='filename',
                        type=str,
                        default='scat_covs_3c',
                        help='filename prefix to be created')
    args = parser.parse_args()

    args.scat_cov_filename = make_h5_file_name(args)

    compute_scat_cov(args)
