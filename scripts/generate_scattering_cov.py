""" Generate scattering covariance dataset from Mars waveforms. """
import argparse
import h5py
import obspy
import numpy as np
import os
from tqdm import tqdm
import torch

from scatcov.frontend import analyze
from facvae.utils import (configsdir, datadir, parse_input_args, read_config,
                          is_night_time_event, make_h5_file_name, Pooling)

# Path to Mars data directory.
MARS_PATH = datadir('mars')

# Scattering covariance generation parameters.
SCATCOV_CONFIG_FILE = 'generate_scatcov.json'

# Datastream merge method.
MERGE_METHOD = 1
FILL_VALUE = 'interpolate'

# Windowing parameters.
OFFSET = 0


def setup_hdf5_file(path, scat_cov_filename, window_size, max_win_num,
                    num_components, scatcov_size_list, subwindow_size_list,
                    filter_key):
    """
    Setting up an HDF5 file to write scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, scat_cov_filename)

    # HDF5 File.
    file = h5py.File(file_path, 'a')

    file.require_dataset('filter_key', (len(filter_key), ),
                         data=filter_key,
                         chunks=True,
                         dtype=h5py.string_dtype())
    # Scattering covariance dataset of size `max_win_num x num_components x
    # scat_cov_size x 2`. The dataset will be resized at the end to reflect
    # the actual number of windows in the data. Chunks are chosen to be
    # efficient for extracting single-component scattering covariances for each
    # window.
    scatcov_group = file.require_group('scat_cov')
    for subwindow_size, scatcov_size in zip(subwindow_size_list,
                                            scatcov_size_list):
        scatcov_group.require_dataset(
            str(subwindow_size),
            (max_win_num, num_components, scatcov_size[0], 2),
            chunks=(1, num_components, scatcov_size[0], 2),
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
                         chunks=True,
                         dtype=h5py.string_dtype())

    file.require_dataset('filename', (max_win_num, ),
                         chunks=True,
                         dtype=h5py.string_dtype())

    file.close()


def update_hdf5_file(path, scat_cov_filename, filename, window_idx, waveform,
                     scat_covariances_list, subwindow_size_list,
                     time_intervals):
    """
    Update the HDF5 file by writing new scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, scat_cov_filename)

    # HDF5 file.
    file = h5py.File(file_path, 'r+')

    # Write `scat_covariances` to the HDF5 file.
    for subwindow_size, scat_covariances in zip(subwindow_size_list,
                                                scat_covariances_list):
        file['scat_cov'][str(subwindow_size)][window_idx,
                                              ...] = scat_covariances
    # Write `waveform` to the HDF5 file.
    # file['waveform'][window_idx, ...] = waveform
    # Write `time_intervals` to the HDF5 file.
    file['time_interval'][window_idx, ...] = [
        str(event_time) for event_time in time_intervals
    ]
    file['filename'][window_idx] = str(filename)

    file.close()


def finalize_dataset_size(path, scat_cov_filename, num_windows):
    # Path to the file.
    file_path = os.path.join(path, scat_cov_filename)

    # HDF5 file.
    file = h5py.File(file_path, 'r+')

    for key in file.keys():
        if key not in ['filter_key', 'scat_cov']:
            file[key].resize(num_windows, axis=0)
        elif key == 'scat_cov':
            for dset_name in file[key].keys():
                file[key][dset_name].resize(num_windows, axis=0)

    file.close()


def compute_scat_cov(args):

    # Path to raw data and directory for creating scattering dataset.
    waveform_path = datadir(os.path.join(MARS_PATH, 'waveforms_UVW_raw'))
    scat_cov_path = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))
    raw_data_files = os.listdir(waveform_path)

    if args.filter_key:
        filtered_filenames = []
        for filter_key in args.filter_key:
            filenames = list(filter(lambda k: filter_key in k, raw_data_files))
            filtered_filenames.extend(filenames)
        raw_data_files = filtered_filenames

    # Extract some properties of the data to setup HDF5 file.
    data_stream = obspy.read(os.path.join(waveform_path, raw_data_files[0]))
    # Number of components.
    num_components = len(data_stream)
    # Data sampling rate.
    sampling_rate = data_stream[0].meta.sampling_rate

    # Shape of the scattering covariance for one window size.
    print('Setting up HDF5 file for various time averaging kernel sizes ...')

    y = analyze(data_stream[0][:args.window_size].astype(np.float32),
                Q=args.q,
                J=args.j,
                r=len(args.q),
                keep_ps=True,
                model_type=args.model_type,
                cuda=args.cuda,
                normalize='each_ps',
                estim_operator=Pooling(
                    kernel_size=args.avgpool_base**min(args.avgpool_exp)),
                qs=[1.0] if args.model_type == 'scat' else None,
                nchunks=1).y

    scatcov_size_list = []
    subwindow_size_list = []
    avg_pool_list = []
    for avgpool_exp in tqdm(args.avgpool_exp):
        subwindow_size_list.append(args.avgpool_base**(avgpool_exp))
        avg_pool_list.append(
            Pooling(kernel_size=args.avgpool_base**(avgpool_exp -
                                                    min(args.avgpool_exp))))
        scatcov_size_list.append(avg_pool_list[-1](y).shape[1:])

    # Stride should be equal to the proportion to the smallest window size.
    stride = args.avgpool_base**(min(args.avgpool_exp) - max(args.avgpool_exp))

    # Max window number.
    max_win_num = int(
        len(raw_data_files) * 24 * 3600 * sampling_rate / args.window_size /
        stride)

    # Setup HDF5 file.
    setup_hdf5_file(scat_cov_path, args.scat_cov_filename, args.window_size,
                    max_win_num, num_components, scatcov_size_list,
                    subwindow_size_list, args.filter_key)

    discarded_files = 0
    num_windows = 0
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

                    if args.detrend:
                        data_stream = data_stream.detrend(type='spline',
                                                          order=2,
                                                          dspline=2000,
                                                          plot=False)

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
                            data_stream[0].meta.sampling_rate * stride,
                            offset=OFFSET))

                    time_intervals = []
                    batched_window = []
                    for window in windowed_data:
                        if args.use_day_data or is_night_time_event(
                                window[0].meta.starttime,
                                window[0].meta.endtime):
                            time_intervals.append((window[0].meta.starttime,
                                                   window[0].meta.endtime))
                            batched_window.append(
                                np.stack([tr.data for tr in window]))

                    window_num = len(batched_window)
                    if window_num > 0:
                        batched_window = np.stack(batched_window)
                        batched_window = batched_window.reshape(
                            window_num * num_components,
                            args.window_size).astype(np.float32)

                        y_list = []

                        # Compute scattering covariance.
                        y = analyze(
                            batched_window,
                            Q=args.q,
                            J=args.j,
                            r=len(args.q),
                            keep_ps=True,
                            model_type=args.model_type,
                            cuda=args.cuda,
                            normalize='each_ps',
                            estim_operator=Pooling(
                                kernel_size=args.avgpool_base**min(
                                    args.avgpool_exp)),
                            qs=[1.0] if args.model_type == 'scat' else None,
                            nchunks=args.nchunks).y

                        for avg_pool, scatcov_size, subwindow_size in tqdm(
                                zip(avg_pool_list, scatcov_size_list,
                                    subwindow_size_list)):
                            y_ = avg_pool(y)
                            y_ = y_.reshape(window_num, num_components,
                                            *scatcov_size)
                            y_ = torch.permute(y_, (0, 3, 1, 2)).numpy()
                            y_ = y_[:, -1, :, :]
                            y_list.append(y_)

                        batched_window = batched_window.reshape(
                            window_num, num_components, args.window_size)

                        for b in range(window_num):
                            # CASE 1: keep real and imag parts by considering
                            # it as different real coefficients
                            scat_covariances_list = []
                            for avgpool_idx in range(len(avg_pool_list)):
                                scat_covariances_list.append(
                                    np.stack([
                                        y_list[avgpool_idx][b, ...].real,
                                        y_list[avgpool_idx][b, ...].imag
                                    ],
                                             axis=-1).astype(np.float32))

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
                            update_hdf5_file(scat_cov_path,
                                             args.scat_cov_filename, filename,
                                             num_windows, batched_window[b,
                                                                         ...],
                                             scat_covariances_list,
                                             subwindow_size_list,
                                             time_intervals[b])
                            num_windows += 1
                            pb.set_postfix({
                                'discarded':
                                f'{discarded_files/(file_idx + 1):.4f}'
                            })

                else:
                    discarded_files += 1

    finalize_dataset_size(scat_cov_path, args.scat_cov_filename, num_windows)


if __name__ == "__main__":
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), SCATCOV_CONFIG_FILE))
    args = parse_input_args(args)
    args.q = [int(j) for j in args.q.replace(' ', '').split(',')]
    args.j = [int(j) for j in args.j.replace(' ', '').split(',')]
    args.avgpool_exp = [
        int(j) for j in args.avgpool_exp.replace(' ', '').split(',')
    ]
    args.filter_key = args.filter_key.replace(' ', '').split(',')
    args.scat_cov_filename = make_h5_file_name(args)

    if args.window_size != args.avgpool_base**max(args.avgpool_exp):
        raise ValueError("window_size and largest average pool kernel size "
                         "must match")

    compute_scat_cov(args)
