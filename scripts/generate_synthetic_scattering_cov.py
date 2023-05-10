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
                          make_h5_file_name, Pooling)

# Path to Mars data directory.
DATA_PATH = datadir('synthetic_dataset')

# Scattering covariance generation parameters.
SCATCOV_CONFIG_FILE = 'generate_synthetic_scatcov.json'

LARGE_SCALE_LENGTH = 4096
MID_SCALE_LENGTH = 1024
FINE_SCALE_LENGTH = 256

LARGE_SCALE_WN_PATH = os.path.join(DATA_PATH, 'scale_LARGE_white_noise')
LARGE_SCALE_MRW_PATH = os.path.join(DATA_PATH, 'scale_LARGE_mrw')

MID_SCALE_TURB_PATH = os.path.join(DATA_PATH, 'scale_MEDIUM_turbulence')

FINE_SCALE_SYM_PATH = os.path.join(DATA_PATH, 'scale_FINE_symmetrical_exp')
FINE_SCALE_ASYM_PATH = os.path.join(DATA_PATH, 'scale_FINE_asymmetrical_exp')

R = 100
NUM_COMPONENTS = 1
NUM_TIME_SERIES = 100

LARGE_SCALE_AMP = 1.0
MID_SCALE_AMP = 2.0
FINE_SCALE_AMP = 4.0

OFFSET = 0


def load_data(dpath, n_events):
    """ Load n_events data from source stored under directory dpath. """
    n_files = int(np.ceil(n_events / 512))  # each file contains 512 events
    fnames = os.listdir(dpath)

    if n_files > len(fnames):
        fnames = [fnames[i % len(fnames)] for i in range(n_files)]
    else:
        idx = list(
            np.random.choice(np.arange(len(fnames), dtype=int),
                             size=n_files,
                             replace=False))
        fnames = [fnames[i] for i in idx]

    xl = [np.load(os.path.join(dpath, f)) for f in fnames]
    x = np.concatenate(xl)

    return x[:n_events, :]


def plateau_function(size):
    """ A plateau function used for smooth interpolation. """
    ts = np.arange(size)

    sigma = size / 64
    gaussian = np.exp(-(ts - size / 32)**2 / (2 * sigma**2))

    plateau = np.cumsum(gaussian)
    plateau = plateau * plateau[::-1]
    plateau = plateau / plateau.max()

    return plateau


def synthesize_time_series():
    # load white noise
    x_wn = load_data(LARGE_SCALE_WN_PATH, n_events=R)

    # load mrw
    x_mrw = load_data(LARGE_SCALE_MRW_PATH, n_events=R)

    # mix the 2 with smooth transition
    smooth_factor = np.zeros_like(x_wn)
    smooth_factor[::2, :] = plateau_function(x_wn.shape[-1])

    x_large_scale = smooth_factor * x_mrw + (1 - smooth_factor) * x_wn

    # load turbulent events
    x_turb = load_data(MID_SCALE_TURB_PATH, n_events=R)

    # choose random places for these turbulent events
    random_positions = np.random.randint(low=0,
                                         high=LARGE_SCALE_LENGTH -
                                         MID_SCALE_LENGTH,
                                         size=(R, ))
    random_idces = np.arange(MID_SCALE_LENGTH)[
        None, :] + random_positions[:, None]

    x_medium_scales = np.zeros((R, LARGE_SCALE_LENGTH))
    for i, idx in enumerate(random_idces):
        x_medium_scales[
            i, idx] = x_turb[i, :] * plateau_function(MID_SCALE_LENGTH)

    # load symmetrical_events
    x_sym_exp = load_data(FINE_SCALE_SYM_PATH, n_events=4 * R)

    # position them randomly
    random_window = np.random.randint(0, R, size=4 * R)
    random_position_in_window = np.random.randint(0,
                                                  LARGE_SCALE_LENGTH -
                                                  FINE_SCALE_LENGTH,
                                                  size=4 * R)

    x_fine_scales = np.zeros((R, LARGE_SCALE_LENGTH))

    for i, (iw,
            ipos) in enumerate(zip(random_window, random_position_in_window)):
        x_fine_scales[iw, ipos:ipos + FINE_SCALE_LENGTH] = x_sym_exp[i, :]

    # load asymmetrical_events
    x_asym_exp = load_data(FINE_SCALE_ASYM_PATH, n_events=4 * R)

    # position them randomly
    random_window = np.random.randint(0, R, size=4 * R)
    random_position_in_window = np.random.randint(0,
                                                  LARGE_SCALE_LENGTH -
                                                  FINE_SCALE_LENGTH,
                                                  size=4 * R)

    for i, (iw,
            ipos) in enumerate(zip(random_window, random_position_in_window)):
        x_fine_scales[iw, ipos:ipos + FINE_SCALE_LENGTH] = x_asym_exp[i, :]

    x_large_scale = LARGE_SCALE_AMP * x_large_scale
    x_medium_scales = MID_SCALE_AMP * x_medium_scales
    x_fine_scales = FINE_SCALE_AMP * x_fine_scales

    # sum the 3 components with different amplitudes
    x_synthetic = x_large_scale + x_medium_scales + x_fine_scales

    return x_large_scale, x_medium_scales, x_fine_scales, x_synthetic


def setup_hdf5_file(
    path,
    scat_cov_filename,
    window_size,
    max_win_num,
    num_components,
    scatcov_size_list,
    subwindow_size_list,
):
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
    file.require_dataset('waveform_large_scale',
                         (max_win_num, num_components, window_size),
                         chunks=(1, 1, window_size),
                         dtype=np.float32)
    file.require_dataset('waveform_mid_scale',
                         (max_win_num, num_components, window_size),
                         chunks=(1, 1, window_size),
                         dtype=np.float32)
    file.require_dataset('waveform_fine_scale',
                         (max_win_num, num_components, window_size),
                         chunks=(1, 1, window_size),
                         dtype=np.float32)

    file.close()


def update_hdf5_file(
    path,
    scat_cov_filename,
    window_idx,
    waveform,
    waveform_large_scale,
    waveform_mid_scale,
    waveform_fine_scale,
    scat_covariances_list,
    subwindow_size_list,
):
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
    file['waveform'][window_idx, ...] = waveform
    file['waveform_large_scale'][window_idx, ...] = waveform_large_scale
    file['waveform_mid_scale'][window_idx, ...] = waveform_mid_scale
    file['waveform_fine_scale'][window_idx, ...] = waveform_fine_scale

    file.close()


def finalize_dataset_size(path, scat_cov_filename, num_windows):
    # Path to the file.
    file_path = os.path.join(path, scat_cov_filename)

    # HDF5 file.
    file = h5py.File(file_path, 'r+')

    for key in file.keys():
        if key not in ['scat_cov']:
            file[key].resize(num_windows, axis=0)
        elif key == 'scat_cov':
            for dset_name in file[key].keys():
                file[key][dset_name].resize(num_windows, axis=0)

    file.close()


def window_data(args, data_stream, stride):
    data_stream = obspy.Trace(data=data_stream.reshape(-1))

    # Turn the trace to a batch of windowed data with size
    # `window_size`.
    data_stream = data_stream.slide((args.window_size - 1),
                                    args.window_size * stride,
                                    offset=OFFSET)

    batched_window = []
    for window in data_stream:
        batched_window.append(np.stack([tr.data for tr in window]))

    window_num = len(batched_window)

    batched_window = np.stack(batched_window)
    batched_window = batched_window.reshape(
        window_num * NUM_COMPONENTS, args.window_size).astype(np.float32)

    return batched_window


def compute_scat_cov(args):
    # Path to directory for creating scattering dataset.
    scat_cov_path = datadir(os.path.join(DATA_PATH, 'scat_covs_h5'))

    # Extract some properties of the data to setup HDF5 file.
    data_stream = synthesize_time_series()

    # Shape of the scattering covariance for one window size.
    print('Setting up HDF5 file for various time averaging kernel sizes ...')

    y = analyze(data_stream[-1][0, :args.window_size].astype(np.float32),
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
    max_win_num = int(NUM_TIME_SERIES * R * LARGE_SCALE_LENGTH /
                      args.window_size / stride)

    # Setup HDF5 file.
    setup_hdf5_file(scat_cov_path, args.scat_cov_filename, args.window_size,
                    max_win_num, NUM_COMPONENTS, scatcov_size_list,
                    subwindow_size_list)

    num_windows = 0
    for idx in tqdm(range(NUM_TIME_SERIES)):

        (x_large_scale, x_medium_scales, x_fine_scales,
         data_stream) = synthesize_time_series()

        data_stream = window_data(args, data_stream, stride)
        x_large_scale = window_data(args, x_large_scale, stride)
        x_medium_scales = window_data(args, x_medium_scales, stride)
        x_fine_scales = window_data(args, x_fine_scales, stride)

        window_num = len(data_stream)

        data_stream = np.stack(data_stream)
        data_stream = data_stream.reshape(
            window_num * NUM_COMPONENTS, args.window_size).astype(np.float32)

        y_list = []

        # Compute scattering covariance.
        y = analyze(data_stream,
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
                    nchunks=args.nchunks).y

        for avg_pool, scatcov_size, subwindow_size in tqdm(
                zip(avg_pool_list, scatcov_size_list, subwindow_size_list)):
            y_ = avg_pool(y)
            y_ = y_.reshape(window_num, NUM_COMPONENTS, *scatcov_size)
            y_ = torch.permute(y_, (0, 3, 1, 2)).numpy()
            y_ = y_[:, -1, :, :]
            y_list.append(y_)

        data_stream = data_stream.reshape(window_num, NUM_COMPONENTS,
                                                args.window_size)

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

            update_hdf5_file(
                scat_cov_path,
                args.scat_cov_filename,
                num_windows,
                data_stream[b, ...],
                x_large_scale[b, :],
                x_medium_scales[b, :],
                x_fine_scales[b, :],
                scat_covariances_list,
                subwindow_size_list,
            )
            num_windows += 1

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
    args.scat_cov_filename = make_h5_file_name(args)

    if args.window_size != args.avgpool_base**max(args.avgpool_exp):
        raise ValueError("window_size and largest average pool kernel size "
                         "must match")

    compute_scat_cov(args)
