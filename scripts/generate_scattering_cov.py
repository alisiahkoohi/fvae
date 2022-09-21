""" Generate scattering covariance dataset from Mars waveforms. """
import argparse
import h5py
import obspy
import numpy as np
import os
from tqdm import tqdm

from scatcov.frontend import analyze, cplx
from scatcov.utils import to_numpy
from facvae.utils import datadir

MARS_PATH = datadir('mars')
SCAT_COV_FILENAME = 'scat_covs_UVW_raw_q1-2_q2-4.h5'


def windows(x, window_size, stride, offset):
    """ Separate x into windows on last axis, discard any residual.
    """
    num_window = (x.shape[-1] - window_size - offset) // stride + 1
    windowed_x = np.stack([
        x[..., i * stride + offset:window_size + i * stride + offset]
        for i in range(num_window)
    ], -2)  # (C) x nb_w x w

    return windowed_x


def setup_hdf5_file(path):
    """
    Setting up an HDF5 file to write scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, SCAT_COV_FILENAME)

    # Horizons file
    file = h5py.File(file_path, 'a')
    file.close()


def update_hdf5_file(path, filename, batch, waveform, scat_covariances):
    """
    Update the HDF5 file by writing new scattering covariances.
    """
    # Path to the file.
    file_path = os.path.join(path, SCAT_COV_FILENAME)

    # HDF5 file.
    file = h5py.File(file_path, 'r+')

    # Group for the given file.
    file_group = file.create_group(filename + '_' + str(batch))

    # HDF5 dataset for waveform.
    file_group.create_dataset('waveform', data=waveform, dtype=np.float32)

    # HDF5 dataset for waveform.
    file_group.create_dataset('scat_cov',
                              data=scat_covariances,
                              dtype=np.float32)
    file.close()


def compute_scat_cov(window_size, num_oct, cuda):

    waveform_path = datadir(os.path.join(MARS_PATH, 'waveforms_UVW_raw'))
    scat_cov_path = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))
    raw_data_files = os.listdir(waveform_path)

    setup_hdf5_file(scat_cov_path)
    discarded_files = 0
    with tqdm(raw_data_files,
              unit='file',
              colour='#B5F2A9',
              dynamic_ncols=True) as pb:
        for i, file in enumerate(pb):
            if os.stat(os.path.join(waveform_path, file)).st_size > 0:
                # Read data into a stream format.
                data_stream = obspy.read(os.path.join(waveform_path, file))

                # Only keep files that do not have gaps.
                if len(data_stream.get_gaps()) == 0:

                    # The following line although will not do interpolation—because
                    # there are not gaps—but will combine different streams into
                    # one.
                    trace = data_stream.merge(method=1,
                                              fill_value="interpolate")[0]
                    trace = trace.data

                    # Filter out smaller than `window_size` data.
                    # TODO: Decide on a more concrete way of choosing window size.
                    # 2**17 comes from previous experiments. (may want to reduce it
                    # for mars quakes in the range of 30 minutes)
                    if trace.size >= window_size:
                        # Turn the trace to a batch of windowed data with size
                        # `window_size`.
                        windowed_trace = windows(trace,
                                                 window_size=window_size,
                                                 stride=window_size // 2,
                                                 offset=0)

                        # Compute scattering covariance. RX is a DescribedTensor.
                        # RX.y is a tensor of size B x nb_coeff x T x 2

                        # RX.info is a dataframe with nb_coeff rows that describes
                        # each RX.y[:, i_coeff, :, :] for 0 <= i_coeff < nb_coeff

                        # Here, the batch dimension (1st dimension) corresponds to
                        # the different windows

                        RX = analyze(windowed_trace,
                                     J=num_oct,
                                     Q1=2,
                                     Q2=4,
                                     moments='cov',
                                     cuda=cuda,
                                     normalize=True,
                                     nchunks=windowed_trace.shape[0] *
                                     2)  # reduce nchunks to accelerate
                        for b in range(windowed_trace.shape[0]):

                            # b = 0  # for test only: choose the first window to compute scattering covariance
                            y = RX.y[b, :, 0, :]

                            # y_phase = np.angle(y)
                            # y_phase[np.abs(y) < 0.001] = 0.0  # rule phase instability, the threshold must be adapted

                            # CASE 1: keep real and imag parts by considering it as different real coefficients
                            scat_covariances = to_numpy(y).ravel()
                            # from IPython import embed; embed()
                            # CASE 2: only keeps the modulus of the scattering covariance, hence discarding time asymmetry info
                            # scat_covariances = np.abs(cplx.to_np(y))

                            # CASE 3: only keep the phase, which looks at time asymmetry in the data
                            # scat_covariances = y_phase

                            update_hdf5_file(scat_cov_path, file, b,
                                             windowed_trace[b, ...],
                                             scat_covariances)
                            # fname = Path(file).stem + f'_w{b}' + '.npy'
                            # np.save(os.path.join(scat_cov_path, fname),
                            #         scat_covariances)
                            pb.set_postfix({
                                'shape':
                                scat_covariances.shape,
                                'discarded':
                                f'{discarded_files/(i + 1):.4f}'
                            })

                else:
                    discarded_files += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--window_size',
                        dest='window_size',
                        type=int,
                        default=2**17,
                        help='Window size of raw waveforms')
    parser.add_argument('--num_oct',
                        dest='num_oct',
                        type=int,
                        default=8,
                        help='Number of octaves in the scattering transform')
    parser.add_argument('--cuda',
                        dest='cuda',
                        type=int,
                        default=1,
                        help='set to 1 for running on GPU, 0 for CPU')
    args = parser.parse_args()

    compute_scat_cov(args.window_size, args.num_oct, args.cuda)
