""" Generate scattering covariance dataset from Cascadia waveforms. """
import argparse
import obspy
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

from facvae.scat_cov.frontend import analyze, cplx
from facvae.utils import datadir

CASCADIA_PATH = datadir('cascadia')
MARS_PATH = datadir('mars')


def windows(x, window_size, stride, offset):
    """ Separate x into windows on last axis, discard any residual.
    """
    num_window = (x.shape[-1] - window_size - offset) // stride + 1
    windowed_x = np.stack([
        x[..., i * stride + offset:window_size + i * stride + offset]
        for i in range(num_window)
    ], -2)  # (C) x nb_w x w

    return windowed_x


def compute_scat_cov(window_size, num_oct, cuda, dataset):

    if dataset == 'cascadia':
        waveform_path = datadir(os.path.join(CASCADIA_PATH, 'waveform'))
        scat_cov_path = datadir(os.path.join(CASCADIA_PATH, 'scat_cov'))
        raw_data_files = os.listdir(waveform_path)
    elif dataset == 'mars':
        waveform_path = datadir(os.path.join(MARS_PATH, 'waveform'))
        scat_cov_path = datadir(os.path.join(MARS_PATH, 'scat_cov'))
        raw_data_files = os.listdir(waveform_path)

    discarded_files = 0
    with tqdm(raw_data_files,
              unit='file',
              colour='#B5F2A9',
              dynamic_ncols=True) as pb:
        for i, file in enumerate(pb):
            # Read data into a stream format.
            data_stream = obspy.read(os.path.join(waveform_path, file))

            # Only keep files that do not have gaps.
            if len(data_stream.get_gaps()) == 0:
                # Merge the two traces in the stream and extract data.
                trace = data_stream.merge(method=1,
                                          fill_value="interpolate")[0]
                # Some preprocessing.
                # TODO: ask Rudy where do these numbers come from.
                if dataset == 'cascadia':
                    trace.filter('highpass', freq=1.0)
                    trace.filter('lowpass', freq=10.0)
                    trace.taper(0.01)
                    trace = trace.data[50000:-50000]
                else:
                    trace = trace.data

                # Filter out smaller than `window_size` data.
                # TODO: Decide on a more concrete way of choosing window size.
                if trace.size >= window_size:
                    # Turn the trace to a batch of windowed data with size
                    # `window_size`.
                    windowed_trace = windows(trace,
                                             window_size=window_size,
                                             stride=window_size // 2,
                                             offset=0)
                    # Compute scattering covariance.
                    RX = analyze(windowed_trace,
                                 J=num_oct,
                                 moments='cov',
                                 cuda=cuda,
                                 nchunks=windowed_trace.shape[0]
                                 )  # reduce nchunks to accelerate
                    # for r in range(windowed_trace.shape[0]):
                    r = 0
                    y = cplx.to_np(RX.select(n1=r))
                    y[np.abs(y) < 0.001] = 0.0
                    scat_covariances = np.angle(y)  # only take the phase
                    scat_covariances = scat_covariances.astype(np.float32)
                    # scat_covariances = RX.select(n1=r).ravel().detach().numpy()

                    fname = Path(file).stem + f'_w{r}' + '.npy'
                    np.save(os.path.join(scat_cov_path, fname),
                            scat_covariances)
                    pb.set_postfix({
                        'shape':
                        scat_covariances.shape,
                        'discarded':
                        f'{discarded_files/(i + 1):.4f}'
                    })
            else:
                discarded_files += 1


def probe_mars_data():
    waveform_path = datadir(os.path.join(MARS_PATH, 'waveform'))
    raw_data_files = os.listdir(waveform_path)

    sampling_rates = []
    for file in raw_data_files:
        # Read data into a stream format.
        data_stream = obspy.read(os.path.join(waveform_path, file))
        print(file)
        data_stream.print_gaps()


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
    parser.add_argument('--dataset',
                        dest='dataset',
                        type=str,
                        default='cascadia',
                        help='cascadia or mars')
    args = parser.parse_args()

    compute_scat_cov(args.window_size, args.num_oct, args.cuda, args.dataset)
    # probe_mars_data()
