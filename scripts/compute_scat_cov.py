""" Generate scattering covariance dataset from cascadia waveforms. """
import obspy
from mpire import WorkerPool
from pathlib import Path
import numpy as np
import os

from scatcov.frontend import analyze, cplx
from facvae.utils import datadir

def windows(x, w, s, offset):
    """ Separate x into windows on last axis, discard any residual. """
    nb_w = (x.shape[-1] - w - offset) // s + 1
    xw = np.stack([x[..., k * s + offset: w + k * s + offset] for k in range(nb_w)], -2)  # (C) x nb_w x w

    return xw


def worker(files, gpu, saving_path, w, J):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for i_f, f in enumerate(files):
        st = obspy.read(str(f))
        st.merge(method=1, fill_value="interpolate")
        tr = st[0]
        tr.filter('highpass', freq=1.0)
        tr.filter('lowpass', freq=10.0)
        tr.taper(0.01)
        x = tr.data[50000:-50000]

        if x.size < w:
            continue

        x_w = windows(x, w=w, s=w//2, offset=0)
        RX = analyze(x_w, J=J, moments='cov', cuda=True, nchunks=x_w.shape[0])  # reduce nchunks to accelerate

        for r in range(x_w.shape[0]):
            y = cplx.to_np(RX.select(n1=r))
            y[np.abs(y) < 0.001] = 0.0
            scat_covariances = np.angle(y)  # only take the phase
            # scat_covariances = RX.select(n1=r).ravel().detach().numpy()
            if not np.prod(scat_covariances.shape) == 0:
                fname = f.stem + f'_w{r}' + '.npy'
                np.save(saving_path / fname, scat_covariances)


if __name__ == "__main__":
    # for that particular case we only kept the phase of scattering covariance to see if a VAE can be learnt to separate
    # events based only on the phase, which is related to the time asymmetry in the data
    w = 2 ** 17  # 2 ** 17
    J = 8
    gpus = [0]
    dirname = 'test_window_17_phase'

    n_gpus = len(gpus)

    # path to the waveforms
    CASCADIA_PATH = Path(datadir('cascadia'))
    files = list((CASCADIA_PATH / 'waveform').iterdir())
    files_chunks = np.array_split(files, n_gpus*20)
    saving_path = Path(datadir(os.path.join('cascadia', 'scat_covariances', dirname)))
    saving_path.mkdir(exist_ok=True)

    # multiprocessed scattering covariance computation on gpus
    inputs = [(f, gpus[i % len(gpus)], saving_path, w, J) for (i, f) in enumerate(files_chunks)]

    n_gpus = len(gpus)
    results = []
    with WorkerPool(n_jobs=n_gpus*3) as pool:
        for result in pool.imap_unordered(worker, inputs, progress_bar=True):
            results.append(result)
