import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import datetime
import obspy
import numpy as np
from mpire import WorkerPool
from obspy.core import UTCDateTime
import os
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.signal import spectrogram, correlate, correlation_lags
import scipy.signal as signal
import torch
from tqdm import tqdm

from facvae.utils import (plotsdir, create_lmst_xticks, lmst_xtick,
                          roll_zeropad, get_waveform_path_from_time_interval)

sns.set_style("white")
font = {'family': 'serif', 'style': 'normal', 'size': 18}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

SAMPLING_RATE = 20

# Datastream merge method.
MERGE_METHOD = 1
FILL_VALUE = 'interpolate'

PATHS = [
    plotsdir(
        'pyramid_2019_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_14616'
    ),
    plotsdir(
        'pyramid_2019_seed-31_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_14616'
    ),
    # plotsdir(
    #     'pyramid_2019_seed-21_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_14616'
    # ),
    plotsdir(
        'pyramid_2019_seed-11_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_14616'
    ),
    # plotsdir(
    #     'pyramid_2019_seed-1_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_14616'
    # )
]

color_codes = [
    "#be22d6", "#22c1d6", "#aab304", "#1e3ec9", "#c92020", "#15992a", "#d48955"
]

cluster_idx_coverter = {
    '1024': [
    lambda i: [0, 1, 2, 3, 4, 5, 6, 7, 8][i],
    lambda i: [1, 3, 2, 7, 0, 6, 4, 5, 8][i],
    # lambda i: [2, 7, 5, 3, 0, 8, 4, 1, 6][i],
    lambda i: [6, 1, 8, 7, 4, 2, 5, 0, 3][i],
    # lambda i: [7, 6, 8, 1, 3, 4, 2, 0, 5][i],
    ],
    '4096': [
    lambda i: [0, 1, 2, 3, 4, 5, 6, 7, 8][i],
    lambda i: [2, 0, 6, 7, 8, 1, 3, 5, 4][i],
    # lambda i: [5, 7, 1, 2, 3, 6, 8, 0, 4][i],
    lambda i: [2, 7, 4, 1, 8, 3, 6, 0, 5][i],
    # lambda i: [5, 6, 4, 2, 3, 8, 7, 1, 0][i]
    ],
    '16384': [
    lambda i: [0, 1, 2, 3, 4, 5, 6, 7, 8][i],
    lambda i: [0, 8, 2, 7, 1, 5, 6, 4, 3][i],
    # lambda i: [1, 2, 6, 7, 4, 3, 8, 0, 5][i],
    lambda i: [2, 5, 7, 6, 3, 1, 4, 0, 8][i],
    # lambda i: [6, 2, 5, 3, 8, 0, 7, 4, 1][i]
    ],
    '65536': [
    lambda i: [0, 1, 2, 3, 4, 5, 6, 7, 8][i],
    lambda i: [4, 1, 5, 2, 6, 0, 8, 7, 3][i],
    # lambda i: [2, 5, 0, 7, 1, 3, 8, 4, 6][i],
    lambda i: [0, 5, 7, 4, 1, 3, 2, 8, 6][i],
    # lambda i: [2, 6, 3, 8, 1, 5, 4, 0, 7][i]
    ]
}

# Plot histogram of cluster times.
for cluster in range(9):
    print('Plotting time histograms for cluster {}'.format(cluster))
    for scale in tqdm(['1024', '4096', '16384', '65536']):
        fig = plt.figure(figsize=(5, 1.5))
        for i, path in enumerate(PATHS):
            mid_time_intervals = np.load(os.path.join(
                path, 'mid_time_intervals.npy'),
                                         allow_pickle=True).item()
            sns.histplot(mid_time_intervals[scale][str(
                cluster_idx_coverter[scale][i](cluster))],
                         color=color_codes[i % len(color_codes)],
                         element="step",
                         alpha=0.3,
                         binwidth=0.005,
                         kde=False)
            #  label='cluster ' + str(cluster))
        ax = plt.gca()
        ax.set_ylabel('')
        ax.set_xlim([
            matplotlib.dates.date2num(datetime.datetime(
                1900, 1, 1, 0, 0, 0, 0)),
            matplotlib.dates.date2num(
                datetime.datetime(1900, 1, 1, 23, 59, 59, 999999)),
        ])
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=5))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
        # ax.legend(fontsize=12)
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(os.path.join(
            plotsdir(os.path.join('seed_experiment', 'scale_' + scale)),
            'time_histogram_cluster-' + str(cluster) + '.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=200,
                    pad_inches=.02)
        plt.close(fig)
