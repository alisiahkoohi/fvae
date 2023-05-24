"""Plotting script for marsquake cleaning.
"""

import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import obspy
import seaborn as sns
from matplotlib.dates import DateFormatter

from facvae.utils import (checkpointsdir, configsdir, datadir,
                          GlitchSeparationSetup, parse_input_args, plotsdir,
                          query_experiments, read_config)

from srcsep.utils import GlitchSeparationSetup

import torch
import torch.nn as nn
from srcsep.frontend import analyze

from facvae.utils import (plotsdir, create_lmst_xticks, lmst_xtick,
                          roll_zeropad, get_waveform_path_from_time_interval)
from datetime import datetime

from obspy import UTCDateTime


class Pooling(nn.Module):

    def __init__(self, kernel_size):
        super(Pooling, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size)

    def forward(self, x):
        y = self.pool(x.view(x.shape[0], -1, x.shape[-1]))
        return y.view(x.shape[:-1] + (-1, ))


sns.set_style("darkgrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

# Random seed.
SEED = 12
np.random.seed(SEED)

# Configuration file.
# SRC_SEP_CONFIG_FILE = 'source_separation_sunrise_deglitch.json'
SRC_SEP_CONFIG_FILE = 'source_separation_synthetic_remove_fine_scale.json'
# SRC_SEP_CONFIG_FILE = 'source_separation_small_scale.json'
SAVE_DIR = 'final_plots'

# Paths to raw Mars waveforms.
DATA_PATH = datadir('synthetic_dataset')
SCAT_COV_PATH = datadir(os.path.join(DATA_PATH, 'scat_covs_h5'))


def collect_results(experiment_args, keys):
    """Collect the results from the experiments."""
    results = {}
    for args in experiment_args:
        for file in os.listdir(checkpointsdir(args.experiment)):
            h5_path = os.path.join(checkpointsdir(args.experiment), file)
            with h5py.File(h5_path, 'r') as f:
                results[file] = {key: f[key][...] for key in keys}
                results[file]['args'] = args

    return results


def plot_result(experiment_results):

    x_obs = {key: [] for key in experiment_results.keys()}
    x_hat = {key: [] for key in experiment_results.keys()}
    x_removed = {key: [] for key in experiment_results.keys()}

    for filename, experiment in experiment_results.items():
        x_obs[filename] = experiment['x_obs'][0, 0, :]
        x_hat[filename] = experiment['x_hat'][0, 0, :]
        x_removed[filename] = (experiment['x_obs'][0, 0, :] -
                               experiment['x_hat'][0, 0, :])
        glitch_idx = experiment['glitch_idx']

        fig = plt.figure(figsize=(5, 1.5))
        plt.plot(np.arange(args.window_size),
                 experiment['x_obs'][0, 0, :],
                 color="#000000",
                 lw=.6,
                 alpha=0.7)
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.xlim([0, args.window_size - 1])
        plt.ylim([
            experiment['x_obs'][0, 0, :].min() * 1.02,
            experiment['x_obs'][0, 0, :].max() * 0.98
        ])
        # plt.ylabel("Observed")
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.savefig(os.path.join(
            plotsdir(
                os.path.join(SAVE_DIR, experiment['args'].experiment,
                             filename[:-3])),
            "x_obs_" + str(glitch_idx) + ".png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

        fig = plt.figure(figsize=(5, 1.5))
        plt.plot(np.arange(args.window_size),
                 experiment['x_hat'][0, 0, :],
                 color="#000000",
                 lw=.6,
                 alpha=0.7)
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.xlim([0, args.window_size - 1])
        plt.ylim([
            experiment['x_obs'][0, 0, :].min() * 1.02,
            experiment['x_obs'][0, 0, :].max() * 0.98
        ])
        # plt.ylabel("Predicted")
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.savefig(os.path.join(
            plotsdir(
                os.path.join(SAVE_DIR, experiment['args'].experiment,
                             filename[:-3])),
            "x_hat_" + str(glitch_idx) + ".png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

        fig = plt.figure(figsize=(5, 1.5))
        plt.plot(np.arange(args.window_size),
                 experiment['x_obs'][0, 0, :] - experiment['x_hat'][0, 0, :],
                 color="#000000",
                 lw=.6,
                 alpha=0.7)
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.xlim([0, args.window_size - 1])
        plt.ylim([
            experiment['x_obs'][0, 0, :].min() * 1.02,
            experiment['x_obs'][0, 0, :].max() * 0.98
        ])
        # plt.ylabel("Estimated glitch")
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.savefig(os.path.join(
            plotsdir(
                os.path.join(SAVE_DIR, experiment['args'].experiment,
                             filename[:-3])),
            "glitch_" + str(glitch_idx) + ".png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

    x_obs_arr = []
    for i in range(len(x_obs.keys())):
        x_obs_arr.extend(x_obs[f'reconstruction_{i}.h5'])
    x_hat_arr = []
    for i in range(len(x_hat.keys())):
        x_hat_arr.extend(x_hat[f'reconstruction_{i}.h5'])
    x_removed_arr = []
    for i in range(len(x_removed.keys())):
        x_removed_arr.extend(x_removed[f'reconstruction_{i}.h5'])

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot(np.arange(len(x_obs_arr)),
             x_obs_arr,
             color="#000000",
             lw=.6,
             alpha=0.8,
             )
    ax = plt.gca()
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.ylim([min(x_obs_arr) * 1.1, max(x_obs_arr) * 0.9])
    plt.savefig(os.path.join(
        plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
        "real-data.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot(np.arange(len(x_obs_arr)),
             x_hat_arr,
             color="#000000",
             lw=.6,
             alpha=0.8,
             )
    ax = plt.gca()
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.ylim([min(x_obs_arr) * 1.1, max(x_obs_arr) * 0.9])
    plt.savefig(os.path.join(
        plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
        "deglitched.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot(np.arange(len(x_obs_arr)),
             np.array(x_obs_arr) - np.array(x_hat_arr),
             color="#000000",
             lw=.6,
             alpha=0.8,
             )
    ax = plt.gca()
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.ylim([min(x_obs_arr) * 1.1, max(x_obs_arr) * 0.9])
    plt.savefig(os.path.join(
        plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
        "separated.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)


if __name__ == '__main__':
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), SRC_SEP_CONFIG_FILE))
    args = parse_input_args(args)

    args.q = ([int(j) for j in args.q.replace(' ', '').split(',')], )
    args.j = ([int(j) for j in args.j.replace(' ', '').split(',')], )

    experiment_args = query_experiments(SRC_SEP_CONFIG_FILE, True,
                                        **vars(args))
    experiment_results = collect_results(experiment_args,
                                         ['x_obs', 'x_hat', 'glitch_idx'])
    # args.window_size = int(args.scale_n)
    plot_result(experiment_results)