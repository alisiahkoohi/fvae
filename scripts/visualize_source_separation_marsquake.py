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
                          MarsquakeSeparationSetup, parse_input_args, plotsdir,
                          query_experiments, read_config)

import torch
import torch.nn as nn
from srcsep.frontend import analyze


class Pooling(nn.Module):
    def __init__(self, kernel_size):
        super(Pooling, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size)

    def forward(self, x):
        y = self.pool(x.view(x.shape[0], -1, x.shape[-1]))
        return y.view(x.shape[:-1] + (-1, ))


sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

# Random seed.
SEED = 12
np.random.seed(SEED)

# Configuration file.
CONFIG_FILE = 'source_separation_marsquake_S1133c.json'
SAVE_DIR = 'final_plots'

# Paths to raw Mars waveforms.
MARS_PATH = datadir('mars')
MARS_RAW_PATH = datadir(os.path.join(MARS_PATH, 'raw'))


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

        fig = plt.figure(figsize=(7, 1.5))
        plt.plot(np.arange(args.window_size),
                 experiment['x_obs'][0, 0, :],
                 color="#000000",
                 lw=.6,
                 alpha=0.7)
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        plt.xlim([0, args.window_size - 1])
        plt.ylabel("Observed")
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.savefig(os.path.join(
            plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
            filename + "_x_obs.png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 1.5))
        plt.plot(np.arange(args.window_size),
                 experiment['x_hat'][0, 0, :],
                 color="#000000",
                 lw=.6,
                 alpha=0.7)
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        plt.xlim([0, args.window_size - 1])
        plt.ylabel("Predicted")
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.savefig(os.path.join(
            plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
            filename + "_x_hat.png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 1.5))
        plt.plot(np.arange(args.window_size),
                 experiment['x_obs'][0, 0, :] - experiment['x_hat'][0, 0, :],
                 color="#000000",
                 lw=.6,
                 alpha=0.7)
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        plt.xlim([0, args.window_size - 1])
        plt.ylabel("Estimated glitch")
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.savefig(os.path.join(
            plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
            filename + "_glitch.png"),
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

    mars_srcsep = MarsquakeSeparationSetup(MARS_RAW_PATH, args.marsquake)
    time_axis = mars_srcsep.get_time_axis(len(x_hat_arr))
    P_start, S_start, PP_start, SS_start = mars_srcsep.get_arrival_times()

    fig = plt.figure(figsize=(7, 2))
    plt.plot_date(time_axis, x_obs_arr, '-', color="#000000", lw=.6, alpha=0.7)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.grid(True)
    plt.axvline(P_start,
                linestyle='--',
                linewidth=0.8,
                alpha=0.8,
                color="r",
                label="P-wave start")
    plt.axvline(S_start,
                linestyle='--',
                linewidth=0.8,
                alpha=0.8,
                color="b",
                label="S-wave start")
    if PP_start is not None:
        plt.axvline(PP_start,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.8,
                    color="g",
                    label="PP-wave start")
    if SS_start is not None:
        plt.axvline(SS_start,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.8,
                    color="y",
                    label="SS-wave start")
    plt.legend(loc='lower right', fontsize=8)
    ax.set_yticklabels([])
    plt.ylim([min(x_obs_arr), max(x_obs_arr)])
    plt.xlim([time_axis[0], time_axis[-1]])
    plt.ylabel("Raw data")
    # ax.yaxis.set_label_position("right")
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.savefig(os.path.join(
        plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
        "real-data.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 2))
    plt.plot_date(time_axis, x_hat_arr, '-', color="#000000", lw=.6, alpha=0.7)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.grid(True)
    plt.axvline(P_start,
                linestyle='--',
                linewidth=0.8,
                alpha=0.0,
                color="r",
                label="P-wave start")
    plt.axvline(S_start,
                linestyle='--',
                linewidth=0.8,
                alpha=0.0,
                color="b",
                label="S-wave start")
    if PP_start is not None:
        plt.axvline(PP_start,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.8,
                    color="g",
                    label="PP-wave start")
    if SS_start is not None:
        plt.axvline(SS_start,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.8,
                    color="y",
                    label="SS-wave start")
    ax.set_yticklabels([])
    plt.ylim([min(x_obs_arr), max(x_obs_arr)])
    plt.xlim([time_axis[0], time_axis[-1]])
    plt.ylabel("Background noise")
    # ax.yaxis.set_label_position("right")
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.savefig(os.path.join(
        plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
        "background.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 2))
    plt.plot_date(time_axis,
                  x_removed_arr,
                  '-',
                  color="#000000",
                  lw=.6,
                  alpha=0.7)
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.grid(True)
    plt.axvline(P_start,
                linestyle='--',
                linewidth=0.8,
                alpha=0.8,
                color="r",
                label="P-wave start")
    plt.axvline(S_start,
                linestyle='--',
                linewidth=0.8,
                alpha=0.8,
                color="b",
                label="S-wave start")
    if PP_start is not None:
        plt.axvline(PP_start,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.8,
                    color="g",
                    label="PP-wave start")
    if SS_start is not None:
        plt.axvline(SS_start,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.8,
                    color="y",
                    label="SS-wave start")
    plt.legend(loc='lower right', fontsize=8)
    ax.set_yticklabels([])
    plt.ylim([min(x_obs_arr), max(x_obs_arr)])
    plt.xlim([time_axis[0], time_axis[-1]])
    plt.ylabel("Marsquake")
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.savefig(os.path.join(
        plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
        "marsquake.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    scat_obs_arr = analyze(np.array(x_obs_arr).astype(np.float32),
                           Q=(6, 6),
                           J=(8, 8),
                           r=1,
                           keep_ps=True,
                           model_type='scat',
                           cuda=0,
                           normalize='each_ps',
                           estim_operator=Pooling(kernel_size=1),
                           qs=[1.0],
                           nchunks=1).y.numpy()[0, ...]

    scat_hat_arr = analyze(np.array(x_hat_arr).astype(np.float32),
                           Q=(6, 6),
                           J=(8, 8),
                           r=1,
                           keep_ps=True,
                           model_type='scat',
                           cuda=0,
                           normalize='each_ps',
                           estim_operator=Pooling(kernel_size=1),
                           qs=[1.0],
                           nchunks=1).y.numpy()[0, ...]

    scat_removed_arr = analyze(np.array(x_removed_arr).astype(np.float32),
                               Q=(6, 6),
                               J=(8, 8),
                               r=1,
                               keep_ps=True,
                               model_type='scat',
                               cuda=0,
                               normalize='each_ps',
                               estim_operator=Pooling(kernel_size=1),
                               qs=[1.0],
                               nchunks=1).y.numpy()[0, ...]

    for scat, fname in zip(
        [scat_obs_arr, scat_hat_arr, scat_removed_arr],
        ['scat-real-data', 'scat-background', 'scat-marsquake']):

        fig = plt.figure(figsize=(7, 2))
        plt.imshow(np.abs(scat),
                   aspect='auto',
                   norm=matplotlib.colors.PowerNorm(
                       1.2,
                       vmin=100 * np.min(np.abs(scat_obs_arr)),
                       vmax=0.25 * np.max(np.abs(scat_obs_arr))),
                   cmap='magma',
                   resample=True,
                   interpolation="lanczos",
                   filterrad=1,
                   extent=[
                       matplotlib.dates.date2num(time_axis[0]),
                       matplotlib.dates.date2num(time_axis[-1]), 0,
                       scat.shape[0]
                   ])
        ax = plt.gca()
        ax.xaxis_date()
        plt.axvline(P_start,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.8,
                    color="r",
                    label="P-wave start")
        plt.axvline(S_start,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.8,
                    color="b",
                    label="S-wave start")
        if PP_start is not None:
            plt.axvline(PP_start,
                        linestyle='--',
                        linewidth=0.8,
                        alpha=0.8,
                        color="g",
                        label="PP-wave start")
        if SS_start is not None:
            plt.axvline(SS_start,
                        linestyle='--',
                        linewidth=0.8,
                        alpha=0.8,
                        color="y",
                        label="SS-wave start")
        ax.set_yticklabels([])
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.grid(False)
        # plt.colorbar(fraction=0.03, pad=0.01)
        plt.ylabel("Scattering coefficients", fontsize=9)
        plt.legend(loc='lower right', fontsize=8)
        # ax.yaxis.set_label_position("right")
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.savefig(os.path.join(
            plotsdir(os.path.join(SAVE_DIR, experiment['args'].experiment)),
            fname + '.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)


if __name__ == '__main__':
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), CONFIG_FILE))
    args = parse_input_args(args)

    args.q = ([int(j) for j in args.q.replace(' ', '').split(',')], )
    args.j = ([int(j) for j in args.j.replace(' ', '').split(',')], )
    args.cluster = ([
        int(j) for j in args.cluster.replace(' ', '').split(',')
    ],)
    args.scale = ([
        int(j) for j in args.scale.replace(' ', '').split(',')
    ],)

    experiment_args = query_experiments(CONFIG_FILE, False, **vars(args))
    experiment_results = collect_results(experiment_args, ['x_obs', 'x_hat'])
    plot_result(experiment_results)
