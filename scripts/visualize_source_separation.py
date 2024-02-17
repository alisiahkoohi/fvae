"""Plotting script for marsquake cleaning.
"""

import os
import re

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from obspy.core import UTCDateTime

from facvae.utils import (
    checkpointsdir,
    configsdir,
    datadir,
    parse_input_args,
    plotsdir,
    query_experiments,
    read_config,
    process_sequence_arguments,
    create_lmst_xticks,
    collect_results,
)

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 11}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

# Random seed.
SEED = 12
np.random.seed(SEED)

# Configuration file.
SRC_SEP_CONFIG_FILE = 'source_separation.json'
SAVE_DIR = 'final_plots'

# Paths to raw Mars waveforms.
MARS_PATH = datadir('mars')
MARS_RAW_PATH = datadir(os.path.join(MARS_PATH, 'raw'))


def plot_result(args, experiment_results):

    x_obs = {key: [] for key in experiment_results.keys()}
    x_hat = {key: [] for key in experiment_results.keys()}
    x_removed = {key: [] for key in experiment_results.keys()}
    time_intervals = {key: [] for key in experiment_results.keys()}

    for filename, experiment in experiment_results.items():
        x_obs[filename] = experiment['x_obs'][0, 0, :]
        x_hat[filename] = experiment['x_hat'][0, 0, :]
        x_removed[filename] = (experiment['x_obs'][0, 0, :] -
                               experiment['x_hat'][0, 0, :])
        time_intervals[filename] = [
            UTCDateTime(experiment['glitch_time'][0].decode('utf-8')),
            UTCDateTime(experiment['glitch_time'][1].decode('utf-8'))
        ]

        glitch_idx = experiment['glitch_idx']

        time_interval = create_lmst_xticks(
            *time_intervals[filename],
            time_zone='LMST',
            window_size=args.scale_n[0],
        )
        fig = plt.figure(figsize=(5, 1.5))
        plt.plot_date(
            time_interval,
            experiment['x_obs'][0, 0, :],
            color="#000000",
            lw=.6,
            alpha=0.7,
            xdate=True,
            fmt='',
        )
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        # Set the x-axis locator and formatter
        ax.xaxis.set_major_locator(
            matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax.set_yticklabels([])
        ax.set_xlim([time_interval[0], time_interval[-1]])
        plt.ylim([
            experiment['x_obs'][0, 0, :].min() * 1.02,
            experiment['x_obs'][0, 0, :].max() * 0.98
        ])
        ax.yaxis.set_label_position("right")
        plt.xlabel("Time (LMST)")
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(os.path.join(
            plotsdir(
                os.path.join(experiment['args'].experiment, SAVE_DIR,
                             filename[:-3])),
            "x_obs_" + str(glitch_idx) + ".png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

        fig = plt.figure(figsize=(5, 1.5))
        plt.plot_date(
            time_interval,
            experiment['x_hat'][0, 0, :],
            color="#000000",
            lw=.6,
            alpha=0.7,
            xdate=True,
            fmt='',
        )
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        # Set the x-axis locator and formatter
        ax.xaxis.set_major_locator(
            matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax.set_yticklabels([])
        ax.set_xlim([time_interval[0], time_interval[-1]])
        plt.ylim([
            experiment['x_obs'][0, 0, :].min() * 1.02,
            experiment['x_obs'][0, 0, :].max() * 0.98
        ])
        ax.yaxis.set_label_position("right")
        plt.xlabel("Time (LMST)")
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(os.path.join(
            plotsdir(
                os.path.join(experiment['args'].experiment, SAVE_DIR,
                             filename[:-3])),
            "x_hat_" + str(glitch_idx) + ".png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

        fig = plt.figure(figsize=(5, 1.5))
        plt.plot_date(
            time_interval,
            experiment['x_obs'][0, 0, :] - experiment['x_hat'][0, 0, :],
            color="#000000",
            lw=.6,
            alpha=0.7,
            xdate=True,
            fmt='',
        )
        ax = plt.gca()
        ax.grid(True)
        ax.set_xticklabels([])
        # Set the x-axis locator and formatter
        ax.xaxis.set_major_locator(
            matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax.set_yticklabels([])
        ax.set_xlim([time_interval[0], time_interval[-1]])
        plt.ylim([
            experiment['x_obs'][0, 0, :].min() * 1.02,
            experiment['x_obs'][0, 0, :].max() * 0.98
        ])
        ax.yaxis.set_label_position("right")
        plt.xlabel("Time (LMST)")
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(os.path.join(
            plotsdir(
                os.path.join(experiment['args'].experiment, SAVE_DIR,
                             filename[:-3])),
            "glitch_" + str(glitch_idx) + ".png"),
                    format="png",
                    bbox_inches="tight",
                    dpi=400,
                    pad_inches=.02)
        plt.close(fig)

    start_time = []
    end_time = []
    for key in time_intervals.keys():
        start_time.append(time_intervals[key][0])
        end_time.append(time_intervals[key][1])
    start_time = min(start_time)
    end_time = max(end_time)
    time_intervals = create_lmst_xticks(
        start_time,
        end_time,
        time_zone='LMST',
        window_size=args.scale_g[0],
    )

    x_obs_arr = np.zeros([args.scale_g[0]])
    for key in x_obs.keys():
        i = int(re.search(r'\d+', key).group())
        if len(x_obs[key]) > 0:
            x_obs_arr[i * args.scale_n[0]:(i + 1) *
                      args.scale_n[0]] = x_obs[key]

    x_hat_arr = np.zeros([args.scale_g[0]])
    for key in x_hat.keys():
        i = int(re.search(r'\d+', key).group())
        if len(x_hat[key]) > 0:
            x_hat_arr[i * args.scale_n[0]:(i + 1) *
                      args.scale_n[0]] = x_hat[key]

    x_removed_arr = np.zeros([args.scale_g[0]])
    for key in x_removed.keys():
        i = int(re.search(r'\d+', key).group())
        if len(x_removed[key]) > 0:
            x_removed_arr[i * args.scale_n[0]:(i + 1) *
                          args.scale_n[0]] = x_removed[key]

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot_date(time_intervals,
                  x_obs_arr,
                  color="#000000",
                  lw=.6,
                  alpha=0.8,
                  fmt='')
    ax = plt.gca()
    ax.grid(True)
    ax.xaxis.set_major_locator(
        matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
    ax.set_xlim([time_intervals[0], time_intervals[-1]])
    plt.xticks(rotation=0)

    # plt.legend(loc='upper right', fontsize=6)
    plt.ylim([min(x_obs_arr) * 1.1, max(x_obs_arr) * 0.9])
    # plt.xlim([-70, len(x_obs_arr) - 1 + 70])
    # ax.yaxis.set_label_position("left")
    # plt.ylabel("Raw data")
    plt.xlabel("Time (LMST)")
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    plt.savefig(os.path.join(
        plotsdir(os.path.join(experiment['args'].experiment, SAVE_DIR)),
        "real-data.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot_date(time_intervals,
                  x_hat_arr,
                  color="#000000",
                  lw=.6,
                  alpha=0.8,
                  fmt='')
    ax = plt.gca()
    ax.grid(True)
    ax.xaxis.set_major_locator(
        matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
    ax.set_xlim([time_intervals[0], time_intervals[-1]])
    plt.xticks(rotation=0)

    # plt.legend(loc='upper right', fontsize=6)
    plt.ylim([min(x_obs_arr) * 1.1, max(x_obs_arr) * 0.9])
    # plt.xlim([-70, len(x_obs_arr) - 1 + 70])
    # ax.yaxis.set_label_position("left")
    # plt.ylabel("Raw data")
    plt.xlabel("Time (LMST)")
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    plt.savefig(os.path.join(
        plotsdir(os.path.join(experiment['args'].experiment, SAVE_DIR)),
        "deglitched.png"),
                format="png",
                bbox_inches="tight",
                dpi=400,
                pad_inches=.02)
    plt.close(fig)

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot_date(time_intervals,
                  np.array(x_obs_arr) - np.array(x_hat_arr),
                  color="#000000",
                  lw=.6,
                  alpha=0.8,
                  fmt='')
    ax = plt.gca()
    ax.grid(True)
    ax.xaxis.set_major_locator(
        matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
    ax.set_xlim([time_intervals[0], time_intervals[-1]])
    plt.xticks(rotation=0)

    # plt.legend(loc='upper right', fontsize=6)
    plt.ylim([min(x_obs_arr) * 1.1, max(x_obs_arr) * 0.9])
    # plt.xlim([-70, len(x_obs_arr) - 1 + 70])
    # ax.yaxis.set_label_position("left")
    # plt.ylabel("Raw data")
    plt.xlabel("Time (LMST)")
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    plt.savefig(os.path.join(
        plotsdir(os.path.join(experiment['args'].experiment, SAVE_DIR)),
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

    experiment_args = query_experiments(SRC_SEP_CONFIG_FILE, True,
                                        **vars(args))
    experiment_results = collect_results(experiment_args, [
        'x_obs',
        'x_hat',
        'glitch_idx',
        'glitch_time',
    ])

    args = process_sequence_arguments(args)
    plot_result(args, experiment_results)
