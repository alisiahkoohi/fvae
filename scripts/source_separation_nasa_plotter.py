"""Script to separate a cluster from given data."""

import os
import re
import shutil
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import srcsep
import torch
from mpire import WorkerPool
from obspy.core import UTCDateTime
from tqdm import tqdm

from facvae.utils import (
    collect_results,
    configsdir,
    create_lmst_xticks,
    create_namespace_from_args,
    datadir,
    make_experiment_name,
    parse_input_args,
    plotsdir,
    process_sequence_arguments,
    query_experiments,
    read_config,
)
from scripts.snippet_extractor import SnippetExtractor
from scripts.visualize_source_separation import plot_result

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH: str = datadir("mars")
MARS_SCAT_COV_PATH: str = datadir(os.path.join(MARS_PATH, "scat_covs_h5"))

# Configuration file.
SRC_SEP_CONFIG_FILE: str = "source_separation_glitch.json"

# Seed for reproducibility.
SEED: int = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


sns.set_style("whitegrid")
font = {"family": "serif", "style": "normal", "size": 11}
matplotlib.rc("font", **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


SAVE_DIR = "NASA_plots"


def plot_result(args, x_obs, x_hat, glitch_time, glitch_idx):
    time_interval = create_lmst_xticks(
        *glitch_time,
        time_zone="LMST",
        window_size=x_obs.shape[0],
    )
    fig = plt.figure(figsize=(5, 1.5))
    plt.plot_date(
        time_interval,
        x_obs,
        color="#000000",
        lw=0.6,
        alpha=0.7,
        xdate=True,
        fmt="",
    )
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    # Set the x-axis locator and formatter
    ax.xaxis.set_major_locator(
        matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6)
    )
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S"))
    ax.set_yticklabels([])
    ax.set_xlim([time_interval[0], time_interval[-1]])
    plt.ylim(
        [
            x_obs.min() * 1.02,
            x_obs.max() * 0.98,
        ]
    )
    ax.yaxis.set_label_position("right")
    plt.xlabel("Time (LMST)")
    ax.tick_params(axis="both", which="major", labelsize=10)
    plt.savefig(
        os.path.join(
            plotsdir(
                os.path.join(
                    args.experiment,
                    SAVE_DIR,
                )
            ),
            "x_obs_" + str(glitch_idx) + ".png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0.02,
    )
    plt.close(fig)

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot_date(
        time_interval,
        x_hat,
        color="#000000",
        lw=0.6,
        alpha=0.7,
        xdate=True,
        fmt="",
    )
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    # Set the x-axis locator and formatter
    ax.xaxis.set_major_locator(
        matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6)
    )
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S"))
    ax.set_yticklabels([])
    ax.set_xlim([time_interval[0], time_interval[-1]])
    plt.ylim(
        [
            x_obs.min() * 1.02,
            x_obs.max() * 0.98,
        ]
    )
    ax.yaxis.set_label_position("right")
    plt.xlabel("Time (LMST)")
    ax.tick_params(axis="both", which="major", labelsize=10)
    plt.savefig(
        os.path.join(
            plotsdir(os.path.join(args.experiment, SAVE_DIR)),
            "x_hat_" + str(glitch_idx) + ".png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0.02,
    )
    plt.close(fig)

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot_date(
        time_interval,
        x_obs - x_hat,
        color="#000000",
        lw=0.6,
        alpha=0.7,
        xdate=True,
        fmt="",
    )
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    # Set the x-axis locator and formatter
    ax.xaxis.set_major_locator(
        matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6)
    )
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S"))
    ax.set_yticklabels([])
    ax.set_xlim([time_interval[0], time_interval[-1]])
    plt.ylim(
        [
            x_obs.min() * 1.02,
            x_obs.max() * 0.98,
        ]
    )
    ax.yaxis.set_label_position("right")
    plt.xlabel("Time (LMST)")
    ax.tick_params(axis="both", which="major", labelsize=10)
    plt.savefig(
        os.path.join(
            plotsdir(os.path.join(args.experiment, SAVE_DIR)),
            "glitch_" + str(glitch_idx) + ".png",
        ),
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0.02,
    )
    plt.close(fig)


if __name__ == "__main__":
    # Remove cached directory if it exists from a previous run.
    if os.path.exists(os.path.join(srcsep.__path__[0], "_cached_dir")):
        shutil.rmtree(os.path.join(srcsep.__path__[0], "_cached_dir"))

    # Command line arguments for source separation.
    args = read_config(os.path.join(configsdir(), SRC_SEP_CONFIG_FILE))
    args = parse_input_args(args)

    # To be used for plotting the results.
    experiment_args = query_experiments(
        SRC_SEP_CONFIG_FILE,
        False,
        **vars(args),
    )

    args.experiment = make_experiment_name(args)
    args = process_sequence_arguments(args)

    # Read pretrained fVAE config JSON file specified by args.
    vae_args = read_config(
        os.path.join(
            configsdir("fvae_models"),
            args.facvae_model,
        )
    )
    vae_args = create_namespace_from_args(vae_args)
    vae_args.experiment = make_experiment_name(vae_args)
    vae_args = process_sequence_arguments(vae_args)
    vae_args.filter_key = args.filter_key

    # Setup snippet extractor to extract snippets from the Mars dataset required
    # for source separation optimization.
    snippet_extractor = SnippetExtractor(
        vae_args,
        os.path.join(MARS_SCAT_COV_PATH, vae_args.h5_filename),
    )

    # Extract the data that needs to be "cleaned" of some other sources.
    glitch, glitch_time = snippet_extractor.waveforms_per_scale_cluster(
        vae_args,
        args.cluster_g,
        args.scale_g,
        sample_size=1,
        component="U",
        timescale=args.scale_n[0],
        num_workers=1,
        overwrite_idx=args.overwrite_idx,
    )
    glitch = glitch[0, ...]
    glitch_time = glitch_time[0]
    glitch = glitch[:, None, :].astype(np.float64)

    filepath_deglitched = os.path.join(
        datadir("deglitched"), "2019/JUN/11/11.UVW_deglitched.mseed"
    )
    filepath_glitch = os.path.join(
        datadir("deglitched"), "2019/JUN/11/11.UVW_glitches.mseed"
    )

    deglitched = snippet_extractor.get_waveform_with_time_interval(
        window_time_interval=glitch_time[13],
        filepath=filepath_deglitched,
    )
    removed = snippet_extractor.get_waveform_with_time_interval(
        window_time_interval=glitch_time[13],
        filepath=filepath_glitch,
    )
    # Extract same data from the NASA deglitched dataset.
    plot_result(
        args,
        glitch[13, 0, :],
        deglitched[0],
        glitch_time[13],
        glitch_idx=13,
    )

    deglitched = snippet_extractor.get_waveform_with_time_interval(
        window_time_interval=glitch_time[26],
        filepath=filepath_deglitched,
    )
    removed = snippet_extractor.get_waveform_with_time_interval(
        window_time_interval=glitch_time[26],
        filepath=filepath_glitch,
    )
    # Extract same data from the NASA deglitched dataset.
    plot_result(
        args,
        glitch[26, 0, :],
        deglitched[0],
        glitch_time[26],
        glitch_idx=26,
    )

    deglitched = snippet_extractor.get_waveform_with_time_interval(
        window_time_interval=glitch_time[38],
        filepath=filepath_deglitched,
    )
    removed = snippet_extractor.get_waveform_with_time_interval(
        window_time_interval=glitch_time[38],
        filepath=filepath_glitch,
    )
    # Extract same data from the NASA deglitched dataset.
    plot_result(
        args,
        glitch[38, 0, :],
        deglitched[0],
        glitch_time[38],
        glitch_idx=38,
    )

    deglitched = snippet_extractor.get_waveform_with_time_interval(
        window_time_interval=(glitch_time[0][0], glitch_time[-1][1]),
        filepath=filepath_deglitched,
    )
    removed = snippet_extractor.get_waveform_with_time_interval(
        window_time_interval=(glitch_time[0][0], glitch_time[-1][1]),
        filepath=filepath_glitch,
    )
    # Extract same data from the NASA deglitched dataset.
    plot_result(
        args,
        glitch[:, 0, :].reshape(-1),
        deglitched[0],
        (glitch_time[0][0], glitch_time[-1][1]),
        glitch_idx="all",
    )
