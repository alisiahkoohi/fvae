import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .project_path import plotsdir

sns.set_style("darkgrid")
font = {'family': 'serif', 'style': 'normal', 'size': 10}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")


def plot_reconstructed_background_noise(args, name, n, g, nt):
    """Plots deglitching results.

    Args:
        n: background noise signal
        g: transient localized event, a "glitch"
        nt: candidate for n
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].plot((n + g)[0, 0, :], linewidth=0.4)
    axes[0, 0].set_title('glitched signal')
    lim = axes[0, 0].get_ylim()

    axes[1, 0].plot(g[0, 0, :], linewidth=0.4)
    axes[1, 0].set_title('glitch')
    axes[1, 0].set_ylim(lim)

    axes[0, 1].plot(n[0, 0, :], linewidth=0.4)
    axes[0, 1].set_title('background noise')
    axes[0, 1].set_ylim(lim)

    axes[1, 1].plot(nt[0, 0, :], linewidth=0.4, color='coral')
    axes[1, 1].set_title(r'$\tilde{n}$')
    axes[1, 1].set_ylim(lim)

    axes[0, 2].plot(n[0, 0, :] - nt[0, 0, :], linewidth=0.4, color='goldenrod')
    axes[0, 2].set_title(r'$n-\tilde{n}$')
    axes[0, 2].set_ylim(lim)

    for ax in axes.ravel():
        ax.axhline(0.0, color='black', linewidth=0.2)

    fig.savefig(os.path.join(plotsdir(args.experiment), name + '.png'),
                format="png",
                bbox_inches="tight",
                dpi=300,
                pad_inches=.05)
    plt.close(fig)


def plot_deglitching(args, name, x_obs, x_hat):
    """Plots deglitching results.

    Args:
        n: background noise signal
        g: transient localized event, a "glitch"
        nt: candidate for n
    """

    x_removed = x_obs - x_hat

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot(np.arange(int(args.scale_n[0])),
             x_obs[0, 0, :],
             color="#000000",
             lw=.6,
             alpha=0.7)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, int(args.scale_n[0]) - 1])
    plt.ylim([x_obs[0, 0, :].min() * 1.02, x_obs[0, 0, :].max() * 0.98])
    plt.ylabel("Observed")
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='both', which='major', labelsize=8)
    fig.savefig(os.path.join(plotsdir(os.path.join(args.experiment, name)),
                             'x_obs.png'),
                format="png",
                bbox_inches="tight",
                dpi=300,
                pad_inches=.05)
    plt.close(fig)

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot(np.arange(int(args.scale_n[0])),
             x_hat[0, 0, :],
             color="#000000",
             lw=.6,
             alpha=0.7)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, int(args.scale_n[0]) - 1])
    plt.ylim([x_obs[0, 0, :].min() * 1.02, x_obs[0, 0, :].max() * 0.98])
    plt.ylabel("Predicted")
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='both', which='major', labelsize=8)
    fig.savefig(os.path.join(plotsdir(os.path.join(args.experiment, name)),
                             'x_hat.png'),
                format="png",
                bbox_inches="tight",
                dpi=300,
                pad_inches=.05)
    plt.close(fig)

    fig = plt.figure(figsize=(5, 1.5))
    plt.plot(np.arange(int(args.scale_n[0])),
             x_removed[0, 0, :],
             color="#000000",
             lw=.6,
             alpha=0.7)
    ax = plt.gca()
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, int(args.scale_n[0]) - 1])
    plt.ylim([x_obs[0, 0, :].min() * 1.02, x_obs[0, 0, :].max() * 0.98])
    plt.ylabel("Removed")
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='both', which='major', labelsize=8)
    fig.savefig(os.path.join(plotsdir(os.path.join(args.experiment, name)),
                             'x_removed.png'),
                format="png",
                bbox_inches="tight",
                dpi=300,
                pad_inches=.05)
    plt.close(fig)
