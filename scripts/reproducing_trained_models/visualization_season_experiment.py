import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import datetime
import numpy as np
import os
from tqdm import tqdm

from facvae.utils import plotsdir, catalogsdir

sns.set_style("white")
font = {'family': 'serif', 'style': 'normal', 'size': 18}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)
sfmt.set_scientific(True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

SAMPLING_RATE = 20

# Sunrise and sunset file.
SUNRISE_SUNSET_FILE = os.path.join(catalogsdir(), 'sunrise_sunset.dat')

# Datastream merge method.
MERGE_METHOD = 1
FILL_VALUE = 'interpolate'

PATHS = [
    plotsdir(
        'nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29_full_summer1-2'
    ),
    plotsdir(
        'nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29_full_fall1-2'
    ),
    plotsdir(
        'nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29_full_winter1'
    ),
    plotsdir(
        'nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29_full_spring1-2'
    )
]

COLOR_CODES = [
    "#d27575",
    "#529b9c",
]

SCALES = ['1024', '4096', '16384', '65536']

CLUSTER_IDX_CONVERTER = {
    scale: [
        lambda i: [0, 1, 2, 3, 4, 5, 6, 7, 8][i],
        lambda i: [0, 1, 2, 3, 4, 5, 6, 7, 8][i],
        lambda i: [0, 1, 2, 3, 4, 5, 6, 7, 8][i],
        lambda i: [0, 1, 2, 3, 4, 5, 6, 7, 8][i],
    ]
    for scale in SCALES
}

SOL_TO_SEASON = {
    'Spring 1': [116, 305],
    'Summer 1': [306, 483],
    'Fall 1': [484, 624],
    'Winter 1': [625, 782],
    'Spring 2': [783, 978],
    'Summer 2': [979, 1155],
    'Fall 2': [1156, -1]
}


def get_datetime_from_decimal_time(decimal_time):
    """
    Convert decimal time to datetime object.
    """

    # Extract the hour, minute, and second components
    hour_component = int(decimal_time)
    minute_component = int((decimal_time - hour_component) * 60)
    second_component = int(
        ((decimal_time - hour_component) * 60 - minute_component) * 60)

    # Create the datetime object
    result_datetime = datetime.datetime(1900, 1, 1, hour_component,
                                        minute_component, second_component)

    return result_datetime


def obtain_sunset_and_sunrise_times():

    sunrise_sunset_data = {}
    with open(SUNRISE_SUNSET_FILE, 'r') as file:
        # Read each line and split it into elements
        for i, line in enumerate(file):
            elements = line.strip().split()
            # Convert elements to appropriate data types if needed
            if i > 0:
                sunrise_sunset_data[int(elements[0])] = {
                    'sunrise': float(elements[1]),
                    'sunset': float(elements[2])
                }
    return sunrise_sunset_data


if __name__ == '__main__':

    sunrise_sunset_data = obtain_sunset_and_sunrise_times()
    avg_sunrise_summar_fall = 0.0

    avg_sunrise_sunset_summar_fall = {
        'sunrise': [],
        'sunset': [],
    }
    for sun_state in ['sunrise', 'sunset']:
        for season in ['Summer 1', 'Summer 2', 'Fall 1', 'Fall 2']:
            avg_sunrise_sunset_summar_fall[sun_state].extend([
                sunrise_sunset_data[i][sun_state] for i in range(
                    SOL_TO_SEASON[season][0], SOL_TO_SEASON[season][1])
            ])
        avg_sunrise_sunset_summar_fall[
            sun_state] = get_datetime_from_decimal_time(
                np.mean(avg_sunrise_sunset_summar_fall[sun_state]))

    avg_sunrise_winter_spring_fall = {
        'sunrise': [],
        'sunset': [],
    }
    for sun_state in ['sunrise', 'sunset']:
        for season in ['Winter 1', 'Spring 1', 'Spring 2']:
            avg_sunrise_winter_spring_fall[sun_state].extend([
                sunrise_sunset_data[i][sun_state] for i in range(
                    SOL_TO_SEASON[season][0], SOL_TO_SEASON[season][1])
            ])
        avg_sunrise_winter_spring_fall[
            sun_state] = get_datetime_from_decimal_time(
                np.mean(avg_sunrise_winter_spring_fall[sun_state]))

    # Plot histogram of cluster times.
    for cluster in range(9):
        print('Plotting time histograms for cluster {}'.format(cluster))
        for scale in tqdm(SCALES):
            fig = plt.figure(figsize=(5, 1.5))
            for i, path in enumerate(PATHS):
                if i % 2 == 0:
                    mid_time_intervals_list = []
                mid_time_intervals = np.load(os.path.join(
                    path, 'mid_time_intervals.npy'),
                                             allow_pickle=True).item()

                mid_time_intervals_list.extend(mid_time_intervals[scale][str(
                    CLUSTER_IDX_CONVERTER[scale][i](cluster))])

                if i % 2 == 1:
                    sns.histplot(
                        mid_time_intervals_list,
                        color=COLOR_CODES[((i % 4) - 1) // 2],
                        element="step",
                        alpha=0.4,
                        binwidth=0.005,
                        label=['Summer/Fall',
                               'Winter/Spring'][((i % 4) - 1) // 2],
                        kde=False,
                        stat='probability',
                    )

                    if ((i % 4) - 1) // 2 == 0:
                        plt.axvline(
                            x=avg_sunrise_sunset_summar_fall['sunrise'],
                            color=COLOR_CODES[((i % 4) - 1) // 2],
                            linestyle='solid',
                        )
                        plt.axvline(
                            x=avg_sunrise_sunset_summar_fall['sunset'],
                            color=COLOR_CODES[((i % 4) - 1) // 2],
                            linestyle='solid',
                        )
                    elif ((i % 4) - 1) // 2 == 1:
                        plt.axvline(
                            x=avg_sunrise_winter_spring_fall['sunrise'],
                            color=COLOR_CODES[((i % 4) - 1) // 2],
                            linestyle='dashed',
                        )
                        plt.axvline(
                            x=avg_sunrise_winter_spring_fall['sunset'],
                            color=COLOR_CODES[((i % 4) - 1) // 2],
                            linestyle='dashed',
                        )

                    #  label='cluster ' + str(cluster))
                    ax = plt.gca()
                    ax.set_ylabel('Proportion', fontsize=10)
                    ax.set_xlim([
                        matplotlib.dates.date2num(
                            datetime.datetime(1900, 1, 1, 0, 0, 0, 0)),
                        matplotlib.dates.date2num(
                            datetime.datetime(1900, 1, 1, 23, 59, 59, 999999)),
                    ])
                    ax.xaxis.set_major_locator(
                        matplotlib.dates.HourLocator(interval=5))
                    ax.xaxis.set_major_formatter(
                        matplotlib.dates.DateFormatter('%H'))
                    # ax.set_yticklabels([])
                    ax.set_xlabel('Time (LMST)', fontsize=10)
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    ax.yaxis.set_major_formatter(sfmt)
                    # Adjust the font size of the exponent
                    ax.yaxis.offsetText.set_fontsize(10)
                    ax.legend(fontsize=8, ncol=4)
            plt.savefig(os.path.join(
                plotsdir(os.path.join('season_experiment', 'scale_' + scale)),
                'time_histogram_cluster-' + str(cluster) + '.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=200,
                        pad_inches=.02)
            plt.close(fig)
