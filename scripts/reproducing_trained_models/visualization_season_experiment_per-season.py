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

PATHS = {
    'Summer':
    plotsdir(
        'nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29_full_summer1-2'
    ),
    'Fall':
    plotsdir(
        'nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29_full_fall1-2'
    ),
    'Winter':
    plotsdir(
        'nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29_full_winter1'
    ),
    'Spring':
    plotsdir(
        'nature_full-mission_max_epoch-1000_batchsize-16384_lr-0.001_lr_final-0.001_ncluster-9_latent_dim-32_w_rec-0.15_wd-0.0_hidden_dim-1024_nlayer-4_window_size-65536_scales-1024-4096-16384-65536_seed-29_full_spring1-2'
    )
}

COLOR_CODES = {
    'Spring': "#a1c659",
    "Summer": "#52819c",
    "Fall": "#d27575",
    "Winter": "#8e6dbf"
}

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
    'Spring': [(116, 305), (783, 978)],
    'Summer': [(306, 483), (979, 1155)],
    'Fall': [(484, 624), (1156, -1)],
    'Winter': [(625, 782)],
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

    avg_sunrise_sunset_time = {
        season: {
            'sunrise': [],
            'sunset': []
        }
        for season in SOL_TO_SEASON.keys()
    }

    for season, _ in SOL_TO_SEASON.items():
        for sun_state in ['sunrise', 'sunset']:
            for season_range in SOL_TO_SEASON[season]:
                avg_sunrise_sunset_time[season][sun_state].extend([
                    sunrise_sunset_data[i][sun_state]
                    for i in range(season_range[0], season_range[1])
                ])
            avg_sunrise_sunset_time[season][
                sun_state] = get_datetime_from_decimal_time(
                    np.mean(avg_sunrise_sunset_time[season][sun_state]))

    # Plot histogram of cluster times.
    for cluster in range(9):
        print('Plotting time histograms for cluster {}'.format(cluster))
        for scale in tqdm(SCALES):
            fig = plt.figure(figsize=(5, 1.5))
            for i, (season, path) in enumerate(PATHS.items()):
                mid_time_intervals = np.load(os.path.join(
                    path, 'mid_time_intervals.npy'),
                                             allow_pickle=True).item()

                mid_time_intervals = mid_time_intervals[scale][str(
                    CLUSTER_IDX_CONVERTER[scale][i](cluster))]

                sns.histplot(
                    mid_time_intervals,
                    color=COLOR_CODES[season],
                    element="step",
                    alpha=0.25,
                    binwidth=0.005,
                    label=season,
                    kde=False,
                    stat='probability',
                )
                plt.axvline(
                    x=avg_sunrise_sunset_time[season]['sunrise'],
                    color=COLOR_CODES[season],
                    linestyle='solid',
                )
                plt.axvline(
                    x=avg_sunrise_sunset_time[season]['sunset'],
                    color=COLOR_CODES[season],
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
                plotsdir(os.path.join('season_experiment2', 'scale_' + scale)),
                'time_histogram_cluster-' + str(cluster) + '.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=200,
                        pad_inches=.02)
            plt.close(fig)
