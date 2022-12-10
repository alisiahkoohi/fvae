import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from facvae.utils import MarsDataset
from facvae.utils import plotsdir, create_lmst_xticks, lmst_xtick

matplotlib.use("Agg")
sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 18}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

dataset = MarsDataset(
    'data/mars/scat_covs_h5/3c_window_size-2048_q-6-2-2_r-2_J-7_use_day_data-1_model_type-scat.h5',
    0.99,
    data_types=['scat_cov'],
    load_to_memory=True,
    normalize_data=True,
    filter_key=[
        '2019-JUN-03', '2019-JUN-04', '2019-JUN-05', '2019-JUN-06',
        '2019-JUN-07', '2019-JUN-08', '2019-JUN-09', '2019-JUN-10',
        '2019-JUN-11'
    ])

data_idx = np.sort(dataset.test_idx)

A = dataset.sample_data(data_idx, 'scat_cov')
A = A.reshape(A.shape[0], -1).numpy()
A = StandardScaler().fit_transform(A)

ica = FastICA(n_components=25)
y = ica.fit_transform(A)

fig = plt.figure(figsize=(8, 6))
plt.scatter(y[:, 0], y[:, 1])
plt.title("2-dim PCA projection of scat coefs")
fig.savefig(os.path.join(plotsdir('ica'), '2dim-ica.png'),
            format="png",
            bbox_inches="tight",
            dpi=200,
            pad_inches=.05)
plt.close(fig)

A_ = ica.inverse_transform(y)
print(np.linalg.norm(A_ - A)/np.linalg.norm(A))


n_cluster = 3
# Create a KMeans instance with k clusters: model
model = KMeans(n_clusters=n_cluster)
# Fit model to samples
model.fit(y)

y_c = model.predict(y)
x_c = []
x_c = [y[y_c == j] for j in range(n_cluster)]

fig = plt.figure(figsize=(8, 6))
for j in range(n_cluster):
    plt.scatter(x_c[j][:, 0], x_c[j][:, 1], label=str(j))
plt.legend()
plt.title("K-mean clustering")
fig.savefig(os.path.join(plotsdir('ica'), 'k-means.png'),
            format="png",
            bbox_inches="tight",
            dpi=200,
            pad_inches=.05)
plt.close(fig)


def get_times(time_intervals, event_list):
    event_times = []
    for event, time_interval in zip(event_list, time_intervals):
        if len(event) > 0:
            time_interval = list(time_interval)
            for inner_idx, _ in enumerate(time_interval):
                time_interval[inner_idx] = lmst_xtick(time_interval[inner_idx])
                time_interval[inner_idx] = matplotlib.dates.date2num(
                    time_interval[inner_idx])
            event_times.append(np.array(time_interval).mean())
    return event_times


SAMPLING_RATE = 20
window_size = 2048
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

sample_size = 10

# List of figures and `ax`s to plot waveforms, spectrograms, and
# scattering covariances for each cluster.
figs_axs = [
    plt.subplots(sample_size,
                 n_cluster,
                 figsize=(8 * n_cluster, 8 * n_cluster)) for i in range(2)
]

fig_hist, ax_hist = plt.subplots(n_cluster + 2,
                                 1,
                                 figsize=(20, 4 * n_cluster),
                                 sharex=True)
# List of file names for each the figures.
names = ['waveform_samples', 'waveform_spectograms']

# Dictionary containing list of all the labels belonging to each
# predicted cluster.
cluster_labels = {str(i): [] for i in range(n_cluster)}
cluster_drops = {str(i): [] for i in range(n_cluster)}
cluster_glitches = {str(i): [] for i in range(n_cluster)}

# Find pressure drop times.
drop_list = dataset.get_drops(data_idx)
# Find glitch times.
glitch_list = dataset.get_glitches(data_idx)
# Find time intervals.
time_intervals = dataset.get_time_interval(data_idx)

# Find the times of pressure drops and glitches.
drop_times = get_times(time_intervals, drop_list)
glitch_times = get_times(time_intervals, glitch_list)

# Plot the histogram of the pressure drop cluster membership.
sns.histplot(drop_times,
             ax=ax_hist[-2],
             color="k",
             element="step",
             alpha=0.3,
             binwidth=0.01,
             label='Pressure drops')
ax_hist[-2].xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
ax_hist[-2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
ax_hist[-2].legend()

# Plot the histogram of the glitch cluster membership.
sns.histplot(glitch_times,
             ax=ax_hist[-1],
             color="k",
             element="step",
             alpha=0.3,
             binwidth=0.01,
             label='Glitches')
ax_hist[-1].xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
ax_hist[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H'))
ax_hist[-1].legend()

# Loop through all the clusters.
for i in range(n_cluster):
    cluster_drops[str(i)] = dataset.get_drops(data_idx[y_c == i])
    cluster_glitches[str(i)] = dataset.get_glitches(data_idx[y_c == i])
    cluster_idxs = data_idx[y_c == i]

    if len(cluster_idxs) > 0:
        cluster_times = dataset.get_time_interval(cluster_idxs)
        for outer_idx, _ in enumerate(cluster_times):
            cluster_times[outer_idx] = list(cluster_times[outer_idx])
            for inner_idx in range(len(cluster_times[outer_idx])):
                cluster_times[outer_idx][inner_idx] = lmst_xtick(
                    cluster_times[outer_idx][inner_idx])
                cluster_times[outer_idx][
                    inner_idx] = matplotlib.dates.date2num(
                        cluster_times[outer_idx][inner_idx])
        cluster_times = np.array(cluster_times).mean(-1)
        sns.histplot(cluster_times,
                     ax=ax_hist[i],
                     color=colors[i % 10],
                     element="step",
                     alpha=0.3,
                     binwidth=0.005,
                     label='cluster ' + str(i) + ' - ' +
                     str(len(cluster_times)))
        ax_hist[i].xaxis.set_major_locator(
            matplotlib.dates.HourLocator(interval=3))
        ax_hist[i].xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter('%H'))
        ax_hist[i].legend()

        cluster_idxs = cluster_idxs[-sample_size:, ...]

        waveforms = dataset.sample_data(cluster_idxs, type='waveform')
        waveform_times = dataset.get_time_interval(cluster_idxs)

        for j in range(len(cluster_idxs)):

            figs_axs[0][1][j, i].plot_date(create_lmst_xticks(
                *waveform_times[j], time_zone='LMST', window_size=window_size),
                                           waveforms[j, 0, :],
                                           xdate=True,
                                           color=colors[i % 10],
                                           lw=1.2,
                                           alpha=0.8,
                                           fmt='')
            figs_axs[0][1][j, i].xaxis.set_major_locator(
                matplotlib.dates.MinuteLocator(interval=10))
            figs_axs[0][1][j, i].xaxis.set_major_formatter(
                matplotlib.dates.DateFormatter('%H:%M'))
            figs_axs[0][1][j, i].set_ylim([-5e-7, 5e-7])
            figs_axs[0][1][j, i].set_title("Waveform from cluster " + str(i))

            figs_axs[1][1][j, i].set_ylim(0.1, SAMPLING_RATE / 2)
            figs_axs[1][1][j, i].specgram(waveforms[j, 0, :],
                                          Fs=SAMPLING_RATE,
                                          mode='magnitude',
                                          cmap='RdYlBu_r')
            figs_axs[1][1][j, i].set_ylim(0.1, SAMPLING_RATE / 2)
            figs_axs[1][1][j, i].set_yscale("log")
            figs_axs[1][1][j,
                           i].set_title("Spectrogram from cluster " + str(i))
            figs_axs[1][1][j, i].grid(False)

fig_hist.savefig(os.path.join(plotsdir('ica'), 'cluster_time_dist.png'),
                 format="png",
                 bbox_inches="tight",
                 dpi=200,
                 pad_inches=.05)
plt.close(fig_hist)

for (fig, _), name in zip(figs_axs, names):
    fig.savefig(os.path.join(plotsdir('ica'), name + '.png'),
                format="png",
                bbox_inches="tight",
                dpi=100,
                pad_inches=.05)
    plt.close(fig)

drop_count_per_cluster = {str(i): 0 for i in range(n_cluster)}
glitch_count_per_cluster = {str(i): 0 for i in range(n_cluster)}
for i in range(n_cluster):
    for drop in cluster_drops[str(i)]:
        drop_count_per_cluster[str(i)] += len(drop)
    for glitch in cluster_glitches[str(i)]:
        glitch_count_per_cluster[str(i)] += len(glitch)

fig = plt.figure(figsize=(8, 6))
plt.bar(range(n_cluster),
        [drop_count_per_cluster[str(i)] for i in range(n_cluster)],
        label='Pressure drops',
        color="k")
plt.xlabel('Clusters')
plt.ylabel('Pressure drop count')
plt.title('Pressure drop count per cluster')
plt.legend(ncol=2, fontsize=12)
plt.gca().set_xticks(range(n_cluster))
fig.savefig(os.path.join(plotsdir('ica'), 'pressure_drop_count.png'),
            format="png",
            bbox_inches="tight",
            dpi=200,
            pad_inches=.05)
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
plt.bar(range(n_cluster),
        [glitch_count_per_cluster[str(i)] for i in range(n_cluster)],
        label='Glitches',
        color="k")
plt.xlabel('Clusters')
plt.ylabel('Glitch count')
plt.title('Glitch count per cluster')
plt.legend(ncol=2, fontsize=12)
plt.gca().set_xticks(range(n_cluster))
fig.savefig(os.path.join(plotsdir('ica'), 'glitch_count.png'),
            format="png",
            bbox_inches="tight",
            dpi=200,
            pad_inches=.05)
plt.close(fig)
