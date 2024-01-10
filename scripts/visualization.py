import os
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import obspy
import numpy as np
from mpire import WorkerPool
from obspy.core import UTCDateTime
from scipy.signal import correlate, correlation_lags
import scipy.signal as signal
import torch
from tqdm import tqdm

from facvae.utils import (plotsdir, create_lmst_xticks, lmst_xtick,
                          roll_nanpad, get_waveform_path_from_time_interval)

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 18}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

SAMPLING_RATE = 20

# Datastream merge method.
MERGE_METHOD = 1
FILL_VALUE = 'interpolate'


class Visualization(object):
    """Class visualizing results of a GMVAE training.
    """

    def __init__(self, args, network, dataset, data_loader, device):
        # Pretrained GMVAE network.
        self.network = network
        self.network.eval()
        # The entire dataset.
        self.dataset = dataset
        # Scales.
        self.scales = args.scales
        self.in_shape = {
            scale: dataset.shape['scat_cov'][scale]
            for scale in self.scales
        }
        # Window size of the dataset.
        self.window_size = args.window_size
        # Device to perform computations on.
        self.device = device

        (self.cluster_membership, self.cluster_membership_prob,
         self.confident_idxs, self.per_cluster_confident_idxs,
         self.latent_features,
         self.window_labels) = self.evaluate_model(args, data_loader)

        # Colors to be used for visualizing different clusters.
        # self.colors = [
        #     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        #     '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        # ]
        self.colors = ['k']

    def get_waveform(self, window_idx, scale):
        window_time_interval = self.get_time_interval(window_idx,
                                                      scale,
                                                      lmst=False)

        filepath = get_waveform_path_from_time_interval(*window_time_interval)

        # Extract some properties of the data to setup HDF5 file.
        data_stream = obspy.read(filepath)
        data_stream = data_stream.merge(method=MERGE_METHOD,
                                        fill_value=FILL_VALUE)

        data_stream = data_stream.slice(*window_time_interval)

        waveform = np.stack([td.data[-int(scale):] for td in data_stream])

        # Return the required subwindow.
        return waveform.astype(np.float32)

    def get_time_interval(self, window_idx, scale, lmst=True):

        # Extract window time interval.
        window_time_interval = self.dataset.get_time_interval([window_idx],
                                                              scale)[0]
        if lmst:
            # Convert to LMST format, usable by matplotlib.
            window_time_interval = (create_lmst_xticks(*window_time_interval,
                                                       time_zone='LMST',
                                                       window_size=int(scale)),
                                    window_time_interval)

        # Return the required time interval.
        return window_time_interval

    def evaluate_model(self, args, data_loader):
        """
        Evaluate the trained FACVAE model.

        Here we pass the data through the trained model and for each window and
        each scale, we extract the cluster membership and the probability of
        the cluster membership. We then sort the windows based on the most
        confident cluster membership.

        Args:
            args: (argparse) arguments containing the model information.
            data_loader: (DataLoader) loader containing the data.
        """

        # Placeholder for cluster membership and probablity for all the data.
        cluster_membership = {
            scale:
            torch.zeros(len(data_loader.dataset),
                        self.dataset.data['scat_cov'][scale].shape[1],
                        dtype=torch.long)
            for scale in self.scales
        }
        cluster_membership_prob = {
            scale:
            torch.zeros(len(data_loader.dataset),
                        self.dataset.data['scat_cov'][scale].shape[1],
                        dtype=torch.float)
            for scale in self.scales
        }

        latent_features = {
            scale:
            torch.zeros(len(data_loader.dataset),
                        self.dataset.data['scat_cov'][scale].shape[1],
                        args.latent_dim,
                        dtype=torch.float)
            for scale in self.scales
        }
        window_labels = {scale: {} for scale in self.scales}

        # Load all the labels to avoid time consuming slicing.
        all_labels = {
            scale: self.dataset.get_labels(data_loader.dataset, scale)
            for scale in self.scales
        }
        argsorted_indices = np.argsort(data_loader.dataset)
        value_idx_map = {
            int(idx): i
            for i, idx in enumerate(argsorted_indices)
        }

        # Extract cluster memberships.
        pbar = tqdm(total=len(data_loader), desc='Evalutating the model')
        for _, idx in enumerate(data_loader):
            # Load data.
            x = self.dataset.sample_data(idx, 'scat_cov')
            # Move to `device`.
            x = {scale: x[scale].to(self.device) for scale in self.scales}
            # Run the input data through the pretrained GMVAE network.
            with torch.no_grad():
                output = self.network(x)
            # Extract the predicted cluster memberships.
            for scale in self.scales:
                cluster_membership[scale][np.sort(idx), :] = output['logits'][
                    scale].argmax(axis=1).reshape(
                        len(idx),
                        self.dataset.data['scat_cov'][scale].shape[1]).cpu()
                cluster_membership_prob[scale][np.sort(idx), :] = output[
                    'prob_cat'][scale].max(axis=1)[0].reshape(
                        len(idx),
                        self.dataset.data['scat_cov'][scale].shape[1]).cpu()
                latent_features[scale][
                    np.sort(idx), :, :] = output['mean'][scale].reshape(
                        len(idx),
                        self.dataset.data['scat_cov'][scale].shape[1],
                        args.latent_dim).cpu()
                for i in idx:
                    label = all_labels[scale][argsorted_indices[value_idx_map[
                        int(i)]]]
                    if label:
                        window_labels[scale][int(i)] = label

            pbar.update(1)

        # Sort indices based on most confident cluster predictions by the
        # network (increasing). The outcome is a dictionary with a key for each
        # scale, where the window indices are stored.
        confident_idxs = {}
        for scale in self.scales:
            # Flatten cluster_membership_prob into a 1D tensor.
            prob_flat = cluster_membership_prob[scale].flatten()

            # Sort the values in the flattened tensor in descending order and
            # return the indices.
            confident_idxs[scale] = torch.argsort(prob_flat,
                                                  descending=True).numpy()

        per_cluster_confident_idxs = {
            scale: {
                str(i): []
                for i in range(args.ncluster)
            }
            for scale in self.scales
        }

        for scale in self.scales:
            for i in range(len(confident_idxs[scale])):
                per_cluster_confident_idxs[scale][str(
                    cluster_membership[scale][confident_idxs[scale]
                                              [i]].item())].append(
                                                  confident_idxs[scale][i])

        return (cluster_membership, cluster_membership_prob, confident_idxs,
                per_cluster_confident_idxs, latent_features, window_labels)

    def load_per_scale_per_cluster_waveforms(
        self,
        args,
        sample_size=100,
        overlap=True,
        scale_idx=None,
        cluster_idx=None,
    ):

        def do_overlap(pair1, pair2):
            start1, end1 = pair1
            start2, end2 = pair2

            # Check for all types of overlap
            return (start1 <= start2 <= end1 or start1 <= end2 <= end1
                    or start2 <= start1 <= end2 or start2 <= end1 <= end2)

        if scale_idx is None:
            scale_idx = self.scales
        if cluster_idx is None:
            cluster_idx = np.arange(args.ncluster)

        self.waveforms = {
            scale: {
                str(i): []
                for i in cluster_idx
            }
            for scale in scale_idx
        }

        self.time_intervals = {
            scale: {
                str(i): []
                for i in cluster_idx
            }
            for scale in scale_idx
        }

        # Serial worker for plotting waveforms for each cluster.
        def load_serial_job(shared_in, i):

            (per_cluster_confident_idxs, scale, get_time_interval, overlap,
             do_overlap, sample_size, get_waveform) = shared_in
            i = i[0]
            utc_time_intervals = []
            window_idx_list = []
            for sample_idx in range(
                    len(per_cluster_confident_idxs[scale][str(i)])):
                window_idx = per_cluster_confident_idxs[scale][str(
                    i)][sample_idx]
                utc_interval = get_time_interval(window_idx, scale, lmst=False)
                should_add = True

                if not overlap:
                    for interval in utc_time_intervals:
                        if do_overlap(interval, utc_interval):
                            should_add = False
                            break

                if should_add:
                    utc_time_intervals.append(utc_interval)
                    window_idx_list.append(window_idx)

                if len(window_idx_list) == sample_size:
                    break

            per_scale_per_cluster_waveforms = []
            per_scale_per_cluster_time_intervals = []
            for window_idx in window_idx_list:
                per_scale_per_cluster_waveforms.append(
                    get_waveform(window_idx, scale))
                per_scale_per_cluster_time_intervals.append(
                    get_time_interval(window_idx, scale)[0])
            return (
                i,
                per_scale_per_cluster_waveforms,
                per_scale_per_cluster_time_intervals,
            )

        print('Reading waveforms')
        for scale in tqdm(scale_idx, desc="Scale loop"):

            # Plot waveforms for each cluster.
            with WorkerPool(
                    n_jobs=len(cluster_idx),
                    shared_objects=(
                        self.per_cluster_confident_idxs,
                        scale,
                        self.get_time_interval,
                        overlap,
                        do_overlap,
                        sample_size,
                        self.get_waveform,
                    ),
                    start_method='fork',
            ) as pool:
                outputs = pool.map(
                    load_serial_job,
                    cluster_idx,
                    progress_bar=False,
                )

            (idxs, waveforms, time_intervals) = zip(*outputs)
            for i in idxs:
                self.waveforms[scale][str(i)] = waveforms[i]
                self.time_intervals[scale][str(i)] = time_intervals[i]

    def plot_fourier(self, args, sample_size):

        # Serial worker for plotting Fourier transforms for each cluster.
        def fourier_serial_job(shared_in, clusters):
            args, scales, waveforms = shared_in
            for cluster in clusters:
                print('Plotting Fourier transforms for cluster {}'.format(
                    cluster))
                for scale in scales:
                    for sample_idx, waveform in enumerate(
                            waveforms[scale][str(cluster)]):
                        for comp in range(waveform.shape[0]):
                            fig = plt.figure(figsize=(7, 2))
                            # Compute the Fourier transform.
                            freqs = np.fft.fftfreq(waveform.shape[1],
                                                   d=1 / SAMPLING_RATE)
                            ft = np.fft.fft(waveform[comp, :], norm='forward')
                            # Plot the Fourier transform.
                            plt.plot(np.fft.fftshift(freqs),
                                     np.fft.fftshift(np.abs(ft)))
                            ax = plt.gca()
                            plt.xlim([0, SAMPLING_RATE / 2])
                            ax.set_ylabel('Amplitude', fontsize=10)
                            ax.set_xlabel('Frequency (Hz)', fontsize=10)
                            ax.set_yscale("log")
                            ax.grid(True)
                            ax.tick_params(axis='both',
                                           which='major',
                                           labelsize=8)
                            plt.savefig(
                                os.path.join(
                                    plotsdir(
                                        os.path.join(
                                            args.experiment, 'scale_' + scale,
                                            'cluster_' + str(cluster),
                                            'component_' + str(comp))),
                                    'fourier_transform_{}.png'.format(
                                        sample_idx),
                                ),
                                format="png",
                                bbox_inches="tight",
                                dpi=200,
                                pad_inches=.02,
                            )
                            plt.close(fig)

        # Plot Fourier transform for each cluster.
        worker_in = np.array_split(np.arange(args.ncluster),
                                   args.ncluster,
                                   axis=0)
        with WorkerPool(
                n_jobs=args.ncluster,
                shared_objects=(args, self.scales, self.waveforms),
                start_method='fork',
        ) as pool:
            pool.map(fourier_serial_job, worker_in, progress_bar=True)

        # Serial worker for plotting spectogram for each cluster.
        def spectogram_serial_job(shared_in, clusters):
            args, scales, waveforms = shared_in
            for cluster in clusters:
                print('Plotting spectrograms for cluster {}'.format(cluster))
                for scale in scales:
                    for sample_idx, waveform in enumerate(
                            waveforms[scale][str(cluster)]):
                        for comp in range(waveform.shape[0]):
                            fig = plt.figure(figsize=(7, 2))
                            # Plot spectrogram.
                            nperseg = min(256, int(scale) // 4)
                            plt.specgram(
                                waveform[comp, :],
                                NFFT=nperseg,
                                noverlap=nperseg // 8,
                                Fs=SAMPLING_RATE,
                                mode='magnitude',
                                cmap='RdYlBu_r',
                            )
                            ax = plt.gca()
                            plt.ylim([0, SAMPLING_RATE / 2])
                            ax.set_xticklabels([])
                            ax.set_ylabel('Frequency (Hz)', fontsize=10)
                            ax.grid(False)
                            ax.tick_params(axis='both',
                                           which='major',
                                           labelsize=8)
                            plt.savefig(
                                os.path.join(
                                    plotsdir(
                                        os.path.join(
                                            args.experiment, 'scale_' + scale,
                                            'cluster_' + str(cluster),
                                            'component_' + str(comp))),
                                    'spectrogram_{}.png'.format(sample_idx),
                                ),
                                format="png",
                                bbox_inches="tight",
                                dpi=200,
                                pad_inches=.02,
                            )
                            plt.close(fig)

        # Plot spectogram for each cluster.
        worker_in = np.array_split(np.arange(args.ncluster),
                                   args.ncluster,
                                   axis=0)
        with WorkerPool(
                n_jobs=args.ncluster,
                shared_objects=(args, self.scales, self.waveforms),
                start_method='fork',
        ) as pool:
            pool.map(spectogram_serial_job, worker_in, progress_bar=True)

    def plot_waveforms(self, args, sample_size=10):
        """Plot waveforms.
        """

        self.load_per_scale_per_cluster_waveforms(
            args,
            sample_size=sample_size,
            overlap=False,
        )
        sns.set_style("darkgrid")

        # self.plot_fourier(args, sample_size)

        # Serial worker for plotting waveforms for each cluster.
        def waveform_serial_job(shared_in, clusters):
            args, scales, waveforms, time_intervals, colors = shared_in
            for cluster in clusters:
                for scale in scales:
                    for sample_idx, waveform in enumerate(
                            waveforms[scale][str(cluster)]):
                        fig, axes = plt.subplots(
                            nrows=3,
                            sharex=True,
                            figsize=(5, 2),
                        )
                        for comp in range(waveform.shape[0]):
                            # Plot waveforms.
                            waveform[comp, :] = waveform[
                                comp, :] / np.linalg.norm(waveform[comp, :])
                            axes[comp].plot_date(
                                time_intervals[scale][str(cluster)]
                                [sample_idx],
                                waveform[comp, :],
                                xdate=True,
                                color=colors[cluster % len(colors)],
                                lw=1.5,
                                alpha=0.9,
                                fmt='',
                            )
                            axes[comp].set_ylim([
                                min(waveform[comp, :].reshape(-1)),
                                max(waveform[comp, :].reshape(-1))
                            ])
                            axes[comp].set_yticklabels([])
                            # axes[comp].set_xticklabels([])
                            # axes[comp].set_ylabel(labels[comp], fontsize=8, rotation=90, labelpad=-3)
                            axes[comp].tick_params(axis='both',
                                                   which='major',
                                                   labelsize=8)
                            axes[comp].grid(False)
                            axes[comp].set_xticklabels([])
                        plt.subplots_adjust(hspace=0)
                        # Set the x-axis locator and formatter
                        # axes[-1].xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(minticks=4, maxticks=6))
                        # axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
                        plt.savefig(
                            os.path.join(
                                plotsdir(
                                    os.path.join(args.experiment,
                                                 'scale_' + scale, 'waveforms',
                                                 'cluster_' + str(cluster))),
                                'waveform_{}.png'.format(sample_idx),
                            ),
                            format="png",
                            bbox_inches="tight",
                            dpi=300,
                            pad_inches=.01,
                        )
                        plt.close(fig)

        # Plot waveforms for each cluster.
        worker_in = np.array_split(
            np.arange(args.ncluster),
            args.ncluster,
            axis=0,
        )
        print(' [*] Plotting waveforms')
        with WorkerPool(
                n_jobs=args.ncluster,
                shared_objects=(
                    args,
                    self.scales,
                    self.waveforms,
                    self.time_intervals,
                    self.colors,
                ),
                start_method='fork',
        ) as pool:
            pool.map(waveform_serial_job, worker_in, progress_bar=True)
        sns.set_style("whitegrid")

    def compute_per_cluster_mid_time_intervals(self, args, num_workers=20):

        def serial_job(shared_in, scale, i, sample_idxs):
            per_cluster_confident_idxs, get_time_interval = shared_in

            mid_time_intervals = []
            for sample_idx in sample_idxs:
                window_idx = per_cluster_confident_idxs[scale][str(
                    i)][sample_idx]
                time_interval = get_time_interval(
                    window_idx,
                    scale,
                    lmst=False,
                )

                time_interval = sum([t.timestamp for t in time_interval]) / 2.0
                time_interval = lmst_xtick(UTCDateTime(time_interval))
                mid_time_intervals.append(time_interval)

            return np.array(mid_time_intervals)

        mid_time_intervals = {
            scale: {
                str(i): []
                for i in range(args.ncluster)
            }
            for scale in self.scales
        }
        print(' [*] Computing waveform midtimes')
        for cluster in tqdm(range(args.ncluster), desc="Cluster loop"):
            for scale in tqdm(self.scales, desc="Scale loop", leave=False):
                split_idxs = np.array_split(
                    np.arange(
                        len(self.per_cluster_confident_idxs[scale][str(
                            cluster)])),
                    num_workers,
                    axis=0,
                )
                worker_in = [(scale, cluster, idxs) for idxs in split_idxs]

                with WorkerPool(
                        n_jobs=num_workers,
                        shared_objects=(
                            self.per_cluster_confident_idxs,
                            self.get_time_interval,
                        ),
                        start_method='fork',
                ) as pool:
                    mid_time_intervals[scale][str(cluster)] = pool.map(
                        serial_job, worker_in, progress_bar=False)
        return mid_time_intervals

    def plot_cluster_time_histograms(self, args):

        mid_time_intervals = self.compute_per_cluster_mid_time_intervals(args)

        np.save(
            os.path.join(plotsdir(args.experiment), 'mid_time_intervals.npy'),
            mid_time_intervals)
        sns.set_style("white")
        # Plot histogram of cluster times.
        print(' [*] Plotting time histograms')
        for cluster in tqdm(range(args.ncluster), desc="Cluster loop"):
            for scale in tqdm(self.scales, desc="Scale loop", leave=False):
                fig = plt.figure(figsize=(5, 1.5))
                sns.histplot(
                    mid_time_intervals[scale][str(cluster)],
                    color=self.colors[cluster % len(self.colors)],
                    stat="density",
                    element="step",
                    alpha=0.3,
                    binwidth=0.005,
                    kde=False,
                    label='Number of windows: ' +
                    str(mid_time_intervals[scale][str(cluster)].shape[0]),
                )
                ax = plt.gca()
                ax.set_ylabel('Density', fontsize=10)
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
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.legend(fontsize=10)
                plt.savefig(
                    os.path.join(
                        plotsdir(
                            os.path.join(args.experiment, 'scale_' + scale,
                                         'time_histograms')),
                        'time_histogram_cluster-' + str(cluster) + '.png',
                    ),
                    format="png",
                    bbox_inches="tight",
                    dpi=200,
                    pad_inches=.02,
                )
                plt.close(fig)
        sns.set_style("whitegrid")

    def centroid_waveforms(self, args):
        """Compute centroid waveform for each cluster.

        Args:
            waveforms: (array) array containing the waveforms

        Returns:
            centroid_waveforms: (array) array containing the centroid waveforms
        """
        sns.set_style("darkgrid")
        self.load_per_scale_per_cluster_waveforms(
            args,
            sample_size=200,
            overlap=False,
        )
        clustert_colors = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ]
        print(' [*] Computing and plotting centroid and centered waveforms')
        for scale in tqdm(self.scales, desc="Scale loop"):
            fig_spec, axes_spec = plt.subplots(
                nrows=3,
                sharex=True,
                figsize=(12, 12),
            )
            for cluster in tqdm(
                    range(args.ncluster),
                    desc="Cluster loop",
                    leave=False,
            ):
                # Extract waveforms for each cluster and put in a 3D array.
                waves = np.array(self.waveforms[scale][str(cluster)])

                # Normalize waveforms.
                for i in range(waves.shape[0]):
                    for j in range(waves.shape[1]):
                        waves[i,
                              j, :] = waves[i, j, :] - np.mean(waves[i, j, :])
                        waves[i,
                              j, :] = waves[i, j, :] / np.std(waves[i, j, :])
                rolled_waveforms = np.full_like(waves, np.nan)
                # rolled_waveforms = np.zeros_like(waves)
                corr_coefs = np.zeros((waves.shape[0]))
                corr_coefs[0] = waves.shape[1] * 1.0
                rolled_waveforms[0, ...] = waves[0, ...]
                bs_waveform = waves[0, ...]

                for i in range(1, waves.shape[0]):
                    correlation = 0.0
                    for j in range(waves.shape[1]):
                        correlation += correlate(
                            bs_waveform[j, :],
                            waves[i, j, :],
                            mode="same",
                        )
                    correlation /= waves.shape[1]
                    lags = correlation_lags(
                        bs_waveform.shape[-1],
                        waves.shape[-1],
                        mode="same",
                    )
                    lag = lags[np.argmax(correlation)]
                    for j in range(waves.shape[1]):
                        rolled_waveforms[i, j, :] = roll_nanpad(
                            waves[i, j, :], lag)
                        corr_coefs[i] += np.ma.corrcoef(
                            np.ma.masked_where(np.isnan(bs_waveform[j, :]),
                                               bs_waveform[j, :]),
                            np.ma.masked_where(
                                np.isnan(rolled_waveforms[i, j, :]),
                                rolled_waveforms[i, j, :]),
                        )[0, 1]

                corr_coefs = corr_coefs / waves.shape[1]
                centroid_waveforms = np.ma.zeros(
                    (rolled_waveforms.shape[1], rolled_waveforms.shape[2]), )
                for i in range(centroid_waveforms.shape[0]):

                    centroid_waveforms[i, :] = np.ma.average(
                        np.ma.masked_where(
                            np.isnan(rolled_waveforms[:, i, :]),
                            rolled_waveforms[:, i, :],
                        ),
                        weights=corr_coefs,
                        axis=0,
                    )

                fig, axes = plt.subplots(
                    nrows=3,
                    sharex=True,
                    figsize=(12, 12),
                )
                y_labels = ['U', 'V', 'W']

                for j in range(centroid_waveforms.shape[0]):
                    # Plot waveforms.
                    axes[j].plot(
                        np.linspace(
                            -centroid_waveforms.shape[1] / 40,
                            centroid_waveforms.shape[1] / 40,
                            num=centroid_waveforms.shape[1],
                            endpoint=True,
                        ),
                        np.ma.masked_where(
                            np.isnan(centroid_waveforms[j, :]),
                            centroid_waveforms[j, :],
                        ),
                        color=self.colors[cluster % len(self.colors)],
                        lw=1.0,
                        alpha=0.9,
                    )

                    axes[j].set_ylim([
                        np.ma.min(
                            np.ma.masked_where(
                                np.isnan(centroid_waveforms[j, :]),
                                centroid_waveforms[j, :]).reshape(-1)),
                        np.ma.max(
                            np.ma.masked_where(
                                np.isnan(centroid_waveforms[j, :]),
                                centroid_waveforms[j, :]).reshape(-1))
                    ])
                    axes[j].set_yticklabels([])
                    axes[j].tick_params(
                        axis='both',
                        which='major',
                    )
                    axes[j].grid(False)
                    axes[j].set_ylabel(y_labels[j], )

                axes[0].set_title(
                    'Centroid Waveform for cluster {}'.format(cluster))
                axes[2].set_xlabel('Time (s)', )
                axes[2].set_xlim(
                    -centroid_waveforms.shape[1] / 40,
                    centroid_waveforms.shape[1] / 40,
                )
                fig.savefig(
                    os.path.join(
                        plotsdir(
                            os.path.join(args.experiment, 'scale_' + scale,
                                         'centroid_waveforms')),
                        'centroid_waveform_' + str(cluster) + '.png',
                    ),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05,
                )
                plt.close(fig)

                for j in range(centroid_waveforms.shape[0]):
                    frequencies, asd = signal.welch(
                        centroid_waveforms[j, :],
                        fs=20,
                        nperseg=1024,
                    )

                    axes_spec[j].loglog(
                        frequencies,
                        np.sqrt(asd),
                        color=clustert_colors[cluster % len(clustert_colors)],
                        lw=1.0,
                        alpha=0.9,
                        label='cluster ' + str(cluster),
                    )

                    axes_spec[j].set_yticklabels([])
                    axes_spec[j].tick_params(
                        axis='both',
                        which='major',
                    )
                    axes_spec[j].grid(True)
                    axes_spec[j].set_ylabel(y_labels[j])

                axes_spec[0].set_title(
                    'Spectral amplitude of centroid waveforms')
                axes_spec[2].set_xlabel('Frequency (Hz)', )
                # add legend
                axes_spec[0].legend(fontsize=10, ncols=4)
                fig_spec.savefig(
                    os.path.join(
                        plotsdir(
                            os.path.join(args.experiment, 'scale_' + scale,
                                         'spectral_amplitude')),
                        'spectral_amplitude_' + str(scale) + '.png',
                    ),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05,
                )

                num_waveforms = 10
                dy = 1.6
                largest_corr = np.argsort(corr_coefs,
                                          axis=0)[::-1][:num_waveforms]

                fig, ax = plt.subplots(1,
                                       centroid_waveforms.shape[0],
                                       figsize=(18, 12),
                                       sharey=True)

                for i in range(rolled_waveforms.shape[1]):
                    for j in range(
                            min(num_waveforms, largest_corr.shape[0],
                                rolled_waveforms.shape[0])):
                        normalized_rolled = np.ma.masked_where(
                            np.isnan(rolled_waveforms[largest_corr[j], i, :]),
                            rolled_waveforms[largest_corr[j], i, :],
                        )
                        normalized_rolled = (
                            normalized_rolled /
                            np.ma.max(np.ma.abs(normalized_rolled)) - j * dy)
                        ax[i].plot(
                            np.linspace(
                                -rolled_waveforms.shape[-1] / 40,
                                rolled_waveforms.shape[-1] / 40,
                                num=rolled_waveforms.shape[-1],
                                endpoint=True,
                            ),
                            normalized_rolled,
                            color=self.colors[cluster % len(self.colors)],
                            lw=1.0,
                            alpha=0.9,
                        )
                        ax[i].axes.yaxis.set_visible(False)
                        ax[i].set_xlabel('Time (s)')
                        ax[i].set_xlim(-rolled_waveforms.shape[-1] / 40,
                                       rolled_waveforms.shape[-1] / 40)
                        ax[i].set_ylim(-(num_waveforms - 1) * dy - 1.5, 1.5)
                ax[0].set_title('U')
                ax[1].set_title('V')
                ax[2].set_title('W')
                fig.suptitle('Cluster {}, aligned waveforms'.format(cluster))
                fig.subplots_adjust(top=0.80)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        plotsdir(
                            os.path.join(args.experiment, 'scale_' + scale,
                                         'aligned_waveforms')),
                        'aligned_waveforms_cluster_' + str(cluster) + '.png',
                    ),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05,
                )
                plt.close(fig)

            plt.close(fig_spec)

        sns.set_style("whitegrid")

    def plot_latent_space(self, args):
        """Plot the latent space learnt by the model

        Args:
            data: (array) corresponding array containing the data
            labels: (array) corresponding array containing the labels
            save: (bool) whether to save the latent space plot

        Returns:
            fig: (figure) plot of the latent space
        """
        # DO NOT PLACE THIS IMPORT AT THE BEGINNING OF THE FILE. umap alters the
        # environment variables, which causes errors when using multiprocessing.
        import umap
        font = {'family': 'serif', 'style': 'normal', 'size': 12}
        matplotlib.rc('font', **font)

        # Colors for each cluster.
        colors = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf',
        ]
        per_point_color = {
            scale: [
                colors[cluster_idx]
                for cluster_idx in self.cluster_membership[scale][:, 0]
            ]
            for scale in self.scales
        }

        # Serial jobs for computing UMAP features.
        def serial_job(latent_features, scale):
            umap_features = {
                scale:
                umap.UMAP(
                    n_neighbors=300,
                    min_dist=5e-1,
                    metric='euclidean',
                ).fit_transform(latent_features[scale][:, 0, :].numpy())
            }
            return umap_features

        # Compute UMAP features.
        with WorkerPool(
                n_jobs=len(self.scales),
                shared_objects=self.latent_features,
                start_method='fork',
        ) as pool:
            feature_list = pool.map(
                serial_job,
                self.scales,
                progress_bar=False,
            )

        umap_features = {}
        for item in feature_list:
            umap_features.update(item)

        # Extract UMAP features of indices with label "BROADBAND"
        label_idx = {scale: [] for scale in self.scales}
        for scale in self.scales:
            for idx in self.window_labels[scale].keys():
                for event_type in args.event_type:
                    for event_quality in args.event_quality:
                        if (event_type + '_' + event_quality
                            ) in self.window_labels[scale][idx]:
                            label_idx[scale].append(int(idx))

        from IPython import embed
        embed()

        for scale in self.scales:
            fig = plt.figure(figsize=(8, 4))
            plt.scatter(
                umap_features[scale][:, 0],
                umap_features[scale][:, 1],
                marker='o',
                c=per_point_color[scale],
                edgecolor='none',
                s=3,
                alpha=0.3,
            )

            if len(label_idx[scale]) > 0:

                plt.scatter(
                    umap_features[scale][label_idx[scale], 0],
                    umap_features[scale][label_idx[scale], 1],
                    marker='*',
                    c="#333333",
                    edgecolor="k",
                    s=40,
                    alpha=0.6,
                )

            legend_elements = []
            for i, label in enumerate(range(args.ncluster)):
                custom_legend = plt.Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    label=f'{label}',
                    markerfacecolor=colors[i],
                    markersize=10,
                )
                legend_elements.append(custom_legend)

            plt.legend(handles=legend_elements, fontsize=8)
            plt.title("Latent samples at scale {}".format(scale))
            fig.savefig(
                os.path.join(
                    plotsdir(
                        os.path.join(
                            args.experiment,
                            'latent_space_visualization' + '_' +
                            '-'.join(args.event_type) + '_' +
                            '-'.join(args.event_quality),
                        )),
                    'umap_scale-' + scale + '.png',
                ),
                format="png",
                bbox_inches="tight",
                dpi=300,
                pad_inches=.05,
            )
            plt.close(fig)

        font = {'family': 'serif', 'style': 'normal', 'size': 18}
        matplotlib.rc('font', **font)
