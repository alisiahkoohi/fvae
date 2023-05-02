import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
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
                          roll_zeropad)

sns.set_style("whitegrid")
font = {'family': 'serif', 'style': 'normal', 'size': 18}
matplotlib.rc('font', **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")

SAMPLING_RATE = 20


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
         self.confident_idxs,
         self.per_cluster_confident_idxs) = self.evaluate_model(
             args, data_loader)

        # Colors to be used for visualizing different clusters.
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        # All the possible labels that data has. Used for evaluating the
        # clustering performance.
        self.labels = [
            'HIGH_FREQUENCY', 'VERY_HIGH_FREQUENCY', 'LOW_FREQUENCY', '2.4_HZ',
            'SUPER_HIGH_FREQUENCY', 'BROADBAND'
        ]

    def latent_features(self, args, data_loader):
        """Obtain latent features learnt by the model

        Args:
            data_loader: (DataLoader) loader containing the data
            return_labels: (boolean) whether to return true labels or not

        Returns:
           features: (array) array containing the features from the data
        """
        N = len(data_loader.dataset)
        features = np.zeros([N, args.latent_dim])
        clusters = np.zeros([N])
        counter = 0
        with torch.no_grad():
            for idx in data_loader:
                # Load data batch.
                x = self.dataset.sample_data(idx, 'scat_cov')
                x = {scale: x[scale].to(self.device) for scale in x.keys()}

                # flatten data
                output = self.network.inference(x, self.network.gumbel_temp,
                                                self.network.hard_gumbel)
                latent_feat = output['mean']
                cluster_membership = output['logits'].argmax(axis=1)

                features[counter:counter +
                         x.size(0), :] = latent_feat.cpu().detach().numpy()[
                             ...]
                clusters[counter:counter + x.size(0)] = cluster_membership.cpu(
                ).detach().numpy()[...]

                counter += x.shape[0]

        return features, clusters

    def get_waveform(self, window_idx, subwindow_idx, scale):
        # Extract waveform with shape num_components (e.g., 3) x window_size.
        waveform = self.dataset.sample_data([window_idx],
                                            type='waveform')[0].numpy()
        # Number of subwindows in the given scale.
        num_sub_windows = self.dataset.data['scat_cov'][scale].shape[1]
        # Reshape waveform to subwindows: num_components x num_sub_windows x
        # subwindow_size
        waveform = waveform.reshape(waveform.shape[0], num_sub_windows, -1)

        # Return the required subwindow.
        return waveform[:, subwindow_idx, :]

    def get_time_interval(self, window_idx, scale, lmst=True):
        # Extract window time interval.
        window_time_interval = self.dataset.get_time_interval([window_idx])[0]
        # Number of subwindows in the given scale.
        scales = [int(s) for s in self.scales]
        num_sub_windows = max(scales) // int(scale)

        # Example start and end times.
        start_time = window_time_interval[0]
        end_time = window_time_interval[1]

        # Calculate total time duration.
        duration = end_time - start_time

        # Use linspace to create subintervals.
        subinterval_starts = np.linspace(start_time.timestamp,
                                         end_time.timestamp,
                                         num=num_sub_windows + 1)
        subintervals = [
            (UTCDateTime(t1), UTCDateTime(t2))
            for t1, t2 in zip(subinterval_starts[:-1], subinterval_starts[1:])
        ]

        # Select the time interval associated with the given subwindow_idx.
        window_time_interval = subintervals[-1]

        if lmst:
            # Convert to LMST format, usable by matplotlib.
            window_time_interval = create_lmst_xticks(*window_time_interval,
                                                      time_zone='LMST',
                                                      window_size=int(scale))

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

        # Extract cluster memberships.
        for i_idx, idx in enumerate(data_loader):
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
                per_cluster_confident_idxs)

    def plot_waveforms(self, args, sample_size=5):
        """Plot waveforms.
        """

        waveforms = {}
        time_intervals = {}
        for scale in self.scales:
            waveforms[scale] = {}
            time_intervals[scale] = {}
            for i in range(args.ncluster):
                waveforms[scale][str(i)] = []
                time_intervals[scale][str(i)] = []
                for sample_idx in range(
                        min(
                            sample_size,
                            len(self.per_cluster_confident_idxs[scale][str(
                                i)]))):
                    (window_idx,
                     subwindow_idx) = self.per_cluster_confident_idxs[scale][
                         str(i)][sample_idx]
                    waveforms[scale][str(i)].append(
                        self.get_waveform(window_idx, subwindow_idx, scale))
                    time_intervals[scale][str(i)].append(
                        self.get_time_interval(window_idx, subwindow_idx,
                                               scale))

        # Plot Fourier transforms for each cluster.
        for cluster in range(args.ncluster):
            print('Plotting Fourier transforms for cluster {}'.format(cluster))
            for scale in tqdm(self.scales):
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
                        ax.tick_params(axis='both', which='major', labelsize=8)
                        plt.savefig(os.path.join(
                            plotsdir(
                                os.path.join(args.experiment, 'scale_' + scale,
                                             'cluster_' + str(cluster),
                                             'component_' + str(comp))),
                            'fourier_transform_{}.png'.format(sample_idx)),
                                    format="png",
                                    bbox_inches="tight",
                                    dpi=200,
                                    pad_inches=.02)
                        plt.close(fig)

        # Plot spectrograms for each cluster.
        for cluster in range(args.ncluster):
            print('Plotting spectrograms for cluster {}'.format(cluster))
            for scale in tqdm(self.scales):
                for sample_idx, waveform in enumerate(
                        waveforms[scale][str(cluster)]):
                    for comp in range(waveform.shape[0]):
                        fig = plt.figure(figsize=(7, 2))
                        # Plot spectrogram.
                        nperseg = min(256, int(scale) // 4)
                        plt.specgram(waveform[comp, :],
                                     NFFT=nperseg,
                                     noverlap=nperseg // 8,
                                     Fs=SAMPLING_RATE,
                                     mode='magnitude',
                                     cmap='RdYlBu_r')
                        ax = plt.gca()
                        plt.ylim([0, SAMPLING_RATE / 2])
                        ax.set_xticklabels([])
                        ax.set_ylabel('Frequency (Hz)', fontsize=10)
                        ax.grid(False)
                        ax.tick_params(axis='both', which='major', labelsize=8)
                        plt.savefig(os.path.join(
                            plotsdir(
                                os.path.join(args.experiment, 'scale_' + scale,
                                             'cluster_' + str(cluster),
                                             'component_' + str(comp))),
                            'spectrogram_{}.png'.format(sample_idx)),
                                    format="png",
                                    bbox_inches="tight",
                                    dpi=200,
                                    pad_inches=.02)
                        plt.close(fig)

        # Plot waveforms for each cluster.
        for cluster in range(args.ncluster):
            print('Plotting waveforms for cluster {}'.format(cluster))
            for scale in tqdm(self.scales):
                for sample_idx, waveform in enumerate(
                        waveforms[scale][str(cluster)]):
                    fig = plt.figure(figsize=(7, 2))
                    for comp in range(waveform.shape[0]):
                        # Plot waveforms.
                        waveform[comp, :] = waveform[comp, :] / np.linalg.norm(
                            waveform[comp, :])
                        plt.plot_date(
                            time_intervals[scale][str(cluster)][sample_idx],
                            waveform[comp, :],
                            xdate=True,
                            color=self.colors[cluster % len(self.colors)],
                            lw=1.2,
                            alpha=1.0 / (comp + 2),
                            fmt='')
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(
                        matplotlib.dates.MinuteLocator(interval=int(
                            np.ceil(int(scale) / SAMPLING_RATE / 60 / 5))))
                    ax.xaxis.set_major_formatter(
                        matplotlib.dates.DateFormatter('%H:%M'))
                    ax.set_yticklabels([])
                    plt.ylim(
                        [min(waveform.reshape(-1)),
                         max(waveform.reshape(-1))])
                    plt.xlim([
                        time_intervals[scale][str(cluster)][sample_idx][0],
                        time_intervals[scale][str(cluster)][sample_idx][-1]
                    ])
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plt.savefig(os.path.join(
                        plotsdir(
                            os.path.join(args.experiment, 'scale_' + scale,
                                         'cluster_' + str(cluster))),
                        'waveform_{}.png'.format(sample_idx)),
                                format="png",
                                bbox_inches="tight",
                                dpi=200,
                                pad_inches=.02)
                    plt.close(fig)

    def compute_per_cluster_mid_time_intervals(self, args, num_workers=8):

        def serial_job(shared_in, scale, i, sample_idxs):
            per_cluster_confident_idxs, get_time_interval = shared_in

            mid_time_intervals = []
            for sample_idx in sample_idxs:
                window_idx = per_cluster_confident_idxs[scale][str(
                    i)][sample_idx]
                time_interval = get_time_interval(window_idx,
                                                  scale,
                                                  lmst=False)
                time_interval = [lmst_xtick(t) for t in time_interval]
                time_interval = [
                    matplotlib.dates.date2num(t) for t in time_interval
                ]
                time_interval = np.array(time_interval).mean(-1)
                mid_time_intervals.append(time_interval)

            return np.array(mid_time_intervals)

        mid_time_intervals = {
            scale: {
                str(i): []
                for i in range(args.ncluster)
            }
            for scale in self.scales
        }

        for cluster in tqdm(range(args.ncluster)):
            for scale in tqdm(self.scales):
                split_idxs = np.array_split(np.arange(
                    len(self.per_cluster_confident_idxs[scale][str(cluster)])),
                                            num_workers,
                                            axis=0)
                worker_in = [(scale, cluster, idxs) for idxs in split_idxs]

                with WorkerPool(
                        n_jobs=num_workers,
                        shared_objects=(self.per_cluster_confident_idxs,
                                        self.get_time_interval),
                        start_method='fork') as pool:
                    mid_time_intervals[scale][str(cluster)] = pool.map(
                        serial_job, worker_in, progress_bar=False)
        return mid_time_intervals

    def plot_cluster_time_histograms(self, args):

        mid_time_intervals = self.compute_per_cluster_mid_time_intervals(args)

        # Plot histogram of cluster times.
        for cluster in range(args.ncluster):
            print('Plotting time histograms for cluster {}'.format(cluster))
            for scale in tqdm(self.scales):
                fig = plt.figure(figsize=(12, 2))
                sns.histplot(mid_time_intervals[scale][str(cluster)],
                             color=self.colors[cluster % len(self.colors)],
                             element="step",
                             alpha=0.3,
                             binwidth=0.005,
                             label='cluster ' + str(cluster) + ' - ' +
                             str(len(mid_time_intervals[scale][str(cluster)])))
                ax = plt.gca()
                ax.xaxis.set_major_locator(
                    matplotlib.dates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(
                    matplotlib.dates.DateFormatter('%H'))
                ax.legend()
                ax.set_yticklabels([])
                ax.tick_params(axis='both', which='major', labelsize=8)
                plt.savefig(os.path.join(
                    plotsdir(os.path.join(args.experiment, 'scale_' + scale)),
                    'time_histogram_cluster-' + str(cluster) + '.png'),
                            format="png",
                            bbox_inches="tight",
                            dpi=200,
                            pad_inches=.02)
                plt.close(fig)
        from facvae.utils import upload_results
        upload_results(args, flag='--progress')

    def centroid_waveform(self, args, waveforms, cluster_idx):
        """Compute centroid waveform for each cluster.

        Args:
            waveforms: (array) array containing the waveforms

        Returns:
            centroid_waveforms: (array) array containing the centroid waveforms
        """
        for i in range(waveforms.shape[0]):
            for j in range(waveforms.shape[1]):
                waveforms[i, j, :] = waveforms[i, j, :] - np.mean(
                    waveforms[i, j, :])
                waveforms[i, j, :] = waveforms[i, j, :] / np.std(
                    waveforms[i, j, :])
        rolled_waveforms = np.zeros_like(waveforms)
        corr_coefs = np.ones((waveforms.shape[0], waveforms.shape[1]))
        rolled_waveforms[-1, ...] = waveforms[-1, ...]
        bs_waveform = waveforms[-1, ...]

        for i in range(waveforms.shape[0] - 1):
            for j in range(waveforms.shape[1]):
                correlation = correlate(bs_waveform[j, :],
                                        waveforms[i, j, :],
                                        mode="same")
                lags = correlation_lags(bs_waveform[j, :].size,
                                        waveforms[i, j, :].size,
                                        mode="same")
                lag = lags[np.argmax(correlation)]
                rolled_waveforms[i,
                                 j, :] = roll_zeropad(waveforms[i, j, :], lag)
                corr_coefs[i, j] = np.corrcoef(bs_waveform[
                    j,
                ], rolled_waveforms[i, j, :])[0, 1]

        centroid_waveforms = np.zeros(
            (rolled_waveforms.shape[1], rolled_waveforms.shape[2]))
        for i in range(centroid_waveforms.shape[0]):
            centroid_waveforms[i, :] = np.average(rolled_waveforms[:, i, :],
                                                  weights=corr_coefs[:, i],
                                                  axis=0)

        fig, ax = plt.subplots(centroid_waveforms.shape[0],
                               1,
                               figsize=(12, 12),
                               sharex=True)

        ax[0].plot(
            np.linspace(0,
                        centroid_waveforms.shape[1] / 20,
                        num=centroid_waveforms.shape[1],
                        endpoint=True),
            centroid_waveforms[0, :],
            color=self.colors[cluster_idx % 10],
            linewidth=1,
        )
        ax[0].set_ylabel('U')
        ax[0].set_title('Centroid Waveform for cluster {}'.format(cluster_idx))
        ax[1].plot(
            np.linspace(0,
                        centroid_waveforms.shape[1] / 20,
                        num=centroid_waveforms.shape[1],
                        endpoint=True),
            centroid_waveforms[1, :],
            color=self.colors[cluster_idx % 10],
            linewidth=1,
        )
        ax[1].set_ylabel('V')
        ax[2].plot(
            np.linspace(0,
                        centroid_waveforms.shape[1] / 20,
                        num=centroid_waveforms.shape[1],
                        endpoint=True),
            centroid_waveforms[2, :],
            color=self.colors[cluster_idx % 10],
            linewidth=1,
        )
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('W')
        ax[2].set_xlim(0, centroid_waveforms.shape[1] / 20)
        fig.savefig(os.path.join(
            plotsdir(args.experiment),
            'centroid_waveform_{}.png'.format(cluster_idx)),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

        num_waveforms = 20
        dy = 1.8
        largest_corr = np.argsort(corr_coefs, axis=0)[::-1][:num_waveforms, :]
        fig, ax = plt.subplots(1,
                               centroid_waveforms.shape[0],
                               figsize=(18, 12),
                               sharey=True)
        for i in range(rolled_waveforms.shape[1]):
            for j in range(num_waveforms):
                ax[i].plot(
                    np.linspace(0,
                                rolled_waveforms.shape[2] / 20,
                                num=rolled_waveforms.shape[2],
                                endpoint=True),
                    rolled_waveforms[largest_corr[j, i], i, :] /
                    np.max(np.abs(rolled_waveforms[largest_corr[j, i], i, :]))
                    - j * dy,
                    color=self.colors[cluster_idx % 10],
                    linewidth=1,
                    alpha=0.7,
                )
                ax[i].axes.yaxis.set_visible(False)
                ax[i].set_xlabel('Time (s)')
                ax[2].set_xlim(0, centroid_waveforms.shape[1] / 20)
                ax[2].set_ylim(-(num_waveforms - 1) * dy - 1.5, 1.5)
        ax[0].set_title('U')
        ax[1].set_title('V')
        ax[2].set_title('W')
        fig.suptitle('Cluster {}, aligned waveforms'.format(cluster_idx))
        fig.subplots_adjust(top=0.80)
        fig.tight_layout()
        fig.savefig(os.path.join(
            plotsdir(args.experiment),
            'aligned_waveforms_cluster_{}.png'.format(cluster_idx)),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

    def reconstruct_data(self, args, data_loader, sample_size=5):
        """Reconstruct Data

        Args:
            data_loader: (DataLoader) loader containing the data
            sample_size: (int) size of random data to consider from data_loader

        Returns:
            reconstructed: (array) array containing the reconstructed data
        """
        # Sample random data from loader
        x = self.dataset.sample_data(next(iter(data_loader)), 'scat_cov')
        indices = np.random.randint(0, x[0].shape[0], size=sample_size)
        x = [x[i][indices, ...] for i in range(len(x))]
        x = [x[i].to(self.device) for i in range(len(x))]

        # Obtain reconstructed data.
        with torch.no_grad():
            output = self.network(x)
            x_rec = output['x_rec']

        x = [x[i].cpu() for i in range(len(x))]
        x_rec = [x_rec[i].cpu() for i in range(len(x_rec))]

        if x[0].shape[-1] > 2:
            fig, ax = plt.subplots(1, sample_size, figsize=(25, 5))
            for i in range(sample_size):
                ax[i].plot(x[0][i, 0, :],
                           lw=.8,
                           alpha=1,
                           color='k',
                           label='original')
                ax[i].plot(x_rec[0][i, 0, :],
                           lw=.8,
                           alpha=0.5,
                           color='r',
                           label='reconstructed')
            plt.legend()
            plt.savefig(os.path.join(plotsdir(args.experiment), 'rec.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=.05)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].scatter(x[:, 0],
                          x[:, 1],
                          s=2,
                          alpha=0.5,
                          color='k',
                          label='original')
            ax[1].scatter(x_rec[:, 0],
                          x_rec[:, 1],
                          s=2,
                          alpha=0.5,
                          color='r',
                          label='reconstructed')
            plt.legend()
            plt.savefig(os.path.join(plotsdir(args.experiment), 'rec.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=.05)
            plt.close(fig)

        if args.dataset == 'mars':
            x = [
                self.dataset.unnormalize(x[i], args.type[i])
                for i in range(len(x))
            ]
            x_rec = [
                self.dataset.unnormalize(x_rec[i], args.type[i])
                for i in range(len(x_rec))
            ]

            fig, ax = plt.subplots(1, sample_size, figsize=(25, 5))
            for i in range(sample_size):
                ax[i].plot(x[0][i, 0, :],
                           lw=.8,
                           alpha=1,
                           color='k',
                           label='original')
                ax[i].plot(x_rec[0][i, 0, :],
                           lw=.8,
                           alpha=0.5,
                           color='r',
                           label='reconstructed')
            plt.legend()
            plt.savefig(os.path.join(plotsdir(args.experiment),
                                     'rec_unnormalized.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=.05)
            plt.close(fig)

    def plot_latent_space(self, args, data_loader, save=False):
        """Plot the latent space learnt by the model

        Args:
            data: (array) corresponding array containing the data
            labels: (array) corresponding array containing the labels
            save: (bool) whether to save the latent space plot

        Returns:
            fig: (figure) plot of the latent space
        """
        # obtain the latent features
        features, clusters = self.latent_features(args, data_loader)
        features_tsne = TSNE(n_components=2,
                             learning_rate='auto',
                             init='pca',
                             early_exaggeration=10,
                             perplexity=200).fit_transform(features)

        # plot only the first 2 dimensions
        # cmap = plt.cm.get_cmap('hsv', args.ncluster)
        label_colors = {i: self.colors[i % 10] for i in range(args.ncluster)}
        colors = [label_colors[int(i)] for i in clusters]

        if features.shape[-1] > 2:
            features_pca = PCA(n_components=2).fit_transform(features)
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(features_pca[:, 0],
                        features_pca[:, 1],
                        marker='o',
                        c=colors,
                        edgecolor='none',
                        cmap=plt.cm.get_cmap('jet', 10),
                        s=10)
            plt.title("Two dimensional PCA of the latent samples")
            plt.savefig(os.path.join(plotsdir(args.experiment),
                                     'pca_latent_space.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=.05)
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(features[:, 0],
                        features[:, 1],
                        marker='o',
                        c=colors,
                        edgecolor='none',
                        cmap=plt.cm.get_cmap('jet', 10),
                        s=10)

            plt.title("Latent samples")
            plt.savefig(os.path.join(plotsdir(args.experiment),
                                     'pca_latent_space.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=.05)
            plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        plt.scatter(features_tsne[:, 0],
                    features_tsne[:, 1],
                    marker='o',
                    c=colors,
                    edgecolor='none',
                    cmap=plt.cm.get_cmap('jet', 10),
                    s=10)
        plt.title("T-SNE visualization of the latent samples")
        plt.savefig(os.path.join(plotsdir(args.experiment),
                                 'latent_space_tsne.png'),
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=.05)
        plt.close(fig)

    def random_generation(self, args, num_elements=3):
        """Random generation for each category

        Args:
            num_elements: (int) number of elements to generate

        Returns:
            generated data according to num_elements
        """
        # categories for each element
        arr = np.array([])
        for i in range(args.ncluster):
            arr = np.hstack([arr, np.ones(num_elements) * i])
        indices = arr.astype(int).tolist()

        categorical = torch.nn.functional.one_hot(
            torch.tensor(indices), args.ncluster).float().to(self.device)
        # infer the gaussian distribution according to the category
        mean, var = self.network.generative.pzy(categorical)

        # gaussian random sample by using the mean and variance
        noise = torch.randn_like(var)
        std = torch.sqrt(var)
        gaussian = mean + noise * std

        # generate new samples with the given gaussian
        gaussian = gaussian.to(self.device)
        samples = self.network.generative.pxz(gaussian).cpu().detach().numpy()

        if samples.shape[-1] > 2:
            fig, ax = plt.subplots(num_elements,
                                   args.ncluster,
                                   figsize=(8 * args.ncluster,
                                            4 * args.ncluster))
            for i in range(args.ncluster):
                for j in range(num_elements):
                    ax[j, i].plot(samples[i * num_elements + j, 0, :],
                                  color=self.colors[i % 10],
                                  lw=1.2,
                                  alpha=0.8)
                    ax[j, i].set_title("Sample from cluster " + str(i))
            plt.savefig(os.path.join(plotsdir(args.experiment),
                                     'joint_samples.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=.05)
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(8, 6))
            for i in range(args.ncluster):
                plt.scatter(samples[i * num_elements:(i + 1) * num_elements,
                                    0],
                            samples[i * num_elements:(i + 1) * num_elements,
                                    1],
                            color=self.colors[i % 10],
                            s=2,
                            alpha=0.5)
            plt.title('Generated joint samples')
            # ax[i].axis('off')
            plt.savefig(os.path.join(plotsdir(args.experiment),
                                     'joint_samples.png'),
                        format="png",
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=.05)
            plt.close(fig)
