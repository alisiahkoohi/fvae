import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import datetime
import obspy
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
                          roll_zeropad, get_waveform_path_from_time_interval)

# Datastream merge method.
MERGE_METHOD = 1
FILL_VALUE = 'interpolate'


class SnippetExtractor(object):
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
        # Device to perform computations on.
        self.device = device

        (self.cluster_membership, self.cluster_membership_prob,
         self.confident_idxs,
         self.per_cluster_confident_idxs) = self.evaluate_model(
             args, data_loader)

    def get_waveform(self, window_idx, scale):
        window_time_interval = self.get_time_interval(window_idx,
                                                      scale,
                                                      lmst=False)

        filepath = get_waveform_path_from_time_interval(*window_time_interval)

        # Extract some properties of the data to setup HDF5 file.
        data_stream = obspy.read(filepath)
        data_stream = data_stream.merge(method=MERGE_METHOD,
                                        fill_value=FILL_VALUE)
        data_stream = data_stream.detrend(type='spline',
                                          order=2,
                                          dspline=2000,
                                          plot=False)
        data_stream = data_stream.slice(*window_time_interval)

        waveform = np.stack([td.data[-int(scale):] for td in data_stream])

        # Return the required subwindow.
        return waveform.astype(np.float32)

    def get_time_interval(self,
                          window_idx,
                          scale,
                          lmst=True,
                          get_full_interval=False,
                          timescale=None):

        if timescale is None:
            timescale = scale
        # Extract window time interval.
        window_time_interval = self.dataset.get_time_interval([window_idx])[0]
        # Number of subwindows in the given scale.
        scales = [int(s) for s in self.scales]
        num_sub_windows = max(scales) // int(timescale)

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
            return (create_lmst_xticks(*window_time_interval,
                                       time_zone='LMST',
                                       window_size=int(timescale)),
                    window_time_interval)

        if get_full_interval:
            # Return the required time interval.
            return window_time_interval, subintervals
        else:
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

    def multi_cluster_snippets(self, num_snippets=30):

        list_of_per_scale_confident_idxs = []
        for scale in self.scales:
            list_of_per_scale_confident_idxs.append(
                list(self.confident_idxs[scale]))

        def find_first_n_distinct(lists, n):
            distinct_values = set()
            result = []
            list_associations = {}
            for i in range(len(lists[0])):
                for j in range(len(lists)):
                    if lists[j][i] in distinct_values:
                        distinct_values.remove(lists[j][i])
                        list_associations[lists[j][i]].append(j)
                        result.append(
                            (lists[j][i], list_associations[lists[j][i]]))
                        if len(result) == n:
                            return result
                    else:
                        distinct_values.add(lists[j][i])
                        if lists[j][i] not in list_associations:
                            list_associations[lists[j][i]] = []
                        list_associations[lists[j][i]].append(j)

        snippet_idxs = find_first_n_distinct(list_of_per_scale_confident_idxs,
                                             num_snippets)

        multi_cluster_snippet = {}
        for idx, list_idx in snippet_idxs:
            idx_list = [idx]
            multi_cluster_snippet[idx] = {
                'scale': [],
                'cluster': [],
                'waveform': [],
                'time_intervals': []
            }
            for scale_idx in list_idx[:2]:
                multi_cluster_snippet[idx]['scale'].append(
                    int(self.scales[scale_idx]))
                multi_cluster_snippet[idx]['cluster'].append(
                    self.cluster_membership[self.scales[scale_idx]]
                    [self.confident_idxs[self.scales[scale_idx]][idx]].item())
                multi_cluster_snippet[idx]['waveform'].append(
                    self.get_waveform(idx, self.scales[scale_idx]))
                multi_cluster_snippet[idx]['time_intervals'].append(
                    self.get_time_interval(idx, self.scales[scale_idx])[1])

        return multi_cluster_snippet

    def waveforms_per_scale_cluster(self,
                                    args,
                                    cluster_idxs,
                                    scale_idxs,
                                    sample_size=5,
                                    time_preference=None,
                                    get_full_interval=False,
                                    timescale=None):
        """Plot waveforms.
        """

        def do_overlap(pair1, pair2):
            start1, end1 = pair1
            start2, end2 = pair2

            # Check for all types of overlap
            return (start1 <= start2 <= end1 or start1 <= end2 <= end1
                    or start2 <= start1 <= end2 or start2 <= end1 <= end2)

        def is_close(pair1, pair2):
            start1, end1 = pair1
            start2, end2 = pair2

            # Calculate the time difference between the two intervals in hours
            time_difference1 = abs((start1 - end2) / 3600)
            time_difference2 = abs((start2 - end1) / 3600)

            # Check if the time difference is within the desired range
            if time_difference1 <= 3 or time_difference2 <= 1:
                return True
            else:
                return False

        waveforms = []
        time_intervals = []

        for scale, i in zip(scale_idxs, cluster_idxs):
            scale = str(scale)

            print('Reading waveforms for cluster {}, scale {}'.format(
                i, scale))
            utc_time_intervals = []
            window_idx_list = []

            for sample_idx in range(
                    len(self.per_cluster_confident_idxs[scale][str(i)])):
                window_idx = self.per_cluster_confident_idxs[scale][str(
                    i)][sample_idx]
                utc_interval = self.get_time_interval(window_idx, scale)[1]
                should_add = True

                if time_preference is not None:
                    if not is_close(utc_interval, time_preference):
                        should_add = False

                for interval in utc_time_intervals:
                    if do_overlap(interval, utc_interval):
                        should_add = False
                        break

                if should_add:
                    utc_time_intervals.append(utc_interval)
                    window_idx_list.append(window_idx)

                if len(window_idx_list) == sample_size:
                    break

            for window_idx in window_idx_list:
                waveforms.append(self.get_waveform(window_idx, scale))
                if get_full_interval:
                    time_intervals.append(
                        self.get_time_interval(window_idx,
                                            scale,
                                            lmst=False,
                                            get_full_interval=get_full_interval,
                                            timescale=timescale)[1])
                else:
                    time_intervals.append(
                        self.get_time_interval(window_idx,
                                            scale,
                                            lmst=False,
                                            get_full_interval=get_full_interval,
                                            timescale=timescale))
        return np.array(waveforms), time_intervals
