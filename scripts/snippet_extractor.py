import obspy
import numpy as np
import torch
from tqdm import tqdm

from facvae.utils import (
    create_lmst_xticks,
    get_waveform_path_from_time_interval,
    MarsMultiscaleDataset,
    lmst_xtick,
)
from scripts.facvae_trainer import FactorialVAETrainer

# Datastream merge method.
MERGE_METHOD = 1
FILL_VALUE = 'interpolate'

COMPONENT_TO_INDEX = {
    'U': 0,
    'V': 1,
    'W': 2,
}


class SnippetExtractor(object):
    """Class visualizing results of a fVAE training.
    """

    def __init__(self, args, dataset_path, device=torch.device('cpu')):

        # Device to perform computations on.
        self.device = device

        # Load data from the Mars dataset
        self.dataset = MarsMultiscaleDataset(
            dataset_path,
            0.90,
            scatcov_datasets=args.scales,
            load_to_memory=args.load_to_memory,
            normalize_data=args.normalize,
            filter_key=args.filter_key,
        )
        data_loader = torch.utils.data.DataLoader(
            self.dataset.test_idx,
            batch_size=args.batchsize,
            shuffle=False,
            drop_last=False,
        )

        # Initialize facvae trainer with the input arguments, dataset, and device
        facvae_trainer = FactorialVAETrainer(
            args,
            self.dataset,
            self.device,
        )

        # Load a saved checkpoint for testing.
        self.network = facvae_trainer.load_checkpoint(args, args.max_epoch - 1)
        self.network.gumbel_temp = np.maximum(
            args.init_temp * np.exp(-args.temp_decay * (args.max_epoch - 1)),
            args.min_temp,
        )
        self.network.eval()

        # Scales.
        self.scales = args.scales
        self.in_shape = {
            scale: self.dataset.shape['scat_cov'][scale]
            for scale in self.scales
        }

        self.per_cluster_confident_idxs = self.evaluate_model(
            args, data_loader)

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

        return per_cluster_confident_idxs

    def get_waveform(self, window_idx, scale, fine_scale=None):

        window_time_intervals = self.get_time_interval(
            window_idx,
            scale,
            lmst=False,
            fine_scale=fine_scale,
        )

        # Find filname of the waveform. Note that we only use the first time
        # interval in the list to find the filename as all the time intervals
        # inherently belong to the same day (even if fine_scake is not None)
        # since the largest window size (i.e., largest scale) does not span
        # multiple days.
        filepath = get_waveform_path_from_time_interval(
            *window_time_intervals[0])

        # Extract the data, merge the traces, and detrend to prepare for source
        # separation.
        data_stream = obspy.read(filepath)
        data_stream = data_stream.merge(method=MERGE_METHOD,
                                        fill_value=FILL_VALUE)
        data_stream = data_stream.detrend(type='spline',
                                          order=2,
                                          dspline=2000,
                                          plot=False)

        waveforms = []
        for window_time_interval in window_time_intervals:
            sliced_stream = data_stream.slice(*window_time_interval)
            waveforms.append(
                np.array([td.data[-int(scale):] for td in sliced_stream]))

        # Return the required subwindow.
        return np.stack(waveforms), window_time_intervals

    def get_time_interval(self, window_idx, scale, lmst=True, fine_scale=None):

        if fine_scale:
            # Extract window time interval.
            window_time_intervals = self.find_cross_scale_intersecting_windows(
                scale,
                window_idx,
                fine_scale,
            )
        else:
            window_time_intervals = self.dataset.get_time_interval(
                [window_idx],
                scale,
            )

        if lmst:
            # Convert to LMST format, usable by matplotlib.
            window_time_intervals = [
                create_lmst_xticks(
                    *interval,
                    time_zone='LMST',
                    window_size=int(fine_scale) if fine_scale else int(scale),
                ) for interval in window_time_intervals
            ]

        return window_time_intervals

    def find_cross_scale_intersecting_windows(
        self,
        coarse_scale,
        coarse_scale_window_idx,
        fine_scale,
    ):
        """Find the intersecting window indices between two scales.
        """

        if isinstance(fine_scale, int):
            fine_scale = str(fine_scale)

        coarse_scale_time_interval = self.get_time_interval(
            coarse_scale_window_idx,
            coarse_scale,
            lmst=False,
            fine_scale=None,
        )[0]

        min_scale = min([int(s) for s in self.scales])

        scale_ratio = int(coarse_scale) // min_scale
        scale_stride = int(fine_scale) // min_scale

        candidate_idxs = np.sort(
            np.arange(
                coarse_scale_window_idx,
                coarse_scale_window_idx - scale_ratio,
                -scale_stride,
                dtype=int,
            ))

        fine_scale_time_intervals = []
        for idx in candidate_idxs:
            try:
                # Obtaining candidate time interval.
                candidate_time_interval = self.get_time_interval(
                    idx,
                    fine_scale,
                    lmst=False,
                    fine_scale=None,
                )[0]
            except Exception:
                # Continue to the next iteration of the 'for idx' loop.
                continue

            if (coarse_scale_time_interval[0] <= candidate_time_interval[0]
                    and candidate_time_interval[1]
                    <= coarse_scale_time_interval[1]):
                fine_scale_time_intervals.append(candidate_time_interval)

        return fine_scale_time_intervals

    def waveforms_per_scale_cluster(self,
                                    args,
                                    cluster_idxs,
                                    scale_idxs,
                                    sample_size=5,
                                    component='U',
                                    time_preference=None,
                                    get_full_interval=False,
                                    timescale=None):
        """Obtain sliced waveform and time-interval pairs.
        """

        def do_overlap(pair1, pair2):
            start1, end1 = pair1
            start2, end2 = pair2

            # Check for all types of overlap
            return (start1 <= end2) and (start2 <= end1)

        def is_close(pair1, pair2):
            start1, end1 = pair1
            start2, end2 = pair2

            start1 = lmst_xtick(start1)
            end1 = lmst_xtick(end1)
            start2 = lmst_xtick(start2)
            end2 = lmst_xtick(end2)

            # Calculate the time difference between the two intervals in hours
            time_difference1 = abs((start1 - end2).total_seconds() / 3600)
            time_difference2 = abs((start2 - end1).total_seconds() / 3600)

            # Check if the time difference is within the desired range
            if time_difference1 <= 2 or time_difference2 <= 2:
                return True
            else:
                print(min(time_difference1, time_difference2))
                return False

        waveforms = []
        time_intervals = []

        for scale, i in zip(scale_idxs, cluster_idxs):
            scale = str(scale)

            print('Reading waveforms for cluster {}, scale {}'.format(
                i, scale))
            utc_time_intervals = []
            window_idx_list = []

            print('using lmst, maybe utc?')
            for sample_idx in range(
                    len(self.per_cluster_confident_idxs[scale][str(i)])):
                window_idx = self.per_cluster_confident_idxs[scale][str(
                    i)][sample_idx]

                utc_interval = self.get_time_interval(
                    window_idx,
                    scale,
                    lmst=False,
                )[0]
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
                waveform, time_interval = self.get_waveform(
                    window_idx,
                    scale,
                    fine_scale=timescale,
                )

                waveforms.append(waveform[:, COMPONENT_TO_INDEX[component], :])
                time_intervals.append(time_interval)

        return np.array(waveforms), time_intervals
