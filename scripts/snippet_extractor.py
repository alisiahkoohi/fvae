import obspy
import numpy as np
import torch
from tqdm import tqdm
from mpire import WorkerPool
import argparse
from typing import List, Tuple

from facvae.utils import (
    create_lmst_xticks,
    get_waveform_path_from_time_interval,
    MarsMultiscaleDataset,
    lmst_xtick,
)
from scripts.facvae_trainer import FactorialVAETrainer

# Constants for datastream merge method and fill value
MERGE_METHOD: int = 1
FILL_VALUE: str = 'interpolate'

# Mapping of component to index
COMPONENT_TO_INDEX: dict = {
    'U': 0,
    'V': 1,
    'W': 2,
}


# Class for extracting snippets
class SnippetExtractor(object):
    """
    Class for extracting snippets using a trained Factorial Variational
    Autoencoder (fVAE).

    Attributes:
        device (torch.device): Device to perform computations on.
        dataset (MarsMultiscaleDataset): Mars dataset containing multiscale
            information.
        network: Trained fVAE network.
        scales (List[int]): List of scales used in the analysis.
        in_shape (dict): Dictionary containing input shapes for different
            scales.
        per_cluster_confident_idxs (dict): Dictionary to store per-cluster
            confident indices.
    """

    def __init__(
            self,
            args: argparse.Namespace,
            dataset_path: str,
            device: torch.device = torch.device('cpu'),
    ) -> None:
        """
        Initializes the SnippetExtractor.

        Args:
            args (argparse.Namespace): Input arguments.
            dataset_path (str): Path to the Mars dataset.
            device (torch.device, optional): Device to perform computations on
                (default is 'cpu').
        """

        # Device to perform computations on.
        self.device: torch.device = device

        # Load data from the Mars dataset
        self.dataset: MarsMultiscaleDataset = MarsMultiscaleDataset(
            dataset_path,
            0.90,
            scatcov_datasets=args.scales,
            load_to_memory=args.load_to_memory,
            normalize_data=args.normalize,
            filter_key=args.filter_key,
        )
        data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.dataset.test_idx,
            batch_size=args.batchsize,
            shuffle=False,
            drop_last=False,
        )

        # Initialize facvae trainer with the input arguments, dataset, and
        # device.
        facvae_trainer: FactorialVAETrainer = FactorialVAETrainer(
            args,
            self.dataset,
            self.device,
        )

        # Load a saved checkpoint for testing.
        self.network = facvae_trainer.load_checkpoint(args, args.max_epoch - 1)
        self.network.gumbel_temp: np.ndarray = np.maximum(
            args.init_temp * np.exp(-args.temp_decay * (args.max_epoch - 1)),
            args.min_temp,
        )
        self.network.eval()

        # Scales.
        self.scales: List[int] = args.scales
        self.in_shape: dict = {
            scale: self.dataset.shape['scat_cov'][scale]
            for scale in self.scales
        }

        # Dictionary to store per-cluster confident indices
        self.per_cluster_confident_idxs: dict = self.evaluate_model(
            args, data_loader)

    def evaluate_model(
        self,
        args: argparse.Namespace,
        data_loader: torch.utils.data.DataLoader,
    ) -> dict:
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

        # Placeholder for cluster membership and probability for all the data.
        cluster_membership: dict = {
            scale:
            torch.zeros(len(data_loader.dataset),
                        self.dataset.data['scat_cov'][scale].shape[1],
                        dtype=torch.long)
            for scale in self.scales
        }
        cluster_membership_prob: dict = {
            scale:
            torch.zeros(len(data_loader.dataset),
                        self.dataset.data['scat_cov'][scale].shape[1],
                        dtype=torch.float)
            for scale in self.scales
        }

        # Extract cluster memberships.
        pbar: tqdm = tqdm(total=len(data_loader), desc='Evaluating the model')
        for _, idx in enumerate(data_loader):
            # Load data.
            x: dict = self.dataset.sample_data(idx, 'scat_cov')
            # Move to `device`.
            x: dict = {
                scale: x[scale].to(self.device)
                for scale in self.scales
            }
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
        confident_idxs: dict = {}
        for scale in self.scales:
            # Flatten cluster_membership_prob into a 1D tensor.
            prob_flat: torch.Tensor = cluster_membership_prob[scale].flatten()

            # Sort the values in the flattened tensor in descending order and
            # return the indices.
            confident_idxs[scale]: np.ndarray = torch.argsort(
                prob_flat, descending=True).numpy()

        per_cluster_confident_idxs: dict = {
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

    def get_waveform(
        self,
        window_idx: int,
        scale: int,
        fine_scale: int = None,
    ) -> Tuple[
            np.ndarray,
            List[Tuple[float, float]],
    ]:
        """
        Retrieves the waveform data for a specific window and scale.

        Args:
            window_idx (int): Index of the window.
            scale (int): Scale of the waveform.
            fine_scale (int, optional): Fine scale value, if applicable (default
                is None).

        Returns:
            Tuple[np.ndarray, List[Tuple[float, float]]]: A tuple containing the
                waveform data
            and the corresponding time intervals.
        """
        window_time_intervals: List[
            Tuple[float, float],
        ] = self.get_time_interval(
            window_idx,
            scale,
            lmst=False,
            fine_scale=fine_scale,
        )

        # Find filename of the waveform. Note that we only use the first time
        # interval in the list to find the filename as all the time intervals
        # inherently belong to the same day (even if fine_scale is not None)
        # since the largest window size (i.e., largest scale) does not span
        # multiple days.
        filepath: str = get_waveform_path_from_time_interval(
            *window_time_intervals[0])

        # Extract the data, merge the traces, and detrend to prepare for source
        # separation.
        data_stream: obspy.Stream = obspy.read(filepath)
        data_stream: obspy.Stream = data_stream.merge(
            method=MERGE_METHOD,
            fill_value=FILL_VALUE,
        )
        data_stream: obspy.Stream = data_stream.detrend(
            type='spline',
            order=2,
            dspline=2000,
            plot=False,
        )

        waveforms: List[np.ndarray] = []
        for window_time_interval in window_time_intervals:
            sliced_stream: obspy.Stream = data_stream.slice(
                *window_time_interval)
            waveforms.append(
                np.array([td.data[-int(scale):] for td in sliced_stream]))

        # Return the required subwindow.
        return np.stack(waveforms), window_time_intervals

    def get_time_interval(
        self,
        window_idx: int,
        scale: int,
        lmst: bool = True,
        fine_scale: int = None,
    ) -> List[Tuple[float, float]]:
        """
        Retrieves the time intervals for a specific window and scale.

        Args:
            window_idx (int): Index of the window.
            scale (int): Scale of the time interval.
            lmst (bool, optional): If True, converts time intervals to LMST
                format (default is True).
            fine_scale (int, optional): Fine scale value, if applicable (default
                is None).

        Returns:
            List[Tuple[float, float]]: List of time intervals.
        """
        if fine_scale:
            # Extract window time interval.
            window_time_intervals: List[
                Tuple[float, float],
            ] = self.find_cross_scale_intersecting_windows(
                scale,
                window_idx,
                fine_scale,
            )
        else:
            window_time_intervals: List[
                Tuple[float, float],
            ] = self.dataset.get_time_interval(
                [window_idx],
                scale,
            )

        if lmst:
            # Convert to LMST format, usable by matplotlib.
            window_time_intervals: List[Tuple[float, float]] = [
                create_lmst_xticks(
                    *interval,
                    time_zone='LMST',
                    window_size=int(fine_scale) if fine_scale else int(scale),
                ) for interval in window_time_intervals
            ]

        return window_time_intervals

    def find_cross_scale_intersecting_windows(
        self,
        coarse_scale: int,
        coarse_scale_window_idx: int,
        fine_scale: int,
    ) -> List[Tuple[float, float]]:
        """
        Find intersecting window indices between two scales.

        Args:
            coarse_scale (int): Coarse scale value.
            coarse_scale_window_idx (int): Index of the window at the coarse
                scale.
            fine_scale (int): Fine scale value.

        Returns:
            List[Tuple[float, float]]: List of time intervals.
        """

        if isinstance(fine_scale, int):
            fine_scale = str(fine_scale)

        coarse_scale_time_interval: List[
            Tuple[float, float],
        ] = self.get_time_interval(
            coarse_scale_window_idx,
            coarse_scale,
            lmst=False,
            fine_scale=None,
        )[0]

        min_scale: int = min([int(s) for s in self.scales])

        scale_ratio: int = int(coarse_scale) // min_scale
        scale_stride: int = int(fine_scale) // min_scale

        candidate_idxs: np.ndarray = np.sort(
            np.arange(
                coarse_scale_window_idx,
                coarse_scale_window_idx - scale_ratio,
                -scale_stride,
                dtype=int,
            ))

        fine_scale_time_intervals: List[Tuple[float, float]] = []
        for idx in candidate_idxs:
            try:
                # Obtaining candidate time interval.
                candidate_time_interval: List[
                    Tuple[float, float],
                ] = self.get_time_interval(
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

    def waveforms_per_scale_cluster(
        self,
        args: argparse.Namespace,
        cluster_idxs: List[int],
        scale_idxs: List[int],
        sample_size: int = 5,
        component: str = 'U',
        time_preference: Tuple[float, float] = None,
        timescale: int = None,
        num_workers: int = 40,
        overwrite_idx: int = None,
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Obtain sliced waveform and time-interval pairs for specified clusters
        and scales.

        Args:
            args (argparse.Namespace): Input arguments.
            cluster_idxs (List[int]): List of cluster indices.
            scale_idxs (List[int]): List of scale indices.
            sample_size (int, optional): Number of samples to retrieve per
                cluster (default is 5).
            component (str, optional): Component of the waveform to extract
                (default is 'U').
            time_preference (Tuple[float, float], optional): Desired time
                interval preference (default is None).
            timescale (int, optional): Fine scale value for time interval
                (default is None).
            num_workers (int, optional): Number of worker processes for parallel
                processing (default is 40).
            overwrite_idx (int, optional): Overwrite index for the window
                (default is None).

        Returns:
            Tuple[np.ndarray, List[Tuple[float, float]]]: A tuple containing
                waveform data and corresponding time intervals.
        """

        def do_overlap(
            pair1: Tuple[float, float],
            pair2: Tuple[float, float],
        ) -> bool:
            start1, end1 = pair1
            start2, end2 = pair2

            # Check for all types of overlap
            return (start1 <= end2) and (start2 <= end1)

        def is_close(
            pair1: Tuple[float, float],
            pair2: Tuple[float, float],
        ) -> bool:
            start1, end1 = pair1
            start2, end2 = pair2

            start1 = lmst_xtick(start1)
            end1 = lmst_xtick(end1)
            start2 = lmst_xtick(start2)
            end2 = lmst_xtick(end2)

            # Calculate the time difference between the two intervals in hours
            time_difference1: float = min([
                abs((start1 - end2).total_seconds() / 3600),
                abs((start1.replace(day=2) - end2).total_seconds() / 3600),
                abs((start1 - end2.replace(day=2)).total_seconds() / 3600)
            ])
            time_difference2: float = min([
                abs((start2 - end1).total_seconds() / 3600),
                abs((start2.replace(day=2) - end1).total_seconds() / 3600),
                abs((start2 - end1.replace(day=2)).total_seconds() / 3600)
            ])

            # Check if the time difference is within the desired range
            if time_difference1 <= 2 or time_difference2 <= 2:
                return True
            else:
                # print(min(time_difference1, time_difference2))
                return False

        waveforms: List[np.ndarray] = []
        time_intervals: List[Tuple[float, float]] = []

        for scale, i in zip(scale_idxs, cluster_idxs):
            scale = str(scale)

            utc_time_intervals: List[Tuple[float, float]] = []
            window_idx_list: List[int] = []

            for sample_idx in range(
                    len(self.per_cluster_confident_idxs[scale][str(i)])):

                if sample_size == 1 and overwrite_idx is not None:
                    # Allowing the user to overwrite the window index for the
                    # target window to be source separated.
                    window_idx = self.per_cluster_confident_idxs[scale][str(
                        i)][overwrite_idx]
                else:
                    # Extracting the window index from the per-cluster confident
                    # indices.
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

            def load_serial_job(
                shared_in: Tuple,
                window_idx: int,
            ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
                (get_waveform, scale, timescale) = shared_in
                waveform, time_interval = get_waveform(
                    window_idx,
                    scale,
                    fine_scale=timescale,
                )
                return waveform, time_interval

            with WorkerPool(
                    n_jobs=num_workers,
                    shared_objects=(
                        self.get_waveform,
                        scale,
                        timescale,
                    ),
                    start_method='fork',
            ) as pool:
                outputs: List[
                    Tuple[np.ndarray, List[Tuple[float, float]]],
                ] = pool.map(
                    load_serial_job,
                    window_idx_list,
                    progress_bar=False,
                )
            (waveforms_, time_intervals_) = zip(*outputs)
            for w_, t_ in zip(waveforms_, time_intervals_):
                waveforms.append(w_[:, COMPONENT_TO_INDEX[component], :])
                time_intervals.append(t_)

        return np.array(waveforms), time_intervals
