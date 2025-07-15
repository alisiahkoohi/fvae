"""Script to separate a cluster from given data."""

import os
import shutil
from typing import Tuple

import numpy as np
import torch
from mpire import WorkerPool
from tqdm import tqdm
from sklearn.decomposition import FastICA

import srcsep
from srcsep import generate
from facvae.utils import (
    configsdir,
    parse_input_args,
    read_config,
    make_experiment_name,
    checkpointsdir,
    datadir,
    save_exp_to_h5,
    plot_deglitching,
    process_sequence_arguments,
    create_namespace_from_args,
    query_experiments,
    collect_results,
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


def optimize_ica(
    args,
    x_dataset: np.ndarray,
    x_obs: np.ndarray,
    glitch_idx: int,
    glitch_time: Tuple[str, str],
    gpu_id: int,
) -> None:
    """
    Separates sources using ICA on windowed data.

    Args:
        args (Namespace): Command line arguments.
        x_dataset (np.ndarray): Mars background dataset.
        x_obs (np.ndarray): Observed data with glitch.
        glitch_idx (int): Index of the glitch.
        glitch_time (Tuple[str, str): Glitch time.
        gpu_id (int): GPU identifier (unused for ICA but kept for compatibility).
    """

    # Store original data for later use
    x_obs_orig = x_obs.copy()
    x_dataset_orig = x_dataset.copy()

    # Whiten the dataset if normalization is enabled.
    if args.normalize:
        # Calculate mean and standard deviation along axis 0 and 1 for the
        # dataset.
        x_mean = x_dataset.mean(axis=(0, 1))
        x_std = x_dataset.std(axis=(0, 1))
        # Whiten the dataset and the observed data.
        x_dataset = (x_dataset - x_mean) / (x_std + 1e-8)
        x_obs = (x_obs - x_mean) / (x_std + 1e-8)

    # Get data dimensions
    n_samples, n_channels, n_timepoints = x_dataset.shape
    _, _, obs_timepoints = x_obs.shape

    # Initialize output array
    x_hat = np.zeros_like(x_obs)

    # Define window size (you can adjust this based on your needs)
    window_size = min(1024, obs_timepoints)  # Adjust as needed
    overlap = window_size // 2

    # Process each channel separately
    for ch in range(n_channels):
        # Prepare data for ICA
        # Combine dataset and observed data for this channel
        dataset_ch = x_dataset[:, ch, :].T  # Shape: (timepoints, samples)
        obs_ch = x_obs[0, ch, :].reshape(-1, 1)  # Shape: (timepoints, 1)

        # Process in overlapping windows
        for start_idx in range(0, obs_timepoints - window_size + 1, overlap):
            end_idx = start_idx + window_size

            # Extract window from observed data
            obs_window = obs_ch[start_idx:end_idx, :]  # Shape: (window_size, 1)

            # Find suitable background segments from dataset
            # Take segments from dataset that match the window size
            if dataset_ch.shape[0] >= window_size:
                # Sample random segments from dataset
                n_components = min(
                    args.R, n_samples, window_size
                )  # Ensure we don't exceed dimensions
                dataset_segments = []

                for i in range(n_components):
                    if i < n_samples:
                        # Take segment from sample i
                        seg_start = np.random.randint(
                            0, max(1, dataset_ch.shape[0] - window_size + 1)
                        )
                        seg_end = seg_start + window_size
                        dataset_segments.append(
                            dataset_ch[seg_start:seg_end, i : i + 1]
                        )
                    else:
                        # If we need more components, take random segments
                        rand_sample = np.random.randint(0, n_samples)
                        seg_start = np.random.randint(
                            0, max(1, dataset_ch.shape[0] - window_size + 1)
                        )
                        seg_end = seg_start + window_size
                        dataset_segments.append(
                            dataset_ch[
                                seg_start:seg_end, rand_sample : rand_sample + 1
                            ]
                        )

                # Combine segments
                dataset_window = np.concatenate(
                    dataset_segments, axis=1
                )  # Shape: (window_size, n_components)

                # Combine observed data with dataset segments for ICA
                mixed_signals = np.concatenate(
                    [obs_window, dataset_window], axis=1
                )  # Shape: (window_size, n_components+1)

                # Apply ICA if we have enough components
                if mixed_signals.shape[1] >= 2:
                    try:
                        # Apply FastICA
                        ica = FastICA(
                            n_components=mixed_signals.shape[1],
                            random_state=SEED,
                            max_iter=1000,
                            tol=1e-4,
                        )

                        # Fit ICA and transform
                        sources = ica.fit_transform(
                            mixed_signals
                        )  # Shape: (window_size, n_components)

                        # Reconstruct without the first component (assuming it contains the glitch)
                        # This is a simple approach - you might want to use more sophisticated component selection
                        mixing_matrix = (
                            ica.mixing_
                        )  # Shape: (n_components, n_components)

                        # Zero out the first component (or select based on some criterion)
                        sources_filtered = sources.copy()
                        sources_filtered[:, 0] = 0  # Remove first component

                        # Reconstruct the signal
                        reconstructed = (
                            sources_filtered @ mixing_matrix.T
                        )  # Shape: (window_size, n_components)

                        # Take the first column (corresponding to the observed signal)
                        x_hat_window = reconstructed[:, 0]

                    except Exception as e:
                        print(
                            f"ICA failed for channel {ch}, window {start_idx}-{end_idx}: {e}"
                        )
                        # Fallback: use original signal
                        x_hat_window = obs_window[:, 0]
                else:
                    # Not enough components for ICA
                    x_hat_window = obs_window[:, 0]
            else:
                # Dataset too small
                x_hat_window = obs_window[:, 0]

            # Apply window overlap handling (simple averaging for overlapping regions)
            if start_idx == 0:
                x_hat[0, ch, start_idx:end_idx] = x_hat_window
            else:
                # Average overlapping regions
                overlap_start = start_idx
                overlap_end = min(start_idx + overlap, end_idx)
                non_overlap_start = overlap_end

                if overlap_start < overlap_end:
                    # Weighted average for overlap region
                    weight_old = np.linspace(1, 0, overlap_end - overlap_start)
                    weight_new = np.linspace(0, 1, overlap_end - overlap_start)

                    x_hat[0, ch, overlap_start:overlap_end] = (
                        weight_old * x_hat[0, ch, overlap_start:overlap_end]
                        + weight_new
                        * x_hat_window[: overlap_end - overlap_start]
                    )

                # Non-overlapping region
                if non_overlap_start < end_idx:
                    x_hat[0, ch, non_overlap_start:end_idx] = x_hat_window[
                        non_overlap_start - start_idx :
                    ]

    # Undo the whitening if normalization is enabled.
    if args.normalize:
        # Undo whitening for the dataset, observed data, and the reconstructed
        # data.
        x_dataset = x_dataset_orig
        x_obs = x_obs_orig
        x_hat = x_hat * (x_std + 1e-8) + x_mean

    # Plot the deglitching results.
    plot_deglitching(args, "deglitching_ica_" + str(glitch_idx), x_obs, x_hat)

    # Save the results to an HDF5 file.
    save_exp_to_h5(
        os.path.join(
            checkpointsdir(args.experiment),
            "reconstruction_ica_" + str(glitch_idx) + ".h5",
        ),
        args,
        x_obs=x_obs,
        x_dataset=x_dataset,
        glitch_idx=glitch_idx,
        x_hat=x_hat,
        glitch_time=[
            str(glitch_time[0]),
            str(glitch_time[-1]),
        ],
    )


def optimize(
    args,
    x_dataset: np.ndarray,
    x_obs: np.ndarray,
    glitch_idx: int,
    glitch_time: Tuple[str, str],
    gpu_id: int,
) -> None:
    """
    Cleans a glitch from a dataset of Mars background data.

    Args:
        args (Namespace): Command line arguments.
        x_dataset (np.ndarray): Mars background dataset.
        x_obs (np.ndarray): Observed data with glitch.
        glitch_idx (int): Index of the glitch.
        glitch_time (Tuple[str, str): Glitch time.
        gpu_id (int): GPU identifier.
    """

    # Whiten the dataset if normalization is enabled.
    if args.normalize:
        # Calculate mean and standard deviation along axis 0 and 1 for the
        # dataset.
        x_mean = x_dataset.mean(axis=(0, 1))
        x_std = x_dataset.std(axis=(0, 1))
        # Whiten the dataset and the observed data.
        x_dataset = (x_dataset - x_mean) / (x_std + 1e-8)
        x_obs = (x_obs - x_mean) / (x_std + 1e-8)

    # Setup deglitching parameters.
    deglitching_params = {
        "nks": torch.from_numpy(x_dataset).unsqueeze(-2).unsqueeze(-2),
        "x_init": torch.from_numpy(x_obs).unsqueeze(-2).unsqueeze(-2),
        "indep_loss_w": 1.0,
        "x_loss_w": 1.0,
        "fixed_ts": None,
        "cuda": args.cuda,
    }
    # Generate the deglitched data.
    x_hat = generate(
        x_obs,
        x0=x_obs,
        J=args.j,
        Q=args.q,
        wav_type=args.wavelet,
        it=args.max_itr,
        tol_optim=args.tol_optim,
        deglitching_params=deglitching_params,
        cuda=args.cuda,
        nchunks=args.nchunks,
        gpus=[args.gpu_id[gpu_id]],
        exp_name=f"{args.experiment_name}_R-{args.R}_glitch_idx-{glitch_idx}",
    )

    # Undo the whitening if normalization is enabled.
    if args.normalize:
        # Undo whitening for the dataset, observed data, and the reconstructed
        # data.
        x_dataset = x_dataset * (x_std + 1e-8) + x_mean
        x_obs = x_obs * (x_std + 1e-8) + x_mean
        x_hat = x_hat * (x_std + 1e-8) + x_mean

    # Plot the deglitching results.
    plot_deglitching(args, "deglitching_" + str(glitch_idx), x_obs, x_hat)

    # Save the results to an HDF5 file.
    save_exp_to_h5(
        os.path.join(
            checkpointsdir(args.experiment),
            "reconstruction_" + str(glitch_idx) + ".h5",
        ),
        args,
        x_obs=x_obs,
        x_dataset=x_dataset,
        glitch_idx=glitch_idx,
        x_hat=x_hat,
        glitch_time=[
            str(glitch_time[0]),
            str(glitch_time[-1]),
        ],
    )


def source_separation_serial_job(gpu_id: int, shared_in: Tuple, j: int) -> None:
    """
    Load a job for parallel processing.

    Args:
        gpu_id (int): GPU identifier.
        shared_in (Tuple): Shared input data.
        j (int): Index of the job.
    """
    print(f"Starting job {j} on GPU {gpu_id} with shared input.")
    print(f"Shared input keys: {[type(_) for _ in shared_in]}")
    optimize, args, snippets, glitch, glitch_time = shared_in
    g = glitch[j : j + 1 :, :, :]
    g_time = glitch_time[j]
    snippet = snippets[j].astype(np.float64)
    optimize(args, snippet, g, j, g_time, gpu_id)


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
    from IPython import embed

    embed()

    snippets = {j: [] for j in range(glitch.shape[0])}

    # Extract snippets from the Mars dataset based on specified parameters.
    if args.same_snippets:
        print("Extracting same snippets for all glitches.")
        snippet, _ = snippet_extractor.waveforms_per_scale_cluster(
            vae_args,
            args.cluster_n,
            args.scale_n,
            sample_size=args.R,
            component="U",
            time_preference=(glitch_time[0][0], glitch_time[-1][-1]),
            num_workers=args.num_workers,
        )
        for j in range(glitch.shape[0]):
            snippets[j] = snippet

    else:
        for j in tqdm(range(glitch.shape[0]), desc="Extracting snippets"):
            g_time = glitch_time[j]
            snippets[j], _ = snippet_extractor.waveforms_per_scale_cluster(
                vae_args,
                args.cluster_n,
                args.scale_n,
                sample_size=args.R,
                component="U",
                time_preference=g_time,
                num_workers=args.num_workers,
            )

    # Choose optimization function based on args.ica
    optimize_func = optimize_ica if args.run_ica else optimize

    # Parallel processing using WorkerPool.
    with WorkerPool(
        n_jobs=4,  # Number of GPUs (hardcoded)
        pass_worker_id=True,
        shared_objects=(
            optimize_func,
            args,
            snippets,
            glitch,
            glitch_time,
        ),
        start_method="fork",
    ) as pool:
        outputs = pool.map(
            source_separation_serial_job,
            range(glitch.shape[0]),
            progress_bar=False,
        )

    experiment_results = collect_results(
        experiment_args,
        [
            "x_obs",
            "x_hat",
            "glitch_idx",
            "glitch_time",
        ],
    )
    plot_result(args, experiment_results)
