"""Script to separate a cluster from given data."""

import os
import shutil
from typing import Tuple

import numpy as np
import torch
from mpire import WorkerPool
from tqdm import tqdm
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

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
    Separates sources using ICA on 3-channel seismic data and plots all components.
    """
    # Store original data for later use
    x_obs_orig = x_obs.copy()
    x_dataset_orig = x_dataset.copy()

    # Whiten the data if normalization is enabled
    if args.normalize:
        x_mean = x_dataset.mean(axis=(0, 1))
        x_std = x_dataset.std(axis=(0, 1))
        x_obs = (x_obs - x_mean) / (x_std + 1e-8)

    # Cast to float32 for compatibility with FastICA
    x_dataset = x_dataset.astype(np.float32)
    x_obs = x_obs.astype(np.float32)
    if args.normalize:
        x_mean = x_mean.astype(np.float32)
        x_std = x_std.astype(np.float32)

    # Get data dimensions
    if len(x_obs.shape) == 4:
        _, _, n_channels, n_timepoints = x_obs.shape
        mixed_signals = x_obs[0, 0, :, :].T  # Shape: (n_timepoints, n_channels)
    elif len(x_obs.shape) == 3:
        _, n_channels, n_timepoints = x_obs.shape
        mixed_signals = x_obs[0, :, :].T  # Shape: (n_timepoints, n_channels)
    else:
        raise ValueError(f"Unexpected x_obs shape: {x_obs.shape}")

    print(
        f"Applying ICA to {n_channels}-channel data: {n_timepoints} timepoints"
    )

    try:
        # Apply FastICA with 3 components for 3-channel data
        ica = FastICA(
            n_components=3, random_state=SEED, max_iter=1000, tol=1e-4
        )
        sources = ica.fit_transform(mixed_signals)  # Shape: (n_timepoints, 3)
        mixing_matrix = ica.mixing_  # Shape: (3, 3)

        # Calculate and print source characteristics for interpretation
        for i in range(3):
            source_i = sources[:, i]
            kurt = kurtosis(source_i)
            energy = np.sum(source_i**2)
            peak_to_rms = np.max(np.abs(source_i)) / np.sqrt(
                np.mean(source_i**2)
            )
            print(
                f"Source {i}: kurtosis={kurt:.3f}, energy={energy:.3f}, peak/rms={peak_to_rms:.3f}"
            )

        # Plot all 3 separated sources and all 3 reconstructed signals without each source
        for i in range(3):
            # Create signal with only source i
            source_i_only = np.zeros_like(sources)
            source_i_only[:, i] = sources[:, i]
            reconstructed_source_i = (
                source_i_only @ mixing_matrix.T
            )  # Shape: (n_timepoints, 3)

            # Create signal without source i (background)
            sources_without_i = sources.copy()
            sources_without_i[:, i] = 0
            reconstructed_without_i = (
                sources_without_i @ mixing_matrix.T
            )  # Shape: (n_timepoints, 3)

            # Convert to original format for plotting (focusing on U component)
            x_hat_source_i = reconstructed_source_i.T[
                None, None, :, :
            ]  # Shape: (1, 1, 3, n_timepoints)
            x_hat_without_i = reconstructed_without_i.T[
                None, None, :, :
            ]  # Shape: (1, 1, 3, n_timepoints)

            # Undo whitening if normalization was applied
            if args.normalize:
                x_hat_source_i = x_hat_source_i * (x_std + 1e-8) + x_mean
                x_hat_without_i = x_hat_without_i * (x_std + 1e-8) + x_mean

            # Plot separated source i (using U component for visualization)
            plot_deglitching(
                args,
                f"ica_source_{i}_glitch_{glitch_idx}",
                x_obs_orig[:, :, 0, :],  # Original U component
                x_hat_source_i[:, :, 0, :],  # Separated source i (U component)
            )

            # Plot signal without source i (background after removing source i)
            plot_deglitching(
                args,
                f"ica_background_without_source_{i}_glitch_{glitch_idx}",
                x_obs_orig[:, :, 0, :],  # Original U component
                x_hat_without_i[
                    :, :, 0, :
                ],  # Background without source i (U component)
            )

        # For saving purposes, use the original signal (or you could choose a specific reconstruction)
        x_hat = x_obs_orig[
            :, :, 0, :
        ]  # Shape: (1, 1, 1024) - removes channel dim

    except Exception as e:
        print(f"ICA failed: {e}")
        # Make sure fallback also uses correct shape
        x_hat = x_obs_orig[:, :, 0, :]  # Shape: (1, 1, 1024)

    # Prepare data for saving
    if args.normalize:
        x_dataset = x_dataset_orig
        x_obs = x_obs_orig[:, :, 0, :]  # Shape: (1, 1, 1024)

    # Save results
    save_exp_to_h5(
        os.path.join(
            checkpointsdir(args.experiment),
            "reconstruction_ica_" + str(glitch_idx) + ".h5",
        ),
        args,
        x_obs=x_obs,  # Shape: (1, 1, 1024)
        x_dataset=x_dataset,
        glitch_idx=glitch_idx,
        x_hat=x_hat,  # Shape: (1, 1, 1024)
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
    optimize_func, args, snippets, glitch, glitch_time = shared_in
    g = glitch[j : j + 1 :, :, :]
    g_time = glitch_time[j]
    snippet = snippets[j].astype(np.float64)
    optimize_func(args, snippet, g, j, g_time, gpu_id)


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
        component="all" if args.run_ica else "U",
        timescale=args.scale_n[0],
        num_workers=1,
        overwrite_idx=args.overwrite_idx,
    )

    glitch = glitch[0, ...]
    glitch_time = glitch_time[0]
    glitch = glitch[:, None, :].astype(np.float64)

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

    if args.run_ica:
        # Sequential processing
        outputs = []
        for i in range(glitch.shape[0]):
            result = source_separation_serial_job(
                0,
                (optimize_func, args, snippets, glitch, glitch_time),
                i,
            )
            outputs.append(result)

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

    else:
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
