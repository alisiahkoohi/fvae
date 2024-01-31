"""Script to separate a cluster from given data.
"""

import os
import shutil
import numpy as np
import torch
from mpire import WorkerPool
from tqdm import tqdm

import srcsep
from srcsep import generate
from facvae.utils import (
    configsdir,
    parse_input_args,
    read_config,
    make_experiment_name,
    upload_results,
    checkpointsdir,
    datadir,
    save_exp_to_h5,
    plot_deglitching,
    process_sequence_arguments,
    create_namespace_from_args,
)
from scripts.snippet_extractor import SnippetExtractor

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))

# Configuration file.
SRC_SEP_CONFIG_FILE = 'source_separation_large_scale.json'

# Random seed.
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def optimize(args, x_dataset, x_obs, glitch_idx, gpu_id):
    """Clean a glitch from a dataset of Mars background data.
    """

    # Whiten the dataset.
    if args.normalize:
        x_mean = x_dataset.mean(axis=(0, 1))
        x_std = x_dataset.std(axis=(0, 1))
        x_dataset = (x_dataset - x_mean) / (x_std + 1e-8)
        x_obs = (x_obs - x_mean) / (x_std + 1e-8)

    # Setup deglitching parameters.
    deglitching_params = {
        'nks': torch.from_numpy(x_dataset).unsqueeze(-2).unsqueeze(-2),
        'x_init': torch.from_numpy(x_obs).unsqueeze(-2).unsqueeze(-2),
        'indep_loss_w': args.indep_loss_w,
        'x_loss_w': args.x_loss_w,
        'fixed_ts': None,
        'cuda': args.cuda
    }
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
        gpus=[gpu_id],
        exp_name=f'{args.experiment_name}_'
        f'R-{args.R}_'
        f'glitch-{args.glitch}_'
        f'glitch_idx-{glitch_idx}',
    )

    # Undo the whitening.
    if args.normalize:
        x_dataset = x_dataset * (x_std + 1e-8) + x_mean
        x_obs = x_obs * (x_std + 1e-8) + x_mean
        x_hat = x_hat * (x_std + 1e-8) + x_mean

    plot_deglitching(args, 'deglitching_' + str(glitch_idx), x_obs, x_hat)

    # Save the results.
    save_exp_to_h5(
        os.path.join(checkpointsdir(args.experiment),
                     'reconstruction_' + str(glitch_idx) + '.h5'),
        args,
        x_obs=x_obs,
        x_dataset=x_dataset,
        glitch_idx=glitch_idx,
        x_hat=x_hat,
    )


def load_serial_job(gpu_id, shared_in, j):
    # TODO: add control over which GPU to use.
    (optimize, args, snippets, glitch) = shared_in
    g = glitch[j:j + 1:, :, :]
    snippet = snippets[j].astype(np.float64)
    optimize(args, snippet, g, j, gpu_id)


if __name__ == '__main__':

    if os.path.exists(os.path.join(srcsep.__path__[0], '_cached_dir')):
        # Remove cached directory that might have been created by a previous run.
        shutil.rmtree(os.path.join(srcsep.__path__[0], '_cached_dir'))

    # Command line arguments for source separation.
    args = read_config(os.path.join(configsdir(), SRC_SEP_CONFIG_FILE))
    args = parse_input_args(args)
    args.experiment = make_experiment_name(args)
    args = process_sequence_arguments(args)

    # Read pretrained fVAE config JSON file specified by args.
    vae_args = read_config(
        os.path.join(
            configsdir('fvae_models'),
            args.facvae_model,
        ))
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
        get_full_interval=True,
        timescale=args.scale_n[0],
        num_workers=1,
    )

    glitch = glitch[0, ...]
    glitch_time = glitch_time[0]
    glitch = glitch[:, None, :].astype(np.float64)

    snippets = {j: [] for j in range(glitch.shape[0])}

    for j in tqdm(range(glitch.shape[0]), desc='Extracting snippets'):
        g_time = glitch_time[j]
        snippets[j], _ = snippet_extractor.waveforms_per_scale_cluster(
            vae_args,
            args.cluster_n,
            args.scale_n,
            sample_size=args.R,
            time_preference=g_time,
            num_workers=args.num_workers,
        )

    with WorkerPool(
            n_jobs=4,  # TODO: number of GPUs are hardcoded here.
            pass_worker_id=True,
            shared_objects=(
                optimize,
                args,
                snippets,
                glitch,
            ),
            start_method='fork',
    ) as pool:
        outputs = pool.map(
            load_serial_job,
            range(glitch.shape[0]),
            progress_bar=False,
        )

    # Upload results to Weights & Biases for tracking training progress.
    upload_results(args, flag='--progress --transfers 8')
