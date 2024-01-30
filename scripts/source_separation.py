"""Script to separate a cluster from given data.
"""

import os
import numpy as np
import torch

from srcsep import generate, format_tensor
from facvae.utils import (
    configsdir,
    parse_input_args,
    read_config,
    make_experiment_name,
    upload_results,
    checkpointsdir,
    datadir,
    MarsMultiscaleDataset,
    save_exp_to_h5,
    plot_deglitching,
    process_sequence_arguments,
    create_namespace_from_args,
)
from scripts.snippet_extractor import SnippetExtractor

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_RAW_PATH = datadir(os.path.join(MARS_PATH, 'raw'))
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))

# Configuration file.
# SRC_SEP_CONFIG_FILE = 'source_separation_sunrise_deglitch.json'
SRC_SEP_CONFIG_FILE = 'source_separation_large_scale.json'
# SRC_SEP_CONFIG_FILE = 'source_separation_small_scale.json'

# Random seed.
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def optimize(args, x_dataset, x_obs, glitch_idx):
    """Clean a glitch from a dataset of Mars background data.
    """
    # Setup the glitch separation by providing the path glitch json file.
    # from IPython import embed; embed()
    # mars_srcsep = GlitchSeparationSetup(MARS_RAW_PATH, args.glitch)

    # Extract a glitch from the raw data.
    # from IPython import embed; embed()
    # glitch = mars_srcsep.get_windowed_glitch_data(args.window_size)

    if args.normalize:
        x_mean = x_dataset.mean(axis=(0, 1))
        x_std = x_dataset.std(axis=(0, 1))
        # Whiten the dataset.
        x_dataset = (x_dataset - x_mean) / (x_std + 1e-8)
        x_obs = (x_obs - x_mean) / (x_std + 1e-8)

    # Realistic scenario: access to a representative (unsupervised) dataset
    # of signals (with the independence regularization).
    deglitching_params = {
        'nks': format_tensor(x_dataset),
        'x_init': format_tensor(x_obs),
        'indep_loss_w': args.indep_loss_w,
        'x_loss_w': args.x_loss_w,
        'fixed_ts': None,
        'cuda': args.cuda
    }
    x_hat = generate(x_obs,
                     x0=x_obs,
                     J=args.j,
                     Q=args.q,
                     wav_type=args.wavelet,
                     it=args.max_itr,
                     tol_optim=args.tol_optim,
                     deglitching_params=deglitching_params,
                     cuda=args.cuda,
                     nchunks=args.nchunks,
                     gpus=[args.gpu_id],
                     exp_name=f'{args.experiment_name}_'
                     f'R-{args.R}_'
                     f'glitch-{args.glitch}_'
                     f'glitch_idx-{glitch_idx}')

    if args.normalize:
        # Undo the whitening.
        x_dataset = x_dataset * (x_std + 1e-8) + x_mean
        x_obs = x_obs * (x_std + 1e-8) + x_mean
        x_hat = x_hat * (x_std + 1e-8) + x_mean

    plot_deglitching(args, 'deglitching_' + str(glitch_idx), x_obs, x_hat)

    # Save the results.
    save_exp_to_h5(os.path.join(checkpointsdir(args.experiment),
                                'reconstruction_' + str(glitch_idx) + '.h5'),
                   args,
                   x_obs=x_obs,
                   x_dataset=x_dataset,
                   glitch_idx=glitch_idx,
                   x_hat=x_hat)


if __name__ == '__main__':

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

    glitch, glitch_time = snippet_extractor.waveforms_per_scale_cluster(
        vae_args,
        args.cluster_g,
        args.scale_g,
        sample_size=1,
        get_full_interval=True,
        timescale=args.scale_n[0])

    glitch = glitch[0, ...]
    glitch_time = glitch_time[0]

    glitch = glitch[:, None, :].astype(np.float64)
    for j in range(glitch.shape[0]):
        g = glitch[j:j + 1:, :, :]
        g_time = glitch_time[j]

        snippets, snippets_time = snippet_extractor.waveforms_per_scale_cluster(
            vae_args,
            args.cluster_n,
            args.scale_n,
            sample_size=args.R,
            time_preference=g_time)

        from IPython import embed
        embed()

        # snippets = snippets[:, 0:1, :].astype(np.float64)
        # optimize(args, snippets, g, j)

    # Upload results to Weights & Biases for tracking training progress.
    upload_results(args, flag='--progress --transfers 8')
