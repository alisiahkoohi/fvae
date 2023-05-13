"""Script to separate a cluster from given data.
"""

import os
import numpy as np
import torch

from srcsep import generate, analyze, format_tensor
from facvae.utils import (configsdir, parse_input_args,
                          read_config, make_experiment_name, upload_results,
                          checkpointsdir, datadir, MarsMultiscaleDataset)
from scripts.facvae_trainer import FactorialVAETrainer

torch.multiprocessing.set_start_method('spawn', force=True)

# from facvae.utils import plot_deglitching, save_exp_to_h5, SourceSeparationSetup


# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_RAW_PATH = datadir(os.path.join(MARS_PATH, 'raw'))
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))

# Configuration file.
SRC_SEP_CONFIG_FILE = 'source_separation.json'

# Random seed.
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def synthesizer(phi_n, scale, n_per_phi=1):

    # phi_n = phi_n.reshape(-1 , 1, scale)

    n = []

    for j in range(phi_n.shape[0]):
        n.append(generate(phi_n[j:j+1, ...], J=8, S=n_per_phi, it=1000, cuda=True, tol_optim=1e-3))

    n = np.stack(n, axis=0)

    return n

# def optimize(args):
#     """Clean a glitch from a dataset of Mars background data.
#     """
#     # Setup the glitch separation by providing the path glitch json file.
#     mars_srcsep = SourceSeparationSetup(MARS_RAW_PATH, args.glitch)

#     # Extract the data from the files and apply windowing.
#     x_dataset = mars_srcsep.get_windowed_background_data(
#         args.window_size, args.window_size // 2)

#     # If `args.R` is greater than 0, return randomly picked `args.R` windows
#     # from the dataset.
#     if args.R > 0:
#         x_dataset = x_dataset[
#             np.random.permutation(x_dataset.shape[0])[:args.R], :, :]

#     # Extract a glitch from the raw data.
#     glitch = mars_srcsep.get_windowed_glitch_data(args.window_size)

#     for quake_idx in range(glitch.shape[0]):
#         x_obs = glitch[quake_idx:quake_idx + 1, :, :]

#         if args.normalize:
#             x_mean = x_dataset.mean(axis=(0, 1))
#             x_std = x_dataset.std(axis=(0, 1))
#             # Whiten the dataset.
#             x_dataset = (x_dataset - x_mean) / (x_std + 1e-8)
#             x_obs = (x_obs - x_mean) / (x_std + 1e-8)

#         # Realistic scenario: access to a representative (unsupervised) dataset
#         # of signals (with the independence regularization).
#         deglitching_params = {
#             'nks': format_tensor(x_dataset),
#             'x_init': format_tensor(x_obs),
#             'indep_loss_w': args.indep_loss_w,
#             'x_loss_w': args.x_loss_w,
#             'fixed_ts': None,
#             'cuda': args.cuda
#         }
#         x_hat = generate(x_obs,
#                          x0=x_obs,
#                          J=args.j,
#                          Q=args.q,
#                          wav_type=args.wavelet,
#                          it=args.max_itr,
#                          tol_optim=args.tol_optim,
#                          deglitching_params=deglitching_params,
#                          cuda=args.cuda,
#                          nchunks=args.nchunks,
#                          gpus=[args.gpu_id],
#                          exp_name=f'{args.experiment_name}_'
#                          f'R-{args.R}_'
#                          f'indep_loss_w-{args.indep_loss_w}_'
#                          f'x_loss_w-{args.x_loss_w}_'
#                          f'normalize-{args.normalize}_'
#                          f'glitch-{args.glitch}_'
#                          f'quake_idx-{quake_idx}')

#         if args.normalize:
#             # Undo the whitening.
#             x_dataset = x_dataset * (x_std + 1e-8) + x_mean
#             x_obs = x_obs * (x_std + 1e-8) + x_mean
#             x_hat = x_hat * (x_std + 1e-8) + x_mean

#         plot_deglitching(args, 'deglitching_' + str(quake_idx), x_obs, x_hat)

#         # Save the results.
#         save_exp_to_h5(os.path.join(checkpointsdir(
#             args.experiment), 'reconstruction_' + str(quake_idx) + '.h5'),
#                        args,
#                        x_obs=x_obs,
#                        x_dataset=x_dataset,
#                        quake_idx=quake_idx,
#                        x_hat=x_hat)

#         glitch[quake_idx, :, :] = x_hat[0, :, :]
#         upload_results(cmd_args, flag='--progress')


if __name__ == '__main__':
    # Command line arguments.
    from IPython import embed; embed()
    cmd_args = read_config(os.path.join(configsdir(), SRC_SEP_CONFIG_FILE))
    cmd_args = parse_input_args(cmd_args)
    cmd_args.q = [int(j) for j in cmd_args.q.replace(' ', '').split(',')]
    cmd_args.j = [int(j) for j in cmd_args.j.replace(' ', '').split(',')]
    cmd_args.experiment = make_experiment_name(cmd_args)

    # Read pretrained fVAE configuration from the JSON file specified by
    # cmd_args.
    vae_args = read_config(os.path.join(configsdir('fvae_models'), cmd_args.facvae_model))
    vae_args = parse_input_args(vae_args)
    vae_args.experiment = make_experiment_name(vae_args)
    if hasattr(vae_args, 'filter_key'):
        vae_args.filter_key = vae_args.filter_key.replace(' ', '').split(',')
    if hasattr(vae_args, 'scales'):
        vae_args.scales = vae_args.scales.replace(' ', '').split(',')

    # Setting default device (cpu/cuda) depending on CUDA availability and
    # input arguments.
    if torch.cuda.is_available() and cmd_args.cuda > -1:
        device = torch.device('cuda:' + str(cmd_args.cuda))
    else:
        device = torch.device('cpu')

    # Load data from the Mars dataset
    dataset = MarsMultiscaleDataset(os.path.join(MARS_SCAT_COV_PATH,
                                                 vae_args.h5_filename),
                                    0.90,
                                    scatcov_datasets=vae_args.scales,
                                    load_to_memory=vae_args.load_to_memory,
                                    normalize_data=vae_args.normalize,
                                    filter_key=vae_args.filter_key)

    # Create data loaders for train, validation and test datasets

    if len(dataset.train_idx) < vae_args.batchsize:
        vae_args.batchsize = len(dataset.train_idx)

    train_loader = torch.utils.data.DataLoader(dataset.train_idx,
                                               batch_size=vae_args.batchsize,
                                               shuffle=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset.val_idx,
                                             batch_size=vae_args.batchsize,
                                             shuffle=True,
                                             drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset.test_idx,
                                              batch_size=vae_args.batchsize,
                                              shuffle=False,
                                              drop_last=False)

    # Initialize facvae trainer with the input arguments, dataset, and device
    facvae_trainer = FactorialVAETrainer(vae_args, dataset, device)

    # Load a saved checkpoint for testing.
    network = facvae_trainer.load_checkpoint(vae_args, vae_args.max_epoch - 1)
    # Set the gumbel temperature for sampling from the categorical
    # distribution.
    network.gumbel_temp = np.maximum(
        vae_args.init_temp * np.exp(-vae_args.temp_decay * (vae_args.max_epoch - 1)),
        vae_args.min_temp)
    network.eval()
    # Append the number of test samples to the experiment name.
    # args.experiment = args.experiment + '_' + str(len(dataset.test_idx))

    snippets = random_generation(args, vae_args, network, 0, 0, num_samples=3)

    # optimize(cmd_args)

    # Upload results to Weights & Biases for tracking training progress.
    upload_results(args, flag='--progress --transfers 8')


def random_generation(cmd_args, vae_args, network, scale_idx, cluster_idx, dataset, device, num_samples=50):
    """Random generation for one class.

    Args:
        num_samples: (int) number of elements to generate

    Returns:
        generated data according to num_samples
    """
    # categories for each element
    indices = np.ones(num_samples) * cluster_idx
    indices = indices.astype(int).tolist()

    # one hot encoding for cluster_idx at scale_idx. Other scales are simply
    # placeholders and do not influence the generation of scale_idx.
    categorical = {}
    for scale in vae_args.scales:
        categorical[scale] = torch.nn.functional.one_hot(
            torch.tensor(indices), vae_args.ncluster).float().to(device)

    # infer the gaussian distribution according to the category
    mean, var = network.generative.pzy(categorical)

    # gaussian random sample by using the mean and variance
    gaussian = {}
    for scale in vae_args.scales:
        noise = torch.randn_like(var[scale])
        std = torch.sqrt(var[scale])
        gaussian[scale] = mean[scale] + noise * std
        gaussian[scale] = gaussian[scale].to(device)

    # generate new samples with the given gaussian
    with torch.no_grad():
        samples = network.generative.pxz(gaussian)

    samples = dataset.unnormalize(samples[str(scale_idx)].cpu(), 'scat_cov',dset_name=str(scale_idx))
    samples = samples.numpy()

    return samples