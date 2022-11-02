"""Script to train Gaussian mixture variational auto-encoder training.

Typical usage example:

python train_gmvae.py --cuda 1
"""

import numpy as np
import os
import torch

from facvae.utils import (configsdir, datadir, parse_input_args, read_config,
                          make_experiment_name, MarsDataset, upload_results,
                          ToyDataset)
from scripts.gaussian_mixture_vae import GaussianMixtureVAE
from scripts.visualization import Visualization

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))

# GMVAE training default hyperparameters.
MARS_CONFIG_FILE = 'mars.json'
TOY_CONFIG_FILE = 'toy_checkerboard.json'

# Random seed.
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), MARS_CONFIG_FILE))
    args = parse_input_args(args)
    args.experiment = make_experiment_name(args)
    if hasattr(args, 'filter_key'):
        args.filter_key = args.filter_key.replace(' ', '').split(',')

    # Setting default device (cpu/cuda) depending on CUDA availability and
    # input arguments.
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
        # Read in all the data into CPU memory to avoid slowing down GPU.
    else:
        device = torch.device('cpu')
        # Read the data from disk batch by batch.

    # Read Data
    if args.dataset == 'mars':
        dataset = MarsDataset(os.path.join(MARS_SCAT_COV_PATH,
                                           args.h5_filename),
                              0.99,
                              data_types=[args.type],
                              load_to_memory=args.load_to_memory,
                              normalize_data=args.normalize,
                              filter_key=args.filter_key)
    else:
        dataset = ToyDataset(30000, 0.8, dataset_name=args.dataset)

    # Create data loaders for train, validation and test datasets
    train_loader = torch.utils.data.DataLoader(dataset.train_idx,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset.val_idx,
                                             batch_size=args.batchsize,
                                             shuffle=True,
                                             drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset.test_idx,
                                              batch_size=args.batchsize,
                                              shuffle=False,
                                              drop_last=False)

    # Get input dimension.
    gm_vae = GaussianMixtureVAE(args, dataset, device)

    if args.phase == 'train':
        # Training Phase.
        gm_vae.train(args, train_loader, val_loader)
    elif args.phase == 'test':
        network = gm_vae.load_checkpoint(args, args.max_epoch - 1)
        network.gumbel_temp = np.maximum(
            args.init_temp * np.exp(-args.temp_decay * (args.max_epoch - 1)),
            args.min_temp)
        if len(args.filter_key) > 1:
            args.experiment = args.experiment + '_multiday'

        if args.dataset == 'mars':
            vis = Visualization(network, dataset, args.window_size, device)
            vis.plot_waveforms(args, test_loader)
            vis.random_generation(args)
            vis.reconstruct_data(args, train_loader)
            # vis.plot_latent_space(args, test_loader)
        else:
            vis = Visualization(network, dataset, None, device)
            vis.plot_clusters(args, test_loader)
            vis.random_generation(args, num_elements=5000)
            vis.reconstruct_data(args, val_loader, sample_size=5000)
            vis.plot_latent_space(args, val_loader)
    upload_results(args, flag='--progress')
