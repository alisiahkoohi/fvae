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
SCAT_COV_FILENAME = 'scat_covs_q1-2_q2-4_nighttime.h5'

# GMVAE training default hyperparameters.
MARS_CONFIG_FILE = 'mars.json'
TOY_CONFIG_FILE = 'toy.json'

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

    # Setting default device (cpu/cuda) depending on CUDA availability and input
    # arguments.
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
        # Read in all the data into CPU memory to avoid slowing down GPU.
        load_to_memory = True
    else:
        device = torch.device('cpu')
        # Read the data from disk batch by batch.
        load_to_memory = False

    # Read Data
    if args.dataset == 'mars':
        dataset = MarsDataset(os.path.join(MARS_SCAT_COV_PATH,
                                           SCAT_COV_FILENAME),
                              0.8,
                              transform=None,
                              load_to_memory=load_to_memory)
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
    args.input_size = int(np.prod(dataset.sample_data([0]).size()))

    gmvaw = GaussianMixtureVAE(args, dataset, device)

    if args.phase == 'train':
        # Training Phase.
        gmvaw.train(args, train_loader, val_loader)
    elif args.phase == 'test':
        network = gmvaw.load_checkpoint(args, args.max_epoch - 1)
        vis = Visualization(network, dataset, device)
        if args.dataset == 'mars':
            vis.plot_waveforms(args, test_loader)
            # vis.random_generation(args)
            # vis.reconstruct_data(args, val_loader)
        # else:
            # vis.random_generation(args, num_elements=5000)
            # vis.reconstruct_data(args, val_loader, sample_size=5000)
        # vis.plot_latent_space(args, val_loader)
    upload_results(args, flag='--progress')
