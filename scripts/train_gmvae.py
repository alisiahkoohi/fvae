"""Script to train Gaussian mixture variational auto-encoder training.

Typical usage example:

python train_gmvae.py --cuda 1
"""

import numpy as np
import os
import torch

from facvae.vae.gaussian_mixture_vae import GaussianMixtureVAE
from facvae.utils import (configsdir, datadir, parse_input_args, read_config,
                          make_experiment_name, MarsDataset, upload_results)

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_cov'))
SCAT_COV_FILENAME = 'scattering_covariances.h5'

# GMVAE training default hyperparameters.
MARS_CONFIG_FILE = 'mars.json'

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
    mars_dataset = MarsDataset(os.path.join(MARS_SCAT_COV_PATH,
                                            SCAT_COV_FILENAME),
                               0.8,
                               transform=None,
                               load_to_memory=load_to_memory)

    # Create data loaders for train, validation and test datasets
    train_loader = torch.utils.data.DataLoader(mars_dataset.train_idx,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               drop_last=False)
    val_loader = torch.utils.data.DataLoader(mars_dataset.val_idx,
                                             batch_size=args.batchsize,
                                             shuffle=True,
                                             drop_last=False)
    test_loader = torch.utils.data.DataLoader(mars_dataset.test_idx,
                                              batch_size=args.batchsize,
                                              shuffle=False,
                                              drop_last=False)

    # Get input dimension.
    args.input_size = int(np.prod(mars_dataset.sample_data([0]).size()))

    network = GaussianMixtureVAE(args, mars_dataset, device)

    if args.phase == 'train':
        # Training Phase
        history_loss = network.train(args, train_loader, val_loader)
        upload_results(args, flag='--progress')
    elif args.phase == 'test':
        network.load(args, args.max_epoch - 1)
        network.random_generation(args)
        network.plot_latent_space(args, test_loader)
        network.reconstruct_data(args, test_loader, sample_size=5)
        # network.test(test_loader, 1)
