"""Script to train factorial Gaussian mixture variational auto-encoder.

Typical usage example:

python train_facvae.py --cuda 1
"""

import numpy as np
import os
import torch

from facvae.utils import (configsdir, datadir, parse_input_args, read_config,
                          make_experiment_name, SyntheticMultiscaleDataset,
                          upload_results)
from scripts.facvae_trainer import FactorialVAETrainer
from scripts.visualization import Visualization

# Paths to raw Mars waveforms and the scattering covariance thereof.
DATA_PATH = datadir('synthetic_dataset')
SCAT_COV_PATH = datadir(os.path.join(DATA_PATH, 'scat_covs_h5'))

# GMVAE training default hyperparameters.
SYNTHETIC_CONFIG_FILE = 'facvae_synthetic.json'

# Random seed.
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    # Read configuration from the JSON file specified by SYNTHETIC_CONFIG_FILE.
    args = read_config(os.path.join(configsdir(), SYNTHETIC_CONFIG_FILE))

    # Parse input arguments from the command line
    args = parse_input_args(args)

    # Set experiment name based on input arguments
    args.experiment = make_experiment_name(args)

    # Process scales arguments to remove spaces and split by
    # comma
    if hasattr(args, 'scales'):
        args.scales = args.scales.replace(' ', '').split(',')

    # Setting default device (cpu/cuda) depending on CUDA availability and
    # input arguments.
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load data from the Mars dataset
    dataset = SyntheticMultiscaleDataset(os.path.join(SCAT_COV_PATH,
                                                 args.h5_filename),
                                    0.90,
                                    scatcov_datasets=args.scales,
                                    load_to_memory=args.load_to_memory,
                                    normalize_data=args.normalize)

    # Create data loaders for train, validation and test datasets

    if len(dataset.train_idx) < args.batchsize:
        args.batchsize = len(dataset.train_idx)

    train_loader = torch.utils.data.DataLoader(dataset.train_idx,
                                               batch_size=args.batchsize,
                                               shuffle=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset.val_idx,
                                             batch_size=args.batchsize,
                                             shuffle=True,
                                             drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset.test_idx,
                                              batch_size=args.batchsize,
                                              shuffle=False,
                                              drop_last=False)

    # Initialize facvae trainer with the input arguments, dataset, and device
    facvae_trainer = FactorialVAETrainer(args, dataset, device)

    if args.phase == 'train':
        # Training Phase.
        # Train the model using train_loader and val_loader.
        facvae_trainer.train(args, train_loader, val_loader)

    elif args.phase == 'test':
        # Load a saved checkpoint for testing.
        network = facvae_trainer.load_checkpoint(args, args.max_epoch - 1)
        # Set the gumbel temperature for sampling from the categorical
        # distribution.
        network.gumbel_temp = np.maximum(
            args.init_temp * np.exp(-args.temp_decay * (args.max_epoch - 1)),
            args.min_temp)
        # Append the number of test samples to the experiment name.
        args.experiment = args.experiment + '_' + str(len(dataset.test_idx))
        # Create an instance of Visualization class.
        vis = Visualization(args, network, dataset, test_loader, device)
        # Plot waveforms from the test set.
        vis.plot_waveforms(args)
        vis.plot_cluster_time_histograms(args)
        # Reconstruct a sample of the training data.
        # vis.reconstruct_data(args, train_loader)
        # Uncomment the following line to generate random samples.
        # vis.random_generation(args)
        # Uncomment the following line to plot the latent space.
        # vis.plot_latent_space(args, test_loader)

    # Upload results to Weights & Biases for tracking training progress.
    upload_results(args, flag='--progress --transfers 8')
