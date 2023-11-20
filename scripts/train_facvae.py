"""Script to train factorial Gaussian mixture variational auto-encoder.

Typical usage example:

python train_facvae.py --cuda 1
"""

import numpy as np
import os
import torch

from facvae.utils import (configsdir, datadir, parse_input_args, read_config,
                          make_experiment_name, MarsMultiscaleDataset,
                          upload_results)
from scripts.facvae_trainer import FactorialVAETrainer
from scripts.visualization import Visualization

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))

# GMVAE training default hyperparameters.
MARS_CONFIG_FILE = 'facvae_full-mission.json'

if __name__ == "__main__":
    # Read configuration from the JSON file specified by MARS_CONFIG_FILE.
    args = read_config(os.path.join(configsdir(), MARS_CONFIG_FILE))

    # Parse input arguments from the command line
    args = parse_input_args(args)

    # Random seed.
    if hasattr(args, 'seed'):
        seed = args.seed
    else:
        seed = 12
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # A hack to remove the seed from the input arguments so that a specific
    # pretrained model can be used. Not needed later.
    if args.seed == 12:
        del args.seed

    # Set experiment name based on input arguments
    args.experiment = make_experiment_name(args)

    # Process filter_key and scales arguments to remove spaces and split by
    # comma
    if hasattr(args, 'filter_key'):
        args.filter_key = args.filter_key.replace(' ', '').split(',')
    if hasattr(args, 'scales'):
        args.scales = args.scales.replace(' ', '').split(',')

    # Setting default device (cpu/cuda) depending on CUDA availability and
    # input arguments.
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load data from the Mars dataset
    dataset = MarsMultiscaleDataset(os.path.join(MARS_SCAT_COV_PATH,
                                                 args.h5_filename),
                                    0.90,
                                    scatcov_datasets=args.scales,
                                    load_to_memory=args.load_to_memory,
                                    normalize_data=args.normalize,
                                    filter_key=args.filter_key)

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
        if args.extension == '':
            args.experiment = args.experiment + '_' + str(len(
                dataset.test_idx))
        else:
            args.experiment = args.experiment + '_' + args.extension
        # Create an instance of Visualization class.
        vis = Visualization(args, network, dataset, test_loader, device)

        vis.plot_waveforms(args)
        vis.plot_cluster_time_histograms(args)
        vis.centroid_waveforms(args)
        vis.plot_latent_space(args)

    # Upload results to Weights & Biases for tracking training progress.
    upload_results(args, flag='--progress --transfers 20')
