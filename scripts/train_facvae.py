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
from scripts.facvae_trainer import FACVAETrainer
from scripts.visualization import Visualization

# Paths to raw Mars waveforms and the scattering covariance thereof.
MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_covs_h5'))

# GMVAE training default hyperparameters.
MARS_CONFIG_FILE = 'facvae_2022-jun.json'

# Random seed.
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), MULTI_MARS_CONFIG_FILE))
    args = parse_input_args(args)
    args.experiment = make_experiment_name(args)
    if hasattr(args, 'filter_key'):
        args.filter_key = args.filter_key.replace(' ', '').split(',')
    if hasattr(args, 'type'):
        args.type = args.type.replace(' ', '').split(',')

    # Setting default device (cpu/cuda) depending on CUDA availability and
    # input arguments.
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
        # Read in all the data into CPU memory to avoid slowing down GPU.
    else:
        device = torch.device('cpu')
        # Read the data from disk batch by batch.

    # Read Data
    dataset = MarsMultiscaleDataset(os.path.join(MARS_SCAT_COV_PATH,
                                                 args.h5_filename),
                                    0.90,
                                    data_types=args.type,
                                    load_to_memory=args.load_to_memory,
                                    normalize_data=args.normalize,
                                    filter_key=args.filter_key)

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

    facvae_trainer = FACVAETrainer(args, dataset, device)

    if args.phase == 'train':
        # Training Phase.
        facvae_trainer.train(args, train_loader, val_loader)
    elif args.phase == 'test':
        network = facvae_trainer.load_checkpoint(args, args.max_epoch - 1)
        network.gumbel_temp = np.maximum(
            args.init_temp * np.exp(-args.temp_decay * (args.max_epoch - 1)),
            args.min_temp)
        args.experiment = args.experiment + '_' + str(len(dataset.test_idx))

        vis = Visualization(network, dataset, args.window_size, device)
        vis.plot_waveforms(args, test_loader)
        # vis.random_generation(args)
        vis.reconstruct_data(args, train_loader)
        # vis.plot_latent_space(args, test_loader)

    upload_results(args, flag='--progress')
