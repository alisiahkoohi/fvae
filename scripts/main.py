""" Performs VAE-fGMM training on scattering covariance dataset. """
import argparse
import logging
import sys
import os
from pathlib import Path
import random
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from facvae.vae.GMVAE import *
from facvae.utils import datadir

logging.basicConfig(level=logging.INFO)
CASCADIA_PATH = datadir('cascadia')


class Cascadia(Dataset):

    def __init__(self, dirname, train, train_size=None, transform=None):
        self.dirname = dirname
        self.transform = transform
        if train:
            self.files = list(dirname.iterdir())[:(train_size or 10000)]
        else:
            self.files = list(dirname.iterdir())[-10000:]  # 13675

        # compute normalization quantities
        # mean, mean2, final_max = 0.0, 0.0, 1e-6
        # for f in self.files:
        #     data = np.load(f)
        #     mean += data
        #     mean2 += data ** 2.0
        #     final_max = np.maximum(np.abs(data), final_max)
        # self.mean = mean / len(self.files)
        # self.var = mean2 / len(self.files) - mean ** 2.0
        # self.final_max = final_max

    def __getitem__(self, i):
        f = self.files[i]
        x = np.load(f)[None, :]
        x = np.float32(x)
        x = torch.from_numpy(x)

        # x /= self.final_max
        # x = np.abs(x)

        if self.transform:
            x = self.transform(x)

        return x, i

    def __len__(self):
        return len(self.files)


def get_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of DGM Clustering')

    ## Used only in notebooks
    parser.add_argument(
        '-f',
        '--file',
        help=
        'Path for input file. First line should contain number of lines to search in'
    )

    ## Dataset
    parser.add_argument('--dataset',
                        type=str,
                        choices=['mnist', 'cascadia'],
                        default='mnist',
                        help='dataset (default: mnist)')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed (default: 0)')

    ## GPU
    parser.add_argument('--cuda',
                        type=int,
                        default=0,
                        help='use of cuda (default: 1)')
    parser.add_argument('--gpu', type=str, help="GPU devices")

    ## Training
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--batch-size',
                        default=64,
                        type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--batch-size_val',
                        default=200,
                        type=int,
                        help='mini-batch size of validation (default: 200)')
    parser.add_argument('--learning-rate',
                        default=1e-3,
                        type=float,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay-epoch',
                        default=-1,
                        type=int,
                        help='Reduces the learning rate every decay_epoch')
    parser.add_argument('--lr-decay',
                        default=0.5,
                        type=float,
                        help='Learning rate decay for training (default: 0.5)')

    ## Architecture
    parser.add_argument('--num-classes',
                        type=int,
                        default=10,
                        help='number of classes (default: 10)')
    parser.add_argument('--gaussian-size',
                        default=64,
                        type=int,
                        help='gaussian size (default: 64)')
    parser.add_argument('--input-size',
                        default=784,
                        type=int,
                        help='input size (default: 784)')

    ## Partition parameters
    parser.add_argument(
        '--train-proportion',
        default=1.0,
        type=float,
        help=
        'proportion of examples to consider for training only (default: 1.0)')

    ## Gumbel parameters
    parser.add_argument(
        '--init-temp',
        default=1.0,
        type=float,
        help=
        'Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)'
    )
    parser.add_argument(
        '--decay-temp',
        default=1,
        type=int,
        help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
    parser.add_argument(
        '--hard-gumbel',
        default=0,
        type=int,
        help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
    parser.add_argument(
        '--min-temp',
        default=0.5,
        type=float,
        help=
        'Minimum temperature of gumbel-softmax after annealing (default: 0.5)')
    parser.add_argument(
        '--decay-temp_rate',
        default=0.013862944,
        type=float,
        help='Temperature decay rate at every epoch (default: 0.013862944)')

    ## Loss function parameters
    parser.add_argument('--w-gauss',
                        default=1,
                        type=float,
                        help='weight of gaussian loss (default: 1)')
    parser.add_argument('--w-categ',
                        default=1,
                        type=float,
                        help='weight of categorical loss (default: 1)')
    parser.add_argument('--w-rec',
                        default=1,
                        type=float,
                        help='weight of reconstruction loss (default: 1)')
    parser.add_argument(
        '--rec-type',
        type=str,
        choices=['bce', 'mse'],
        default='mse',
        help='desired reconstruction loss function (default: bce)')

    ## Others
    parser.add_argument('--verbose',
                        action='store_true',
                        help='print extra information at every epoch.')
    parser.add_argument('--random-search-it',
                        type=int,
                        default=20,
                        help='iterations of random search (default: 20)')

    return parser.parse_args()


if __name__ == "__main__":
    #########################################################
    ## Input Parameters
    #########################################################
    args = get_args()

    ## Random Seed
    SEED = args.seed
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if args.cuda:
        torch.cuda.manual_seed(SEED)

    #########################################################
    ## Read Data
    #########################################################
    if args.dataset == "mnist":
        print("Loading mnist dataset...")
        train_dataset = datasets.MNIST(datadir(''),
                                       train=True,
                                       download=True,
                                       transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(datadir(''),
                                      train=False,
                                      transform=transforms.ToTensor())
    if args.dataset == 'cascadia':
        logging.info("Loading cascadia dataset...")
        SCAT_DATA_SET = Path(datadir(
            os.path.join('cascadia', 'scat_covariances', 'test_window_17_phase')))
        train_dataset = Cascadia(SCAT_DATA_SET,
                                 train=True,
                                 train_size=20000,
                                 transform=None)
        test_dataset = Cascadia(SCAT_DATA_SET,
                                train=False,
                                transform=None)
        # from IPython import embed; embed()

    #########################################################
    ## Data Partition
    #########################################################
    def partition_dataset(n, proportion=0.8):
        train_num = int(n * proportion)
        indices = np.random.permutation(n)
        train_indices, val_indices = indices[:train_num], indices[train_num:]
        return train_indices, val_indices

    if args.train_proportion == 1.0:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size_val, shuffle=False)
        val_loader = test_loader
    else:
        train_indices, val_indices = partition_dataset(len(train_dataset),
                                                       args.train_proportion)
        # Create data loaders for train, validation and test datasets
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_indices))
        val_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size_val,
            sampler=SubsetRandomSampler(val_indices))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size_val, shuffle=False)

    ## Calculate flatten size of each input data
    args.input_size = np.prod(train_dataset.__getitem__(0)[0].size())
    print(args.input_size)
    #########################################################
    ## Train and Test Model
    #########################################################
    gmvae = GMVAE(args)

    print(f"Saving exp in dir: {gmvae.exp_path}")

    gpus = [''
            ] if args.gpu is None else [int(gp) for gp in args.gpu.split(',')]
    if torch.cuda.device_count() > 1 and len(gpus) > 1:
        print("We have available ", torch.cuda.device_count(), "GPUs!")
        gmvae.network = nn.DataParallel(gmvae.network, device_ids=gpus)

    ## Training Phase
    history_loss = gmvae.train(train_loader, val_loader)

    ## Testing Phase
    accuracy, nmi = gmvae.test(test_loader)
    print("Testing phase...")
    print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi))

    print(f"Saved: {gmvae.exp_path.stem}")
