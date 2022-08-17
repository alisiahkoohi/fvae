""" Performs VAE-fGMM training on scattering covariance dataset. """
import argparse
import logging
import os
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

from facvae.vae.GMVAE import *
from facvae.utils import configsdir, datadir, parse_input_args, read_config

logging.basicConfig(level=logging.INFO)

MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_cov'))
MARS_CONFIG_FILE = 'mars.json'
SEED = 19

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class Mars(Dataset):

    def __init__(self, dirname, train, train_size=None, transform=None):
        self.dirname = dirname
        self.transform = transform
        if train:
            self.files = list(Path(dirname).iterdir())[:(train_size or 10000)]
        else:
            self.files = list(Path(dirname).iterdir())[-10000:]

    def __getitem__(self, i):
        f = self.files[i]
        x = np.load(f)[None, :]
        x = np.float32(x)
        x = torch.from_numpy(x)
        if self.transform:
            x = self.transform(x)
        return x, i

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), MARS_CONFIG_FILE))
    args = parse_input_args(args)
    from IPython import embed
    embed()

    # Read Data
    logging.info("Loading mars scattering covariance dataset.")
    train_dataset = Mars(MARS_SCAT_COV_PATH,
                         train=True,
                         train_size=20000,
                         transform=None)
    test_dataset = Mars(MARS_SCAT_COV_PATH,
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

    # gpus = [''
    #         ] if args.gpu is None else [int(gp) for gp in args.gpu.split(',')]
    # if torch.cuda.device_count() > 1 and len(gpus) > 1:
    #     print("We have available ", torch.cuda.device_count(), "GPUs!")
    #     gmvae.network = nn.DataParallel(gmvae.network, device_ids=gpus)

    ## Training Phase
    history_loss = gmvae.train(train_loader, val_loader)

    ## Testing Phase
    accuracy, nmi = gmvae.test(test_loader)
    print("Testing phase...")
    print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi))

    print(f"Saved: {gmvae.exp_path.stem}")
