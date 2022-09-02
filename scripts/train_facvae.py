""" Performs VAE-fGMM training on scattering covariance dataset. """
from re import S
import h5py
import logging
import os
from torch.utils.data import Dataset

from facvae.vae.GMVAE import *
from facvae.utils import (configsdir, datadir, parse_input_args, read_config,
                          make_experiment_name)

logging.basicConfig(level=logging.INFO)

MARS_PATH = datadir('mars')
MARS_SCAT_COV_PATH = datadir(os.path.join(MARS_PATH, 'scat_cov'))
MARS_CONFIG_FILE = 'mars.json'
SEED = 19
SCAT_COV_FILENAME = 'scattering_covariances.h5'

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class Mars(Dataset):

    def __init__(self, train_proportion, transform=None, load_to_memory=False):
        self.transform = transform
        self.train_proportion = train_proportion

        # Path to the file.
        file_path = os.path.join(MARS_SCAT_COV_PATH, SCAT_COV_FILENAME)
        # HDF5 file.
        self.file = h5py.File(file_path, 'r')
        self.file_keys = list(self.file.keys())

        self.ntrain = int(self.train_proportion * len(self.file))
        self.nval = int((1 - self.train_proportion) / 2 * len(self.file))
        self.ntest = len(self.file) - self.ntrain - self.nval
        idxs = np.random.permutation(len(self.file))
        self.train_idx = idxs[:self.ntrain]
        self.val_idx = idxs[self.ntrain:self.ntrain + self.nval]
        self.test_idx = idxs[:self.ntest]

        self.load_to_memory = load_to_memory
        if self.load_to_memory:
            self.load_all_data()

    def load_all_data(self):
        data = []
        for key in self.file_keys:
            group = self.file[key]
            x = group['scat_cov'][...]
            x = torch.from_numpy(x)
            if self.transform:
                with torch.no_grad():
                    x = self.transform(x)
            data.append(x)
        self.data = torch.stack(data)

    def sample_data(self, idx):
        if self.load_to_memory:
            return self.data[idx, ...]
        else:
            batch_data = []
            for i in idx:
                group = self.file[self.file_keys[i]]
                x = group['scat_cov'][...]
                x = torch.from_numpy(x)
                if self.transform:
                    with torch.no_grad():
                        x = self.transform(x)
                batch_data.append(x)
            return torch.stack(batch_data)


if __name__ == "__main__":
    # Command line arguments.
    args = read_config(os.path.join(configsdir(), MARS_CONFIG_FILE))
    args = parse_input_args(args)
    args.experiment = make_experiment_name(args)

    # Read Data
    logging.info("Loading mars scattering covariance dataset.")
    mars_dataset = Mars(args.train_proportion,
                        transform=None,
                        load_to_memory=True)

    # Create data loaders for train, validation and test datasets
    train_loader = torch.utils.data.DataLoader(mars_dataset.train_idx,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(mars_dataset.val_idx,
                                             batch_size=args.batch_size_val,
                                             shuffle=True,
                                             drop_last=True)
    test_loader = torch.utils.data.DataLoader(mars_dataset.test_idx,
                                              batch_size=args.batch_size_val,
                                              shuffle=False,
                                              drop_last=False)

    args.input_size = np.prod(mars_dataset.sample_data([0]).size())

    gmvae = GMVAE(args)

    ## Training Phase
    history_loss = gmvae.train(args, train_loader, val_loader, mars_dataset)

    ## Testing Phase
    gmvae.test(test_loader, 1, mars_dataset)
    # print("Testing phase...")
    # print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi))

    # print("Testing phase...")
    # print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi))
