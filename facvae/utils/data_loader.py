import h5py
import numpy as np
import torch

from facvae.utils.toy_datasets import (GaussianDataset, CrescentDataset,
                                       CrescentCubedDataset, SineWaveDataset,
                                       AbsDataset, SignDataset,
                                       FourCirclesDataset, DiamondDataset,
                                       TwoSpiralsDataset, CheckerboardDataset,
                                       TwoMoonsDataset)


class MarsDataset(torch.utils.data.Dataset):

    def __init__(self,
                 file_path,
                 train_proportion,
                 data_types=['scat_cov'],
                 transform=None,
                 load_to_memory=False):
        self.transform = transform
        self.load_to_memory = load_to_memory

        # HDF5 file.
        self.file = h5py.File(file_path, 'r')
        self.file_keys = list(self.file.keys())
        self.train_idx, self.val_idx, self.test_idx = self.split_data(
            train_proportion)

        self.data = {
            'waveform': None,
            'scat_cov': None,
        }

        if self.load_to_memory:
            self.load_all_data(data_types)

    def split_data(self, train_proportion):
        ntrain = int(train_proportion * len(self.file))
        nval = int((1 - train_proportion) * len(self.file))

        idxs = np.random.permutation(len(self.file))
        train_idx = idxs[:ntrain]
        val_idx = idxs[ntrain:ntrain + nval]
        test_idx = idxs

        return train_idx, val_idx, test_idx

    def load_all_data(self, data_types):
        for type in data_types:
            data = []
            for key in self.file_keys:
                group = self.file[key]
                x = group[type][...]
                x = torch.from_numpy(x)
                if self.transform:
                    with torch.no_grad():
                        x = self.transform(x)
                data.append(x)
            self.data[type] = torch.stack(data)

    def sample_data(self, idx, type='scat_cov'):
        if self.data[type] is None:
            batch_data = []
            for i in idx:
                group = self.file[self.file_keys[i]]
                x = group[type][...]
                x = torch.from_numpy(x)
                if self.transform:
                    with torch.no_grad():
                        x = self.transform(x)
                batch_data.append(x)
            return torch.stack(batch_data)
        else:
            return self.data[type][idx, ...]


class ToyDataset(torch.utils.data.Dataset):

    def __init__(self, num_points, train_proportion, dataset_name):
        self.num_points = num_points
        self.train_proportion = train_proportion

        toy_datasets = {
            'gaussian': GaussianDataset,
            'crescent': CrescentDataset,
            'crescentcubed': CrescentCubedDataset,
            'sinewave': SineWaveDataset,
            'abs': AbsDataset,
            'sign': SignDataset,
            'fourcircles': FourCirclesDataset,
            'diamond': DiamondDataset,
            'twospirals': TwoSpiralsDataset,
            'checkerboard': CheckerboardDataset,
            'twomoons': TwoMoonsDataset,
        }

        if not dataset_name in toy_datasets.keys():
            raise ValueError('No dataset exists with name ', dataset_name)

        # Create dataset.
        self.data = toy_datasets[dataset_name](num_points=num_points).data

        # Data split.
        self.train_idx, self.val_idx, self.test_idx = self.split_data(
            train_proportion)

    def split_data(self, train_proportion):
        ntrain = int(train_proportion * self.num_points)
        nval = int((1 - train_proportion) * self.num_points)

        idxs = np.random.permutation(self.num_points)
        train_idx = idxs[:ntrain]
        val_idx = idxs[ntrain:ntrain + nval]
        test_idx = idxs

        return train_idx, val_idx, test_idx

    def sample_data(self, idx):
        return self.data[idx, ...]
