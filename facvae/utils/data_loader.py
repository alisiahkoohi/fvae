import h5py
import numpy as np
import os
import torch
import pickle
from mpire import WorkerPool
from obspy.core import UTCDateTime

from facvae.utils import (GaussianDataset, CrescentDataset,
                          CrescentCubedDataset, SineWaveDataset, AbsDataset,
                          SignDataset, FourCirclesDataset, DiamondDataset,
                          TwoSpiralsDataset, CheckerboardDataset,
                          TwoMoonsDataset, catalogsdir)


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
                if type == 'scat_cov':
                    x = group[type][...].reshape(3, 2, -1)[0, ...].reshape(-1)
                elif type == 'waveform':
                    x = group[type][0, :]
                else:
                    raise ValueError('No dataset exists with type ', type)
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
                if type == 'scat_cov':
                    x = group[type][...].reshape(3, 2, -1)[0, ...].reshape(-1)
                elif type == 'waveform':
                    x = group[type][0, :]
                else:
                    raise ValueError('No dataset exists with type ', type)
                x = torch.from_numpy(x)
                if self.transform:
                    with torch.no_grad():
                        x = self.transform(x)
                batch_data.append(x)
            return torch.stack(batch_data)
        else:
            return self.data[type][idx, ...]

    def get_labels(self, idx):
        labels = []
        for i in idx:
            group = self.file[self.file_keys[i]]
            x = group['label'][...].astype(str)
            labels.append(x)
        return labels

    def get_waveform_key(self, idx):
        return [self.file_keys[i] for i in idx]

    def get_time_interval(self, idx):
        labels = []
        for i in idx:
            group = self.file[self.file_keys[i]]
            x = group['window_times'][...]
            x = (UTCDateTime(x[0].decode('utf-8')),
                 UTCDateTime(x[1].decode('utf-8')))
            labels.append(x)
        return labels


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

        if dataset_name not in toy_datasets.keys():
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


class CatalogReader(torch.utils.data.Dataset):

    def __init__(self,
                 path_to_catalog=os.path.join(catalogsdir('v11'),
                                              'events_InSIght.pkl'),
                 window_size=2**12,
                 frequency=20.0):

        self.window_size = window_size
        self.frequency = frequency

        with open(path_to_catalog, 'rb') as f:
            self.df = pickle.load(f)

    def get_window_label(self, key, start_time, end_time):
        df = self.df[(self.df['eventTime'] >= start_time)
                     & (self.df['eventTime'] <= end_time)]
        labels = []
        for _, row in df.iterrows():
            labels.append(row['type'])
        if len(labels) > 0:
            print(key, start_time, end_time, labels)
        return key, labels

    def add_labels_to_h5_file(self, path_to_h5_file, n_workers=8):
        file = h5py.File(path_to_h5_file, 'r+')
        inputs = []
        for key in file.keys():
            start_time, end_time = file[key]['window_times'][...]
            inputs.append((key, UTCDateTime(start_time.decode('utf-8')),
                           UTCDateTime(end_time.decode('utf-8'))))
        with WorkerPool(n_jobs=n_workers) as pool:
            keys_and_labels = pool.map(self.get_window_label,
                                       inputs,
                                       progress_bar=True)
        for key, label in keys_and_labels:
            file[key].require_dataset('label',
                                      shape=len(label),
                                      data=label,
                                      dtype=h5py.string_dtype())
        file.close()
