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
        self.num_windows = self.file['scat_cov'].shape[0]
        self.train_idx, self.val_idx, self.test_idx = self.split_data(
            self.num_windows, train_proportion)

        self.data = {
            'waveform': None,
            'scat_cov': None,
        }

        if self.load_to_memory:
            self.load_all_data(data_types)

    def split_data(self, num_windows, train_proportion):
        ntrain = int(train_proportion * num_windows)
        nval = int((1 - train_proportion) * num_windows)

        idxs = np.random.permutation(num_windows)
        train_idx = idxs[:ntrain]
        val_idx = idxs[ntrain:ntrain + nval]
        test_idx = idxs

        return train_idx, val_idx, test_idx

    def load_all_data(self, data_types):
        for type in data_types:
            if type == 'scat_cov':
                x = self.file['scat_cov'][:, 1, :, :].reshape(
                    self.num_windows, -1)
            elif type == 'waveform':
                x = self.file['waveform'][:, 1, :]
            else:
                raise ValueError('No dataset exists with type ', type)
            x = torch.from_numpy(x)
            if self.transform:
                with torch.no_grad():
                    x = self.transform(x)
            self.data[type] = x

    def sample_data(self, idx, type='scat_cov'):
        if self.data[type] is None:
            if type == 'scat_cov':
                x = self.file['scat_cov'][np.sort(idx),
                                          1, :, :].reshape(len(idx), -1)
            elif type == 'waveform':
                x = self.file['waveform'][np.sort(idx), 1, :]
            else:
                raise ValueError('No dataset exists with type ', type)
            x = torch.from_numpy(x)
            if self.transform:
                with torch.no_grad():
                    x = self.transform(x)
            return x
        else:
            return self.data[type][np.sort(idx), ...]

    def get_labels(self, idx):
        labels_list = self.file['labels'][np.sort(idx), ...].astype(str)
        labels = []
        for i in range(len(idx)):
            label = []
            for v in labels_list[i]:
                if v != '':
                    label.append(v)
            labels.append(label)
        return labels

    def get_waveform_filename(self, idx):
        return [
            f.decode('utf-8') for f in self.file['filename'][np.sort(idx), ...]
        ]

    def get_time_interval(self, idx):
        return [(UTCDateTime(s.decode('utf-8')),
                 UTCDateTime(e.decode('utf-8')))
                for s, e in self.file['time_interval'][np.sort(idx), ...]]


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

    def get_window_label(self, i, start_time, end_time):
        df = self.df[(self.df['eventTime'] >= start_time)
                     & (self.df['eventTime'] <= end_time)]
        labels = []
        for _, row in df.iterrows():
            labels.append(row['type'])
        if len(labels) > 0:
            print(i, start_time, end_time, labels)
        return i, labels

    def add_labels_to_h5_file(self, path_to_h5_file, n_workers=16):
        file = h5py.File(path_to_h5_file, 'r+')

        time_intervals = file['time_interval'][...]
        inputs = []
        for i in range(len(time_intervals)):
            inputs.append(
                (i, UTCDateTime(time_intervals[i][0].decode('utf-8')),
                 UTCDateTime(time_intervals[i][1].decode('utf-8'))))
        with WorkerPool(n_jobs=n_workers) as pool:
            idx_and_labels = pool.map(self.get_window_label,
                                      inputs,
                                      progress_bar=True)

        max_label_len = max([len(j) for _, j in idx_and_labels])
        label_dataset = file.require_dataset(
            'labels', (file['waveform'].shape[0], max_label_len),
            chunks=True,
            dtype=h5py.string_dtype())

        for i, label in idx_and_labels:
            label_dataset[i, :len(label)] = label
        file.close()
