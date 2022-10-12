import h5py
import numpy as np
import torch
import pickle
from mpire import WorkerPool
from obspy.core import UTCDateTime
from typing import Optional
import logging
from tqdm import tqdm

from facvae.utils import (GaussianDataset, CrescentDataset,
                          CrescentCubedDataset, SineWaveDataset, AbsDataset,
                          SignDataset, FourCirclesDataset, DiamondDataset,
                          TwoSpiralsDataset, CheckerboardDataset,
                          TwoMoonsDataset, RunningStats)

NORMALIZATION_BATCH_SIZE = 10000


class MarsDataset(torch.utils.data.Dataset):

    def __init__(self,
                 file_path,
                 train_proportion,
                 data_types=['scat_cov'],
                 load_to_memory=False,
                 normalize_data=True):
        self.load_to_memory = load_to_memory
        self.normalize_data = normalize_data

        # HDF5 file.
        self.file = h5py.File(file_path, 'r')
        self.num_windows = self.file['scat_cov'].shape[0]
        self.shape = {
            'waveform': np.prod(self.file['waveform'][0, 1, :].shape),
            'scat_cov': np.prod(self.file['scat_cov'][0, 1, :, :].shape)
        }
        self.train_idx, self.val_idx, self.test_idx = self.split_data(
            self.num_windows, train_proportion)

        self.data = {
            'waveform': None,
            'scat_cov': None,
        }
        self.normalizer = {
            'waveform': None,
            'scat_cov': None,
        }

        if self.load_to_memory:
            self.load_all_data(data_types)

        if self.normalize_data:
            self.setup_data_normalizer(data_types)
        self.already_normalized = {
            'waveform': False,
            'scat_cov': False,
        }

    def split_data(self, num_windows, train_proportion):
        ntrain = int(train_proportion * num_windows)
        nval = int((1 - train_proportion) * num_windows)

        idxs = np.random.permutation(num_windows)
        train_idx = idxs[:ntrain]
        val_idx = idxs[ntrain:ntrain + nval]
        test_idx = idxs

        return train_idx, val_idx, test_idx

    def setup_data_normalizer(self, data_types):
        logging.info('Setting up data normalizer...')

        for type in data_types:
            running_stats = RunningStats(self.shape[type])
            if self.load_to_memory:
                running_stats.input_samples(
                    self.data[type][np.sort(self.train_idx), ...])
                mean, std = running_stats.compute_stats()
            else:
                for i in range(0, len(self.train_idx),
                               NORMALIZATION_BATCH_SIZE):
                    running_stats.input_samples(
                        self.sample_data(
                            self.train_idx[i:i + NORMALIZATION_BATCH_SIZE],
                            type=type))
                mean, std = running_stats.compute_stats()

            self.normalizer[type] = Normalizer(mean, std)

    def normalize(self, x, type):
        if self.normalizer[type]:
            x = self.normalizer[type].normalize(x)
        return x

    def unnormalize(self, x, type):
        if self.normalizer[type]:
            x = self.normalizer[type].unnormalize(x)
        return x

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
            self.data[type] = x

    def sample_data(self, idx, type):
        if self.data[type] is None:
            if type == 'scat_cov':
                x = self.file['scat_cov'][np.sort(idx),
                                          1, :, :].reshape(len(idx), -1)
            elif type == 'waveform':
                x = self.file['waveform'][np.sort(idx), 1, :]
            else:
                raise ValueError('No dataset exists with type ', type)
            x = self.normalize(x, type)
            x = torch.from_numpy(x)
            return x
        else:
            if (not self.already_normalized[type]) and self.normalize_data:
                self.data[type] = self.normalize(self.data[type][...], type)
                self.already_normalized[type] = True
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


class Normalizer(object):
    """Normalizer a tensor image with training mean and standard deviation.
    Extracts the mean and standard deviation from the training dataset, and
    uses them to normalize an input image.

    Attributes:
        mean: A torch.Tensor containing the mean over the training dataset.
        std: A torch.Tensor containing the standard deviation over the
        training. eps: A small float to avoid dividing by 0.
    """

    def __init__(self,
                 mean: torch.Tensor,
                 std: torch.Tensor,
                 eps: Optional[int] = 1e-6):
        """Initializes a Normalizer object.
        Args:
            mean: A torch.Tensor that contains the mean of input data.
            std: A torch.Tensor that contains the standard deviation of input.
            dimension. eps: A optional small float to avoid dividing by 0.
        """
        super().__init__()

        # Compute the training dataset mean and standard deviation over the
        # batch dimensions.
        self.mean = mean
        self.std = std
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to input sample.
        Args:
            x: A torch.Tensor with the same dimension organization as
            `dataset`.
        Returns:
            A torch.Tensor with the same dimension organization as `x` but
            normalized with the mean and standard deviation of the training
            dataset.
        """
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Restore the normalization from the input sample.
        Args:
            x: A normalized torch.Tensor with the same dimension organization
            as `dataset`.
        Returns:
            A torch.Tensor with the same dimension organization as `x` that has
            been unnormalized.
        """
        return x * (self.std + self.eps) + self.mean


class CatalogReader(torch.utils.data.Dataset):

    def __init__(self, path_to_catalog, window_size, frequency=20.0):

        self.window_size = window_size
        self.frequency = frequency

        with open(path_to_catalog, 'rb') as f:
            self.df = pickle.load(f)

    def get_window_label(self, i, start_time, end_time, target_column_name):
        df = self.df[(self.df['eventTime'] >= start_time)
                     & (self.df['eventTime'] <= end_time)]
        labels = []
        for _, row in df.iterrows():
            labels.append(row[target_column_name])
        if len(labels) > 0:
            print(i, start_time, end_time, labels)
        return i, labels

    def add_events_to_h5_file(self,
                              path_to_h5_file,
                              h5_dataset_name,
                              target_column_name,
                              n_workers=16):
        file = h5py.File(path_to_h5_file, 'r+')

        time_intervals = file['time_interval'][...]
        inputs = []
        for i in tqdm(range(len(time_intervals))):
            inputs.append(
                (i, UTCDateTime(time_intervals[i][0].decode('utf-8')),
                 UTCDateTime(time_intervals[i][1].decode('utf-8')),
                 target_column_name))
        with WorkerPool(n_jobs=n_workers) as pool:
            idx_and_labels = pool.map(self.get_window_label,
                                      inputs,
                                      progress_bar=True)

        max_label_len = max([len(j) for _, j in idx_and_labels])
        label_dataset = file.require_dataset(
            h5_dataset_name, (file['waveform'].shape[0], max_label_len),
            chunks=True,
            dtype=h5py.string_dtype()
            if target_column_name == 'label' else np.float32)

        for i, label in idx_and_labels:
            label_dataset[i, :len(label)] = label
        file.close()
