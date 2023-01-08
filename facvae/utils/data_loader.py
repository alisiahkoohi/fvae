import h5py
import numpy as np
import torch
import pickle
from mpire import WorkerPool
from obspy.core import UTCDateTime
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from typing import Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from facvae.utils import (GaussianDataset, CrescentDataset,
                          CrescentCubedDataset, SineWaveDataset, AbsDataset,
                          SignDataset, FourCirclesDataset, DiamondDataset,
                          TwoSpiralsDataset, CheckerboardDataset,
                          TwoMoonsDataset, RunningStats, plotsdir)

NORMALIZATION_BATCH_SIZE = 10000
IPCA_BATCH_SIZE = 65536//4
IPCA_NUM_COMPONENTS = 512


class MarsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 file_path,
                 train_proportion,
                 data_types=['scat_cov'],
                 load_to_memory=True,
                 normalize_data=False,
                 filter_key=[]):
        self.load_to_memory = load_to_memory
        self.normalize_data = normalize_data
        self.filter_key = filter_key

        # HDF5 file.
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')
        self.num_windows = self.file['scat_cov'].shape[0]
        dim_reduced_datasets = [
            'scat_cov_pca', 'scat_cov_pca_25', 'scat_cov_pca_75',
            'scat_cov_pca_125', 'cov_pca', 'cov_pca_25', 'cov_pca_75',
            'cov_pca_125', 'cov_pca_6-6', 'cov_pca_6-6_25', 'cov_pca_6-6_75',
            'cov_pca_6-6_125'
        ]
        self.shape = {
            'waveform':
            self.file['waveform'].shape[1:],
            'scat_cov': (np.prod(self.file['scat_cov'].shape[1]),
                         np.prod(self.file['scat_cov'].shape[-2:])),
        }
        for dim_red_ds in dim_reduced_datasets:
            if dim_red_ds in self.file.keys():
                self.shape[dim_red_ds] = (1,
                                          np.prod(
                                              self.file[dim_red_ds].shape[1]))

        (self.file_idx, self.train_idx, self.val_idx,
         self.test_idx) = self.split_data(train_proportion)

        if filter_key:
            idxs_dict = {str(i): idx for i, idx in enumerate(self.file_idx)}
            self.idx_converter = lambda idx: np.array(
                [idxs_dict[str(j)] for j in idx])
        else:
            self.idx_converter = lambda idx: idx

        self.data = {'waveform': None, 'scat_cov': None}
        self.normalizer = {'waveform': None, 'scat_cov': None}
        self.already_normalized = {'waveform': False, 'scat_cov': False}
        for dim_red_ds in dim_reduced_datasets:
            if dim_red_ds in self.file.keys():
                self.data[dim_red_ds] = None
                self.normalizer[dim_red_ds] = None
                self.already_normalized[dim_red_ds] = None

        if self.load_to_memory:
            self.load_all_data(data_types)

        if self.normalize_data:
            self.setup_data_normalizer(data_types)

    def split_data(self, train_proportion):
        if self.filter_key:
            all_filenames = self.get_waveform_filename(range(self.num_windows))
            file_idx = []
            for filter_key in self.filter_key:
                filtered_filenames = list(
                    filter(lambda k: filter_key in k[1],
                           enumerate(all_filenames)))
                file_idx.extend([file[0] for file in filtered_filenames])
            file_idx = np.array(file_idx)

            ntrain = int(train_proportion * file_idx.shape[0])
            nval = int((1 - train_proportion) * file_idx.shape[0])

            idxs = np.random.permutation(file_idx.shape[0])
            train_idx = idxs[:ntrain]
            val_idx = idxs[ntrain:ntrain + nval]
            test_idx = idxs

        else:
            file_idx = np.array(list(range(self.num_windows)))
            ntrain = int(train_proportion * file_idx.shape[0])
            nval = int((1 - train_proportion) * file_idx.shape[0])

            idxs = np.random.permutation(file_idx.shape[0])
            train_idx = idxs[:ntrain]
            val_idx = idxs[ntrain:ntrain + nval]
            test_idx = idxs

        return file_idx, train_idx, val_idx, test_idx

    def load_all_data(self, data_types):
        for type in data_types:
            self.data[type] = torch.from_numpy(
                self.file[type][self.file_idx,
                                ...].reshape(self.file_idx.shape[0],
                                             *self.shape[type]))

    def setup_data_normalizer(self, data_types):
        for type in data_types:
            running_stats = RunningStats(self.shape[type], dtype=torch.float32)
            if self.load_to_memory:
                running_stats.input_samples(self.data[type][self.train_idx,
                                                            ...])
            else:
                for i in range(0, len(self.train_idx),
                               NORMALIZATION_BATCH_SIZE):
                    batch = torch.from_numpy(
                        self.file[type][self.idx_converter(
                            np.sort(self.train_idx[i:i +
                                                   NORMALIZATION_BATCH_SIZE])),
                                        ...])
                    running_stats.input_samples(batch.reshape(
                        batch.shape[0], *self.shape[type]),
                                                n_workers=1)
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

    def sample_data(self, idx, type):
        if not isinstance(type, list):
            type = [type]
        out = []
        for t in type:
            if self.data[t] is None:
                x = self.file[t][self.idx_converter(np.sort(idx)),
                                 ...].reshape(len(idx), *self.shape[t])
                x = torch.from_numpy(x)
                x = self.normalize(x, t)
                out.append(x)
            else:
                if (not self.already_normalized[t]) and self.normalize_data:
                    self.data[t] = self.normalize(self.data[t][...], t)
                    self.already_normalized[t] = True
                out.append(self.data[t][np.sort(idx), ...])
        return out

    def get_labels(self, idx):
        labels = []
        if len(idx) > 0:
            labels_list = self.file['labels'][self.idx_converter(np.sort(idx)),
                                              ...].astype(str)

            for i in range(len(idx)):
                label = []
                for v in labels_list[i]:
                    if v != '':
                        label.append(v)
                labels.append(label)
        return labels

    def get_drops(self, idx):
        drops = []
        if len(idx) > 0:
            drops_list = self.file['pressure'][
                self.idx_converter(np.sort(idx)), ...]

            for i, _ in enumerate(idx):
                drop = []
                for v in drops_list[i]:
                    if v != 0.0:
                        drop.append(v)
                drops.append(drop)
        return drops

    def get_glitches(self, idx):
        glitches = []
        if len(idx) > 0:
            glitches_list = self.file['glitches'][
                self.idx_converter(np.sort(idx)), ...]

            for i, _ in enumerate(idx):
                glitch = []
                for v in glitches_list[i]:
                    if v == 1.0:
                        glitch.append(v)
                glitches.append(glitch)
        return glitches

    def get_waveform_filename(self, idx):
        if hasattr(self, 'idx_converter'):
            filenames = [
                f.decode('utf-8') for f in self.file['filename'][
                    self.idx_converter(np.sort(idx)), ...]
            ]
        else:
            filenames = [
                f.decode('utf-8')
                for f in self.file['filename'][np.sort(idx), ...]
            ]
        return filenames

    def get_time_interval(self, idx):
        return [(UTCDateTime(s.decode('utf-8')),
                 UTCDateTime(e.decode('utf-8')))
                for s, e in self.file['time_interval'][
                    self.idx_converter(np.sort(idx)), ...]]

    def pca_dim_reduction(self,
                          type='scat_cov',
                          dim_reduced_type='scat_cov_pca'):
        ipca = IncrementalPCA(n_components=IPCA_NUM_COMPONENTS)

        for i in tqdm(range(0, self.num_windows, IPCA_BATCH_SIZE)):
            batch = self.file[type][i:i + IPCA_BATCH_SIZE, ...]
            batch = batch.reshape(batch.shape[0], -1)
            batch = StandardScaler().fit_transform(batch)
            ipca.partial_fit(batch)

        self.file.close()
        self.file = h5py.File(self.file_path, 'r+')
        self.file.require_dataset(dim_reduced_type,
                                  (self.num_windows, IPCA_NUM_COMPONENTS),
                                  chunks=(128, IPCA_NUM_COMPONENTS),
                                  dtype=np.float32)

        for i in tqdm(range(0, self.num_windows, IPCA_BATCH_SIZE)):
            batch = self.file[type][i:i + IPCA_BATCH_SIZE, ...]
            batch = batch.reshape(batch.shape[0], -1)
            batch = StandardScaler().fit_transform(batch)
            batch = ipca.transform(batch)
            self.file[dim_reduced_type][i:i + batch.shape[0], :] = batch
        self.file.close()
        self.file = h5py.File(self.file_path, 'r')

        pca_pkl_file = open(
            ''.join(self.file_path.split('.')[:-1]) +
            '_pca-n-comp-{}_pca-batchsize-{}.pkl'.format(
                IPCA_NUM_COMPONENTS, IPCA_BATCH_SIZE), 'wb')
        pickle.dump(ipca, pca_pkl_file)
        pca_pkl_file.close()

    def plot_pca_var_ratio(self, n_comp=None, batchsize=None):

        if n_comp is None:
            n_comp = IPCA_NUM_COMPONENTS
        if batchsize is None:
            batchsize = IPCA_BATCH_SIZE

        pca_pkl_filename = (''.join(self.file_path.split('.')[:-1]) +
                            '_pca-n-comp-{}_pca-batchsize-{}.pkl'.format(
                                n_comp, batchsize))

        with open(pca_pkl_filename, 'rb') as f:
            ipca = pickle.load(f)

        plt.figure()
        plt.plot(ipca.explained_variance_ratio_)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()


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
        self.shape = (1, self.data.shape[-1])

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

    def sample_data(self, idx, type):
        return self.data[idx, ...].unsqueeze(1)


class Normalizer():
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
        if target_column_name == 'drop':
            df = self.df[(self.df['eventTime'] >= start_time)
                         & (self.df['eventTime'] <= end_time)]
        elif target_column_name == 'type' or target_column_name == 'glitch':
            df = self.df[((self.df['start_time'] > start_time) &
                          (self.df['start_time'] < end_time)) |
                         ((self.df['end_time'] > start_time) &
                          (self.df['end_time'] < end_time))]
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
            if target_column_name == 'type' else np.float32)

        for i, label in idx_and_labels:
            label_dataset[i, :len(label)] = label
        file.close()
