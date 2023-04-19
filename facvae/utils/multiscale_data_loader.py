import h5py
import numpy as np
import torch
from obspy.core import UTCDateTime

from facvae.utils import RunningStats, Normalizer

NORMALIZATION_BATCH_SIZE = 10000


class MarsMultiscaleDataset(torch.utils.data.Dataset):
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
        self.num_windows = self.file['waveform'].shape[0]
        scatcov_datasets = list(self.file['scat_cov'].keys())
        self.scatcov_datasets = scatcov_datasets
        self.shape = {
            'waveform': self.file['waveform'].shape[1:],
            'scat_cov': {}
        }
        for dset_name in scatcov_datasets:
            self.shape['scat_cov'][dset_name] = (
                self.file['scat_cov'][dset_name].shape[2],
                np.prod(self.file['scat_cov'][dset_name].shape[3:]))

        (self.file_idx, self.train_idx, self.val_idx,
         self.test_idx) = self.split_data(train_proportion)

        if filter_key:
            idxs_dict = {str(i): idx for i, idx in enumerate(self.file_idx)}
            self.idx_converter = lambda idx: np.array(
                [idxs_dict[str(j)] for j in idx])
        else:
            self.idx_converter = lambda idx: idx

        self.data = {
            'waveform': None,
            'scat_cov': {dset_name: None
                         for dset_name in scatcov_datasets}
        }
        self.normalizer = {
            'waveform': None,
            'scat_cov': {dset_name: None
                         for dset_name in scatcov_datasets}
        }
        self.already_normalized = {
            'waveform': False,
            'scat_cov': {dset_name: False
                         for dset_name in scatcov_datasets}
        }

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
            if type == 'scat_cov':
                for dset_name in self.scatcov_datasets:
                    self.data['scat_cov'][dset_name] = torch.from_numpy(
                        self.file['scat_cov'][dset_name][
                            self.file_idx,
                            ...].reshape(self.file_idx.shape[0], -1,
                                         *self.shape['scat_cov'][dset_name]))
            else:
                self.data[type] = torch.from_numpy(
                    self.file[type][self.file_idx,
                                    ...].reshape(self.file_idx.shape[0],
                                                 *self.shape[type]))

    def setup_data_normalizer(self, data_types):
        for type in data_types:
            if type == 'scat_cov':
                for dset_name in self.scatcov_datasets:
                    running_stats = RunningStats(
                        self.shape['scat_cov'][dset_name], dtype=torch.float32)
                    if self.load_to_memory:
                        running_stats.input_samples(
                            self.data['scat_cov'][dset_name]
                            [self.train_idx,
                             ...].reshape(-1,
                                          *self.shape['scat_cov'][dset_name]))
                    else:
                        for i in range(0, len(self.train_idx),
                                       NORMALIZATION_BATCH_SIZE):
                            batch = torch.from_numpy(
                                self.file['scat_cov'][dset_name]
                                [self.idx_converter(
                                    np.sort(self.
                                            train_idx[i:i +
                                                      NORMALIZATION_BATCH_SIZE]
                                            )), ...])
                            running_stats.input_samples(batch.reshape(
                                -1, *self.shape['scat_cov'][dset_name]),
                                                        n_workers=1)
                    mean, std = running_stats.compute_stats()
                    self.normalizer['scat_cov'][dset_name] = Normalizer(
                        mean, std)
            else:
                running_stats = RunningStats(self.shape[type],
                                             dtype=torch.float32)
                if self.load_to_memory:
                    running_stats.input_samples(self.data[type][self.train_idx,
                                                                ...])
                else:
                    for i in range(0, len(self.train_idx),
                                   NORMALIZATION_BATCH_SIZE):
                        batch = torch.from_numpy(
                            self.file[type][self.idx_converter(
                                np.sort(self.
                                        train_idx[i:i +
                                                  NORMALIZATION_BATCH_SIZE])),
                                            ...])
                        running_stats.input_samples(batch.reshape(
                            batch.shape[0], *self.shape[type]),
                                                    n_workers=1)
                mean, std = running_stats.compute_stats()
                self.normalizer[type] = Normalizer(mean, std)

    def normalize(self, x, type, dset_name=None):
        if dset_name:
            if self.normalizer[type][dset_name]:
                x = self.normalizer[type][dset_name].normalize(x)
        else:
            if self.normalizer[type]:
                x = self.normalizer[type].normalize(x)
        return x

    def unnormalize(self, x, type, dset_name=None):
        if dset_name:
            if self.normalizer[type][dset_name]:
                x = self.normalizer[type][dset_name].unnormalize(x)
        else:
            if self.normalizer[type]:
                x = self.normalizer[type].unnormalize(x)
        return x

    def sample_data(self, idx, type):
        if not isinstance(type, list):
            type = [type]
        out = []
        for t in type:
            if t == 'scat_cov':
                scatcov_out = {}
                for dset_name in self.scatcov_datasets:
                    if self.data['scat_cov'][dset_name] is None:
                        x = self.file['scat_cov'][dset_name][
                            self.idx_converter(np.sort(idx)),
                            ...].reshape(-1,
                                         *self.shape['scat_cov'][dset_name])
                        x = torch.from_numpy(x)
                        x = self.normalize(x, 'scat_cov', dset_name=dset_name)
                        scatcov_out[dset_name] = x
                    else:
                        if (not self.already_normalized['scat_cov'][dset_name]
                            ) and self.normalize_data:
                            self.data['scat_cov'][dset_name] = self.normalize(
                                self.data['scat_cov'][dset_name][...],
                                'scat_cov',
                                dset_name=dset_name)
                            self.already_normalized['scat_cov'][
                                dset_name] = True
                        scatcov_out[dset_name] = self.data['scat_cov'][
                            dset_name][np.sort(idx), ...].reshape(
                                -1, *self.shape['scat_cov'][dset_name])
                out.append(scatcov_out)
            else:
                if self.data[t] is None:
                    x = self.file[t][self.idx_converter(np.sort(idx)),
                                     ...].reshape(len(idx), *self.shape[t])
                    x = torch.from_numpy(x)
                    x = self.normalize(x, t)
                    out.append(x)
                else:
                    if (not self.already_normalized[t]
                        ) and self.normalize_data:
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
