import h5py
import numpy as np
import torch
import pickle
from mpire import WorkerPool
from obspy.core import UTCDateTime
from tqdm import tqdm


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
            df = self.df[(self.df['start_time'] <= end_time)
                         & (self.df['end_time'] >= start_time)]
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
                              n_workers=40):

        file = h5py.File(path_to_h5_file, 'r')
        scales = list(file['time_interval'].keys())
        file.close()

        for scale in scales:
            file = h5py.File(path_to_h5_file, 'r')
            time_intervals = file['time_interval'][scale][...]
            file.close()

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

            file = h5py.File(path_to_h5_file, 'r+')
            label_group = file.require_group(h5_dataset_name)
            label_dataset = label_group.require_dataset(
                scale, (file['time_interval'][scale].shape[0], max_label_len),
                chunks=True,
                dtype=h5py.string_dtype()
                if target_column_name == 'type' else np.float32)

            for i, label in idx_and_labels:
                label_dataset[i, :len(label)] = label
            file.close()
