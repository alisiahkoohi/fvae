from typing import Optional, Tuple, List, Dict, Union

import h5py
import numpy as np
import torch
from obspy.core import UTCDateTime

from facvae.utils import RunningStats, Normalizer

NORMALIZATION_BATCH_SIZE = 10000


class MarsMultiscaleDataset():
    """
    A lass to load data from a HDF5 file containing multiscale Martian  data.

    Args:
        file_path (str): Path to the HDF5 file containing the seismic data.
        train_proportion (float): Proportion of the data to be used for
        training.
        data_types (list, optional): List of data types to load from the file.
            The default is ['scat_cov'].
        scatcov_datasets (list, optional): List of scat_cov datasets to load
            from the file. The default is None, in which case all scat_cov
            datasets in the file are loaded.
        load_to_memory (bool, optional): Whether to load the data to memory or
            access it from disk. The default is True.
        normalize_data (bool, optional): Whether to normalize the data. The
            default is False.
        filter_key (list, optional): List of dataset keys to filter the data.
            The default is [].

    Attributes:
        load_to_memory (bool): Whether the data is loaded to memory or accessed
            from disk.
        normalize_data (bool): Whether the data is normalized.
        filter_key (list): List of dataset keys to filter the data.
        file_path (str): Path to the HDF5 file containing the seismic data.
        file (h5py.File): HDF5 file object.
        num_windows (int): Number of data windows in the file.
        scatcov_datasets (list): List of scat_cov datasets in the file.
        shape (dict): Dictionary of shapes of the waveform and scat_cov
            datasets.
        file_idx (list): List of indices of the data windows in the file.
        train_idx (list): List of indices of the data windows used for
            training.
        val_idx (list): List of indices of the data windows used for
            validation.
        test_idx (list): List of indices of the data windows used for testing.
            idx_converter (function): Function to convert the indices from the
            filtered data to the original data.
        data (dict): Dictionary of waveform and scat_cov data.
        normalizer (dict): Dictionary of waveform and scat_cov normalizers.
        already_normalized (dict): Dictionary of flags indicating whether the
            data has already been normalized.

    """
    def __init__(
        self,
        file_path: str,
        train_proportion: float,
        data_types: list = ['scat_cov'],
        scatcov_datasets: list = None,
        load_to_memory: bool = True,
        normalize_data: bool = False,
        filter_key: list = [],
    ) -> None:
        """
        Initializes the MarsMultiscaleDataset class.
        """
        # Set the class attributes based on the constructor arguments.
        self.load_to_memory = load_to_memory
        self.normalize_data = normalize_data
        self.filter_key = filter_key

        # Open the HDF5 file specified by the file_path argument in read mode.
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')

        # Get the number of windows in the waveform dataset.
        self.num_windows = self.file['waveform'].shape[0]

        # If scatcov_datasets is not specified, get a list of all datasets in
        # the scat_cov group.
        if not scatcov_datasets:
            scatcov_datasets = list(self.file['scat_cov'].keys())
        self.scatcov_datasets = scatcov_datasets

        # Create a dictionary that stores the shape of the waveform and
        # scat_cov datasets.
        self.shape = {
            'waveform': self.file['waveform'].shape[1:],
            'scat_cov': {}
        }

        # For each dataset in scatcov_datasets, store its shape in the
        # dictionary.
        for dset_name in scatcov_datasets:
            self.shape['scat_cov'][dset_name] = (
                self.file['scat_cov'][dset_name].shape[2],
                np.prod(self.file['scat_cov'][dset_name].shape[3:]))

        # Split the data into training, validation, and test sets using the
        # specified proportion.
        (self.file_idx, self.train_idx, self.val_idx,
         self.test_idx) = self.split_data(train_proportion)

        # If filter_key is True, create a dictionary that maps indices in the
        # original data to indices in the filtered data.
        if filter_key:
            idxs_dict = {str(i): idx for i, idx in enumerate(self.file_idx)}
            self.idx_converter = lambda idx: np.array(
                [idxs_dict[str(j)] for j in idx])
        else:
            self.idx_converter = lambda idx: idx

        # Create dictionaries to store the data and normalization parameters
        # for each dataset.
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

        # If load_to_memory is True, load all data into memory.
        if self.load_to_memory:
            self.load_all_data(data_types)

        # If normalize_data is True, compute the normalizat ion parameters for
        # each dataset.
        if self.normalize_data:
            self.setup_data_normalizer(data_types)

    def split_data(
        self, train_proportion: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data into train, validation and test sets.

        Args:
            train_proportion (float): Proportion of the data to use for
            training.

        Returns:
            Tuple of four NumPy arrays containing the file indices for the
            train, validation and test sets.
        """
        if self.filter_key:
            # Get the filenames that match the filter keys.
            all_filenames = self.get_waveform_filename(range(self.num_windows))
            file_idx = []
            for filter_key in self.filter_key:
                filtered_filenames = list(
                    filter(lambda k: filter_key in k[1],
                           enumerate(all_filenames)))
                file_idx.extend([file[0] for file in filtered_filenames])
            file_idx = np.array(file_idx)

            # Calculate the sizes of the train and validation sets.
            ntrain = int(train_proportion * file_idx.shape[0])
            nval = int((1 - train_proportion) * file_idx.shape[0])

            # Shuffle the file indices.
            idxs = np.random.permutation(file_idx.shape[0])

            # Assign file indices to the train, validation and test sets.
            train_idx = idxs[:ntrain]
            val_idx = idxs[ntrain:ntrain + nval]
            test_idx = idxs

        else:
            # No filter keys specified, use all file indices.
            file_idx = np.array(list(range(self.num_windows)))

            # Calculate the sizes of the train and validation sets.
            ntrain = int(train_proportion * file_idx.shape[0])
            nval = int((1 - train_proportion) * file_idx.shape[0])

            # Shuffle the file indices.
            idxs = np.random.permutation(file_idx.shape[0])

            # Assign file indices to the train, validation and test sets.
            train_idx = idxs[:ntrain]
            val_idx = idxs[ntrain:ntrain + nval]
            test_idx = idxs

        return file_idx, train_idx, val_idx, test_idx

    def load_all_data(self, data_types: List[str]) -> None:
        """
        Load all the data of the given types from the file into memory.

        Args:
            data_types (List[str]): List of strings representing the data types
            to be loaded.

        Returns:
            None
        """
        for type in data_types:
            if type == 'scat_cov':
                for dset_name in self.scatcov_datasets:
                    # Load scat_cov data
                    self.data['scat_cov'][dset_name] = torch.from_numpy(
                        self.file['scat_cov'][dset_name][
                            self.file_idx,
                            ...].reshape(self.file_idx.shape[0], -1,
                                         *self.shape['scat_cov'][dset_name]))
            else:
                # Load other data types
                self.data[type] = torch.from_numpy(
                    self.file[type][self.file_idx,
                                    ...].reshape(self.file_idx.shape[0],
                                                 *self.shape[type]))

    def setup_data_normalizer(self, data_types: List[str]) -> None:
        """
        Sets up the data normalizer for each data type in data_types.

        Args:
            data_types (List[str]): A list of data types for which the
            normalizer needs to be set up.

        Returns: None
        """

        # For each data type in data_types.
        for type in data_types:
            # If the data type is 'scat_cov'.
            if type == 'scat_cov':
                # For each dataset in the 'scat_cov_datasets'.
                for dset_name in self.scatcov_datasets:
                    # Create a RunningStats object with the shape of the
                    # current dataset.
                    running_stats = RunningStats(
                        self.shape['scat_cov'][dset_name], dtype=torch.float32)
                    if self.load_to_memory:
                        # If the data is loaded in memory, add the samples to
                        # the running_stats object.
                        running_stats.input_samples(
                            self.data['scat_cov'][dset_name]
                            [self.train_idx,
                             ...].reshape(-1,
                                          *self.shape['scat_cov'][dset_name]))
                    else:
                        # If the data is not loaded in memory, load it from the
                        # file in batches and add the samples to the
                        # running_stats object.
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
                    # Compute the mean and standard deviation of the samples.
                    mean, std = running_stats.compute_stats()
                    # Create a Normalizer object with the computed mean and
                    # standard deviation.
                    self.normalizer['scat_cov'][dset_name] = Normalizer(
                        mean, std)
            else:
                # Create a RunningStats object with the shape of the current
                # data type.
                running_stats = RunningStats(self.shape[type],
                                             dtype=torch.float32)
                if self.load_to_memory:
                    # If the data is loaded in memory, add the samples to the
                    # running_stats object.
                    running_stats.input_samples(self.data[type][self.train_idx,
                                                                ...])
                else:
                    # If the data is not loaded in memory, load it from the
                    # file in batches and add the samples to the running_stats
                    # object.
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
                # Compute the mean and standard deviation of the samples.
                mean, std = running_stats.compute_stats()
                # Create a Normalizer object with the computed mean and
                # standard deviation.
                self.normalizer[type] = Normalizer(mean, std)
        return None

    def normalize(self,
                  x: torch.Tensor,
                  type: str,
                  dset_name: Optional[str] = None) -> torch.Tensor:
        """
        Normalize a given tensor based on its corresponding mean and standard
        deviation.

        Args:
            x (torch.Tensor): input tensor
            type (str): type of data to normalize
            dset_name (Optional[str]): dataset name for data of type
            'scat_cov'. Default is None.

        Returns:
            normalized tensor (torch.Tensor): normalized input tensor

        """
        if dset_name:
            if self.normalizer[type][dset_name]:
                x = self.normalizer[type][dset_name].normalize(x)
        else:
            if self.normalizer[type]:
                x = self.normalizer[type].normalize(x)
        return x

    def unnormalize(self,
                    x: torch.Tensor,
                    type: str,
                    dset_name: Optional[str] = None) -> torch.Tensor:
        """
        Unnormalize a given tensor based on its corresponding mean and standard
        deviation.

        Args:
            x (torch.Tensor): input tensor
            type (str): type of data to unnormalize
            dset_name (Optional[str]): dataset name for data of type
            'scat_cov'. Default is None.

        Returns:
            unnormalized tensor (torch.Tensor): unnormalized input tensor

        """
        if dset_name:
            if self.normalizer[type][dset_name]:
                x = self.normalizer[type][dset_name].unnormalize(x)
        else:
            if self.normalizer[type]:
                x = self.normalizer[type].unnormalize(x)
        return x

    def sample_data(self, idx: np.ndarray,
                    type: str) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Returns a dictionary of samples of the specified data type for each
        dataset.

        Args:
            idx (np.ndarray): A batch of indices to sample from the dataset.
            type (str): The type of data to sample.

        Returns:
            dict: A dictionary containing the samples for each dataset.
        """
        if type == 'scat_cov':
            out = {}
            for dset_name in self.scatcov_datasets:
                if self.data['scat_cov'][dset_name] is None:
                    # If the data is not in memory, read a chunk of data from
                    # the file.
                    x = self.file['scat_cov'][dset_name][
                        self.idx_converter(np.sort(idx)),
                        ...].reshape(-1, *self.shape['scat_cov'][dset_name])
                    x = torch.from_numpy(x)
                    x = self.normalize(x, 'scat_cov', dset_name=dset_name)
                    out[dset_name] = x
                else:
                    if (not self.already_normalized['scat_cov'][dset_name]
                        ) and self.normalize_data:
                        # Normalize the data if it hasn't already been
                        # normalized.
                        self.data['scat_cov'][dset_name] = self.normalize(
                            self.data['scat_cov'][dset_name][...],
                            'scat_cov',
                            dset_name=dset_name)
                        self.already_normalized['scat_cov'][dset_name] = True
                    out[dset_name] = self.data['scat_cov'][dset_name][
                        np.sort(idx),
                        ...].reshape(-1, *self.shape['scat_cov'][dset_name])
        else:
            if self.data[type] is None:
                # If the data is not in memory, read a chunk of data from the
                # file.
                x = self.file[type][self.idx_converter(np.sort(idx)),
                                    ...].reshape(len(idx), *self.shape[type])
                x = torch.from_numpy(x)
                out = self.normalize(x, type)
            else:
                if (not self.already_normalized[type]) and self.normalize_data:
                    # Normalize the data if it hasn't already been normalized.
                    self.data[type] = self.normalize(self.data[type][...],
                                                     type)
                    self.already_normalized[type] = True
                out = self.data[type][np.sort(idx), ...]
        return out

    def get_labels(self, idx: List[int]) -> List[List[str]]:
        """
        Returns the list of labels for the given indices.

        Args:
            idx: A list of indices.

        Returns:
            A list of lists of strings representing the labels for the given
            indices.
        """
        labels = []
        # Check if there are any indices
        if len(idx) > 0:
            # Extract labels for the given indices
            labels_list = self.file['labels'][self.idx_converter(np.sort(idx)),
                                              ...].astype(str)

            # Process each index and create a list of labels for each one
            for i in range(len(idx)):
                label = []
                # Extract the label values and append them to the label list
                for v in labels_list[i]:
                    if v != '':
                        label.append(v)
                labels.append(label)
        return labels

    def get_drops(self, idx: List[int]) -> List[List[float]]:
        """
        Returns the list of drops for the given indices.

        Args:
            idx: A list of indices.

        Returns:
            A list of lists of floats representing the drops for the given
            indices.
        """
        drops = []
        # Check if there are any indices
        if len(idx) > 0:
            # Extract drops for the given indices
            drops_list = self.file['pressure'][
                self.idx_converter(np.sort(idx)), ...]

            # Process each index and create a list of drops for each one
            for i, _ in enumerate(idx):
                drop = []
                # Extract the drop values and append them to the drop list
                for v in drops_list[i]:
                    if v != 0.0:
                        drop.append(v)
                drops.append(drop)
        return drops

    def get_glitches(self, idx: List[int]) -> List[List[float]]:
        """
        Returns the list of glitches for the given indices.

        Args:
            idx: A list of indices.

        Returns:
            A list of lists of floats representing the glitches for the given
            indices.
        """
        glitches = []
        # Check if there are any indices
        if len(idx) > 0:
            # Extract glitches for the given indices
            glitches_list = self.file['glitches'][
                self.idx_converter(np.sort(idx)), ...]

            # Process each index and create a list of glitches for each one
            for i, _ in enumerate(idx):
                glitch = []
                # Extract the glitch values and append them to the glitch list
                for v in glitches_list[i]:
                    if v == 1.0:
                        glitch.append(v)
                glitches.append(glitch)
        return glitches

    def get_waveform_filename(self, idx: List[int]) -> List[str]:
        """
        Returns the list of waveform filenames for the given indices.

        Args:
            idx: A list of indices.

        Returns:
            A list of strings representing the waveform filenames for the given
            indices.
        """
        if hasattr(self, 'idx_converter'):
            # If `idx_converter` exists, it is used to convert the given
            # indices to a format that can be used to index into the `file`
            # dataset.
            filenames = [
                f.decode('utf-8') for f in self.file['filename'][
                    self.idx_converter(np.sort(idx)), ...]
            ]
        else:
            # Otherwise, the indices are sorted and used directly to index into
            # the `file` dataset.
            filenames = [
                f.decode('utf-8')
                for f in self.file['filename'][np.sort(idx), ...]
            ]
        return filenames

    def get_time_interval(
            self, idx: List[int]) -> List[Tuple[UTCDateTime, UTCDateTime]]:
        """
        Returns the list of time intervals for the given indices.

        Args:
            idx: A list of indices.

        Returns:
            A list of tuples representing the time intervals for the given
            indices. Each tuple contains two UTCDateTime objects representing
            the start and end times.
        """
        return [(UTCDateTime(s.decode('utf-8')),
                 UTCDateTime(e.decode('utf-8')))
                for s, e in self.file['time_interval'][
                    self.idx_converter(np.sort(idx)), ...]]
