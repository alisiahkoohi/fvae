import torch
from mpire import WorkerPool
import numpy as np
from typing import List, Tuple, Optional


class RunningStats:
    """A class to compute running first- and second-order statistics."""
    def __init__(self,
                 shape: Tuple[int, ...],
                 dtype: torch.dtype = torch.float32):
        """
        Initialize a RunningStats object.

        Args:
            shape: A tuple indicating the shape of the input data.
            dtype: The data type used for the computation (default is
                torch.float32).
        """
        super(RunningStats, self).__init__()
        self.num_samples = 0
        self.shape = shape
        self.dtype = dtype
        self.running_mean = torch.zeros(shape, dtype=dtype)
        self.running_sum_of_differences = torch.zeros(shape, dtype=dtype)

    def input_samples(self, samples: torch.Tensor, n_workers: int = 8) -> None:
        """
        Input new samples and update the running mean and sum of squared
        differences.

        Args:
            samples: A tensor of shape (N, *) where N is the number of samples
                and * is consistent with self.shape.
            n_workers: The number of workers used to compute the running
                statistics (default is 8).

        Returns:
            None
        """
        if n_workers > 1:
            # Split the indices into n_workers chunks.
            split_idxs = np.array_split(np.arange(samples.shape[0]),
                                        n_workers,
                                        axis=0)
            with WorkerPool(n_jobs=n_workers,
                            shared_objects=samples,
                            start_method='fork') as pool:
                # Map the worker function to the indices using a worker pool.
                outputs = pool.map(self.serial_worker,
                                   split_idxs,
                                   progress_bar=True)
            # Unpack the outputs from the worker pool.
            (num_samples, running_mean,
             running_sum_of_differences) = zip(*outputs)
            # Update the total number of samples.
            self.num_samples = sum(num_samples)
            # Update the running mean.
            self.running_mean = sum([
                mean / self.num_samples * num_samples
                for mean, num_samples in zip(running_mean, num_samples)
            ])
            # Update the running sum of squared differences.
            self.running_sum_of_differences += sum(
                [sum_of_diff for sum_of_diff in running_sum_of_differences])
        else:
            # Compute the running statistics using a single worker.
            (self.num_samples, self.running_mean,
             self.running_sum_of_differences) = self.serial_worker(
                 samples, range(samples.shape[0]))

    def serial_worker(
            self, samples: torch.Tensor,
            split_idxs: List[int]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Compute the running mean and sum of squared differences using a single
        worker.

        Args:
            samples: A tensor of shape (N, *) where N is the number of samples
                and * represents any number of dimensions.
            split_idxs: A list of indices to process.

        Returns:
            A tuple containing the number of processed samples, the running
            mean, and the running sum of squared differences.
        """
        num_samples = 0
        running_mean = torch.zeros(self.shape, dtype=self.dtype)
        running_sum_of_differences = torch.zeros(self.shape, dtype=self.dtype)
        for i in split_idxs:
            # Update the running mean.
            num_samples += 1
            delta = samples[i, ...] - running_mean
            running_mean = running_mean + delta / num_samples
            # Update the running sum of squared differences.
            delta2 = samples[i, ...] - running_mean
            running_sum_of_differences += delta * delta2
        return num_samples, running_mean, running_sum_of_differences

    def compute_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the weighted mean and standard deviation.

        Returns:
            Tuple of two torch Tensors, where the first element is the running
            mean and the second element is the standard deviation.
        """
        # Compute the standard deviation based on the intermediate weighted sum
        # and sum of squares.
        std = torch.sqrt(self.running_sum_of_differences / self.num_samples)

        # Return the running mean and standard deviation as a tuple.
        return self.running_mean, std


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
