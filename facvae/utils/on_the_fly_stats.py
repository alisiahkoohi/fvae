import torch
from mpire import WorkerPool
import numpy as np
from typing import Optional

class RunningStats(object):
    """Running first- and second-order statistics.
    """

    def __init__(self, shape, dtype=torch.float32):
        super(RunningStats, self).__init__()

        self.num_samples = 0
        self.shape = shape
        self.dtype = dtype
        self.running_mean = torch.zeros(shape, dtype=dtype)
        self.running_sum_of_differences = torch.zeros(shape, dtype=dtype)

    def input_samples(self, samples, n_workers=8):
        """
        Input new samples and update the quantities.
        """
        if n_workers > 1:
            split_idxs = np.array_split(np.arange(samples.shape[0]),
                                        n_workers,
                                        axis=0)
            with WorkerPool(n_jobs=n_workers,
                            shared_objects=samples,
                            start_method='fork') as pool:
                outputs = pool.map(self.serial_worker,
                                   split_idxs,
                                   progress_bar=True)

            # Unpack the outputs.
            (num_samples, running_mean,
             running_sum_of_differences) = zip(*outputs)
            self.num_samples = sum(num_samples)
            self.running_mean = sum([
                mean / self.num_samples * num_samples
                for mean, num_samples in zip(running_mean, num_samples)
            ])
            self.running_sum_of_differences += sum(
                [sum_of_diff for sum_of_diff in running_sum_of_differences])

        else:
            (self.num_samples, self.running_mean,
             self.running_sum_of_differences) = self.serial_worker(
                 samples, range(samples.shape[0]))

    def serial_worker(self, samples, split_idxs):
        num_samples = 0
        running_mean = torch.zeros(self.shape, dtype=self.dtype)
        running_sum_of_differences = torch.zeros(self.shape, dtype=self.dtype)
        for i in split_idxs:
            num_samples += 1
            delta = samples[i, ...] - running_mean
            running_mean = running_mean + delta / num_samples
            delta2 = samples[i, ...] - running_mean
            running_sum_of_differences += delta * delta2
        return num_samples, running_mean, running_sum_of_differences

    def compute_stats(self):
        """
        Return the weighted mean and standard deviation based on the
        intermediate weighted sum and sum of squares
        """
        return self.running_mean, torch.sqrt(self.running_sum_of_differences /
                                             self.num_samples)


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
