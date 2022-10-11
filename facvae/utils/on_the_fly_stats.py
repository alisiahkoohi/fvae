import numpy as np
import torch


class OnTheFlyStats(object):
    """On-the-fly computation of weighted first- and second-order statistics.
    """

    def __init__(self):
        super(OnTheFlyStats, self).__init__()

        self.sum_of_weights = 0.0
        self.N = 0

        self.weighted_sum = 0.0
        self.weighted_sum_of_squares = 0.0

    def input_samples(self, samples, weights=None):
        """
        Input new samples and update the weighted sum and sum of squares
        """
        if weights and samples.shape[0] != weights.shape[0]:
            raise ValueError('samples and weights must have the same length')
        elif weights is None:
            weights = np.ones(samples.shape[0])

        sum_of_weights = np.sum(weights)

        self.weighted_sum += np.average(samples, weights=weights,
                                        axis=0) * sum_of_weights

        self.weighted_sum_of_squares += np.average(
            samples**2, weights=weights, axis=0) * sum_of_weights

        self.sum_of_weights += sum_of_weights

        self.N += samples.shape[0]

    def compute_stats(self):
        """
        Return the weighted mean and standard deviation based on the
        intermediate weighted sum and sum of squares
        """

        sample_mean = self.weighted_sum / self.sum_of_weights

        sample_var = (self.weighted_sum_of_squares -
                      self.weighted_sum**2 / self.sum_of_weights)
        sample_var *= self.N / ((self.N - 1) * self.sum_of_weights)

        return sample_mean, np.sqrt(sample_var)


class RunningStats(object):
    """Running first- and second-order statistics.
    """

    def __init__(self, shape):
        super(RunningStats, self).__init__()

        self.num_samples = 0
        self.running_mean = torch.zeros(shape, dtype=torch.float)
        self.running_sum_of_differences = torch.zeros(shape, dtype=torch.float)

    def input_samples(self, samples):
        """
        Input new samples and update the quantities.
        """

        for i in range(samples.shape[0]):
            self.num_samples += 1
            delta = samples[i, ...] - self.running_mean
            self.running_mean = self.running_mean + delta / self.num_samples
            delta2 = samples[i, ...] - self.running_mean
            self.running_sum_of_differences += delta * delta2

    def compute_stats(self):
        """
        Return the weighted mean and standard deviation based on the
        intermediate weighted sum and sum of squares
        """

        return self.running_mean, torch.sqrt(self.running_sum_of_differences /
                                             self.num_samples)
