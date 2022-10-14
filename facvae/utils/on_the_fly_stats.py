import torch


class RunningStats(object):
    """Running first- and second-order statistics.
    """

    def __init__(self, shape, dtype=torch.float32):
        super(RunningStats, self).__init__()

        self.num_samples = 0
        self.running_mean = torch.zeros(shape, dtype=dtype)
        self.running_sum_of_differences = torch.zeros(shape, dtype=dtype)

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
