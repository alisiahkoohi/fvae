import torch
import numpy as np
import unittest
from facvae.utils import RunningStats


class TestRunningStats(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 3)
        self.dtype = torch.float32

    def test_init(self):
        rs = RunningStats(self.shape, self.dtype)
        self.assertEqual(rs.num_samples, 0)
        self.assertEqual(rs.shape, self.shape)
        self.assertEqual(rs.dtype, self.dtype)
        self.assertTrue(
            torch.all(
                torch.eq(rs.running_mean,
                         torch.zeros(self.shape, dtype=self.dtype))))
        self.assertTrue(
            torch.all(
                torch.eq(rs.running_sum_of_differences,
                         torch.zeros(self.shape, dtype=self.dtype))))

    def test_input_samples(self):
        rs = RunningStats(self.shape, self.dtype)
        samples = torch.tensor([[1, 2, 3], [4, 5, 6]],
                               dtype=self.dtype).unsqueeze(0)
        samples = samples.repeat(100, 1, 1)
        rs.input_samples(samples)
        self.assertEqual(rs.num_samples, 100)
        self.assertTrue(
            torch.all(
                torch.isclose(
                    rs.running_mean,
                    torch.tensor([[1., 2., 3.], [4., 5., 6.]],
                                 dtype=self.dtype))))
        self.assertTrue(
            torch.all(
                torch.isclose(
                    rs.running_sum_of_differences,
                    torch.tensor([[0., 0., 0.], [0., 0., 0.]],
                                 dtype=self.dtype))))

    def test_compute_stats(self):

        for n_workers in [1, 8]:
            for batchsize  in [25, 100]:
                rs = RunningStats(self.shape, self.dtype)
                samples = torch.tensor([[1, 2, 3], [4, 5, 6]],
                                    dtype=self.dtype).unsqueeze(0)
                samples = samples.repeat(100, 1, 1)
                for i in range(0, samples.shape[0], batchsize):
                    batch = samples[i:i + batchsize, ...]
                    rs.input_samples(batch, n_workers=n_workers)
                mean, std = rs.compute_stats()
                self.assertTrue(
                    torch.all(
                        torch.isclose(
                            mean,
                            torch.tensor([[1., 2., 3.], [4., 5., 6.]],
                                        dtype=self.dtype))))
                self.assertTrue(
                    torch.all(
                        torch.isclose(
                            std,
                            torch.tensor([[0., 0., 0.], [0., 0., 0.]],
                                        dtype=self.dtype))))


if __name__ == '__main__':
    unittest.main()
