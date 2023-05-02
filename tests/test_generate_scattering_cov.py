import torch
import numpy as np
import unittest

from scatcov.frontend import analyze, load_data
from facvae.utils import Pooling


class TestGenerateScatteringCov(unittest.TestCase):

    def setUp(self):
        self.x = load_data(process_name='fbm', R=1, T=1024).astype(np.float32)
        self.avgpool_base = 4
        self.avgpool_exp = [2, 3, 4, 5]
        self.y = analyze(
            self.x,
            Q=[1, 1],
            J=[8, 8],
            r=2,
            keep_ps=True,
            model_type='scat+cov',
            cuda=False,
            normalize='each_ps',
            estim_operator=Pooling(
                kernel_size=self.avgpool_base**min(self.avgpool_exp)),
            qs=None,
            nchunks=1).y

    def test_scatcov(self):
        for avgpool_exp in self.avgpool_exp:
            avg_pool = Pooling(
                kernel_size=self.avgpool_base**(avgpool_exp -
                                                min(self.avgpool_exp)))
            scatcov_estimated = avg_pool(self.y)
            y = analyze(self.x,
                        Q=[1, 1],
                        J=[8, 8],
                        r=2,
                        keep_ps=True,
                        model_type='scat+cov',
                        cuda=False,
                        normalize='each_ps',
                        estim_operator=Pooling(
                            kernel_size=self.avgpool_base**avgpool_exp),
                        qs=None,
                        nchunks=1).y
            self.assertTrue(torch.all(torch.isclose(y, scatcov_estimated)))


if __name__ == '__main__':
    unittest.main()
