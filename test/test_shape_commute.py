import unittest
from functools import reduce
from operator import matmul

import numpy as np

from matfact.shape_commute import reduce_boundary_maps
from matfact.utils import random_rational_matrix


class TestShapeCommute(unittest.TestCase):
    def test_random(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(5):
            dimensions = np.random.randint(1, 10, 8)
            boundary_maps = []
            for m, n in zip(dimensions, dimensions[1:]):
                boundary_maps.append(random_rational_matrix(rng, m, n, 0.2))

            L, reduced, U, P = reduce_boundary_maps(boundary_maps)
            left = reduce(matmul, boundary_maps)
            right = L @ reduce(matmul, reduced) @ U @ P

            # NOTE: Uncomment the following line and the test will fail
            # print(left - right)
            self.assertTrue((left == right).all())
