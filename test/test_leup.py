import unittest
import numpy as np

from matfact.leup import leup
from matfact.utils import random_rational_matrix


class TestLEUP(unittest.TestCase):
    def test_random(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(5):
            A = random_rational_matrix(rng, 8, 4, 0.2)
            L, E, U, P = leup(A)
            self.assertTrue((L @ E @ U @ P == A).all())

            A = random_rational_matrix(rng, 4, 8, 0.2)
            L, E, U, P = leup(A)
            self.assertTrue((L @ E @ U @ P == A).all())

            A = random_rational_matrix(rng, 8, 8, 0.2)
            L, E, U, P = leup(A)
            self.assertTrue((L @ E @ U @ P == A).all())
