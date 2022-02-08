import unittest
from leup import leup
import numpy as np
from fractions import Fraction

def random_rational():
    x = np.random.rand()
    if x < 0.3:
        return Fraction(1,2)
    elif x < 0.5:
        return Fraction(2,3)
    elif x < 0.7:
        return Fraction(-3,4)
    else:
        return Fraction(1,1)

def random_rational_matrix(m, n, p=0.1):
    """
    m x n rational matrix
    every entry is non-zero with probability p
    """
    A = np.zeros((m,n), dtype=object)
    for i in range(m):
        for j in range(n):
            if np.random.rand() < p:
                A[i,j] = random_rational()

    return A


class TestLEUP_rational(unittest.TestCase):

	def test_random(self):

		for seed in range(5):
			
			np.random.seed(seed)
			A = random_rational_matrix(8,4,0.2)
			L, E, U, P = leup(A)
			self.assertTrue((L @ E @ U @ P == A).all())

			np.random.seed(seed)
			A = random_rational_matrix(4,8,0.2)
			L, E, U, P = leup(A)
			self.assertTrue((L @ E @ U @ P == A).all())

			np.random.seed(seed)
			A = random_rational_matrix(8,8,0.2)
			L, E, U, P = leup(A)
			self.assertTrue((L @ E @ U @ P == A).all())
