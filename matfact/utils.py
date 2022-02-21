from fractions import Fraction

import numpy as np
import numpy.typing as npt

from .typing import Mat


def random_rational(rng: np.random.Generator) -> Fraction:
    x = rng.random()
    return Fraction(x)


def random_rational_matrix(rng: np.random.Generator, m: int, n: int, p: float = 0.1) -> Mat:
    """
    m x n rational matrix
    every entry is non-zero with probability p
    """
    A: npt.NDArray[np.object_] = np.zeros((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            if np.random.rand() < p:
                A[i, j] = random_rational(rng)

    return A
