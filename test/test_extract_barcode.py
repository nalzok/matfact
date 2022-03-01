import unittest

import numpy as np

from matfact.shape_commute import reduce_boundary_maps
from matfact.barcode import Barcode, extract_barcode
from matfact.utils import random_rational_matrix
from matfact.typing import EchelonMat


class TestExtractBarcode(unittest.TestCase):
    def test_staged(self) -> None:
        reduced = [
            EchelonMat(np.array([[0, 1, 0], [1, 0, 0]])),
            EchelonMat(np.array([[0, 0, 0], [0, 2, 0], [3, 0, 0]])),
        ]
        barcodes = extract_barcode(reduced)

        # TODO: Check if this is correct
        #
        #   2         1         0
        #
        # e_0 <   . e_0     . e_0
        #      ...         .
        # e_1 <   . e_1 <.... e_1
        #                .
        #           e_2 <     e_2
        #
        # NOTE: order of barcode does not matter
        self.assertEqual(
            barcodes,
            [
                Barcode(birth=0, trace=[2], coefs=[]),
                Barcode(birth=0, trace=[0, 2], coefs=[3]),
                Barcode(birth=0, trace=[1, 1, 0], coefs=[2, 1]),
                Barcode(birth=1, trace=[0, 1], coefs=[1]),
            ],
        )

    def test_random(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(5):
            dimensions = rng.integers(1, 10, 8)
            boundary_maps = []
            for m, n in zip(dimensions, dimensions[1:]):
                boundary_maps.append(random_rational_matrix(rng, m, n, 0.2))

            _, reduced, _, _ = reduce_boundary_maps(boundary_maps)
            # only check if there is any exception
            _ = extract_barcode(reduced)
