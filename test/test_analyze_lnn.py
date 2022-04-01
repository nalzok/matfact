import unittest
import pytest

import numpy as np

from matfact.barcode import Bar
from matfact.analyze_weights import analyze
from lnn.train import train
from lnn.ground_truth import reflector, rank_one


class TestAnalyzeWeight(unittest.TestCase):
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_analyze_lnn_full_rank(self) -> None:
        loss, weights = train(reflector, [4, 4, 4], 8192, True)
        self.assertLessEqual(loss, 0.02)

        barcodes = analyze(weights)

        expected = [
            Bar(
                birth=0,
                trace=[0, 0, 0, 0],
                coefs=[0.8872060680376328, 2.8206796585519216, 0.37110570073127747],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[-0.14907615733109542, -0.615705843267983, 1.1905652749889464],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[2.8059388431841543, -0.7519656475865628, 1.5190546905074143],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[-2.8693334521040432, -0.5807573998700479, 1.8419283121784416],
            ),
        ]

        self.assertEqual(
            [(barcode.birth, barcode.trace) for barcode in barcodes],
            [(barcode.birth, barcode.trace) for barcode in expected],
        )

        for b1, b2 in zip(barcodes, expected):
            self.assertTrue(
                (np.array(b1.coefs).round(1) == np.array(b2.coefs).round(1)).all()
            )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_analyze_lnn_rank_one(self) -> None:
        loss, weights = train(rank_one, [4, 4, 4], 8192, True)
        self.assertLessEqual(loss, 0.02)

        barcodes = analyze(weights)

        expected = [
            Bar(
                birth=0,
                trace=[0, 0, 0, 0],
                coefs=[0.17291343249145688, 1.9554653503881327, 0.2752838730812073],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[
                    0.00028599331487311197,
                    15.885820397224023,
                    0.049065108671815405,
                ],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[-0.06587927215414519, 0.00028046108289991184, 9.979095551252685],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[-0.0006948071053359955, -0.003382645961549524, 2.02554630175558],
            ),
        ]

        self.assertEqual(
            [(barcode.birth, barcode.trace) for barcode in barcodes],
            [(barcode.birth, barcode.trace) for barcode in expected],
        )

        for b1, b2 in zip(barcodes, expected):
            self.assertTrue(
                (np.array(b1.coefs).round(1) == np.array(b2.coefs).round(1)).all()
            )
