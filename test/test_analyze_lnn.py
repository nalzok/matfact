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
                coefs=[1.0446929543993873, 2.9931136600124, 0.2969158887863159],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[-0.12699876916965444, -0.7892851333178763, 1.0976528956825788],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[2.604514776600189, -0.7304937906878862, 1.66439616526737],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[-2.9751976941785596, -0.5480573905866662, 1.8835965995810589],
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
                coefs=[0.4572122456923816, 1.3394106067017428, 0.1521281898021698],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[0.0006235980319857504, -8.409277194418932, -0.05629052126870254],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[
                    -0.06251503753027915,
                    0.00040807728213040906,
                    -4.2531037652089205,
                ],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[-4.214603103848413e-05, -0.06730102356388637, 0.653095891416092],
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
