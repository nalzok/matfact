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
        loss, weights = train(reflector, 4, (4, 4, 4), 8192, True)
        self.assertLessEqual(loss, 0.02)

        barcodes = analyze(weights)

        expected = [
            Bar(
                birth=0,
                trace=[0, 0, 0, 0],
                coefs=[1.7260347248306327, 0.8624790787100541, 0.624169111251831],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[-0.06087884388832704, -2.609149617947658, 0.7064605246507004],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[-4.740531485766412, 0.37911654157765245, 1.735747512987149],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[-3.532434760402335, -0.42226451255246017, 2.0611090706002444],
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
        loss, weights = train(rank_one, 4, (4, 4, 4), 8192, True)
        self.assertLessEqual(loss, 0.02)

        barcodes = analyze(weights)

        expected = [
            Bar(
                birth=0,
                trace=[0, 0, 0, 0],
                coefs=[-0.586055744973576, 1.9457537944135384, -0.0814266949892044],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[1.478229088025529e-07, -0.09736715701861182, 2.511399083768538],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[
                    0.22623033431384743,
                    3.268646127985164e-06,
                    0.012511555379481565,
                ],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[
                    0.36736763050557814,
                    0.014307914059902067,
                    5.395524952267472e-06,
                ],
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
