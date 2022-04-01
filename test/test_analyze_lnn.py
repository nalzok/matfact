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
        loss, weights = train(reflector, [4, 4, 4], 8192)
        self.assertLessEqual(loss, 0.02)

        barcodes = analyze(weights)

        expected = [
            Bar(
                birth=0,
                trace=[0, 0, 0, 0],
                coefs=[0.8867233609221626, 2.8300896445295645, 0.37028828263282776],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[-0.1491364440092249, -0.6179503749319515, 1.1909053426842202],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[2.797325762212711, -0.7491768395528192, 1.5205921013003083],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[-2.8809538403416566, -0.5794136911250758, 1.8455965459097048],
            ),
        ]

        self.assertEqual(
            [(barcode.birth, barcode.trace) for barcode in barcodes],
            [(barcode.birth, barcode.trace) for barcode in expected],
        )

        for b1, b2 in zip(barcodes, expected):
            self.assertTrue(
                (np.array(b1.coefs).round(3) == np.array(b2.coefs).round(3)).all()
            )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_analyze_lnn_rank_one(self) -> None:
        loss, weights = train(rank_one, [4, 4, 4], 8192)
        self.assertLessEqual(loss, 0.02)

        barcodes = analyze(weights)

        expected = [
            Bar(
                birth=0,
                trace=[0, 0, 0, 0],
                coefs=[0.17496910863256113, 1.9590297275423494, 0.2753755748271942],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[0.001618242114158619, 15.73688814920096, 0.04980772431578018],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[-0.04427404053838152, 0.0002770990702196281, 9.856836314367182],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[
                    -1.512531801060868e-05,
                    -0.0002605785416602488,
                    2.018028900351889,
                ],
            ),
        ]

        self.assertEqual(
            [(barcode.birth, barcode.trace) for barcode in barcodes],
            [(barcode.birth, barcode.trace) for barcode in expected],
        )

        for b1, b2 in zip(barcodes, expected):
            self.assertTrue(
                (np.array(b1.coefs).round(2) == np.array(b2.coefs).round(2)).all()
            )
