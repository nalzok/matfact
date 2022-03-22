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
        loss, weights = train(reflector)
        self.assertLessEqual(loss, 3)  # FIXME: find a sensible threshold

        barcodes = analyze(weights)

        expected = [
            Bar(
                birth=0,
                trace=[0, 0, 0, 0],
                coefs=[
                    -2.9596298609309732,
                    -1.2075536437054823,
                    0.06035567820072174,
                ],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[
                    -11.465535969187812,
                    -0.03447097593000098,
                    -0.5075821074647223,
                ],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[-0.00572473391950723, -9.423974351077364, 0.9757686313489],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[
                    -1468.5034898056006,
                    -0.0011456084461221039,
                    0.9918906811696515,
                ],
            ),
            Bar(
                birth=0,
                trace=[4, 4, 4, 4],
                coefs=[
                    -0.008902776687676806,
                    116.81907559531624,
                    0.8270490877888004,
                ],
            ),
            Bar(
                birth=0,
                trace=[5, 5, 5, 5],
                coefs=[
                    0.09947818904668956,
                    3.7245767903727938,
                    -0.5257652640121336,
                ],
            ),
            Bar(
                birth=0,
                trace=[6, 6, 6, 6],
                coefs=[
                    0.16698322760485287,
                    0.11233834547496035,
                    -0.9871734209970051,
                ],
            ),
            Bar(
                birth=0,
                trace=[7, 7, 7, 7],
                coefs=[
                    -0.17722290865747997,
                    -1.4685007434344624,
                    0.9691187897492604,
                ],
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
        loss, weights = train(rank_one)
        # self.assertLessEqual(loss, 3)

        barcodes = analyze(weights)

        expected = [
            Bar(
                birth=0,
                trace=[0, 0, 0, 0],
                coefs=[-1.3012901480198842, -2.1360919624006742, 0.04334279149770737],
            ),
            Bar(
                birth=0,
                trace=[1, 1, 1, 1],
                coefs=[8.063680412047145, 0.07977869260870543, -0.6083749712205598],
            ),
            Bar(
                birth=0,
                trace=[2, 2, 2, 2],
                coefs=[-0.0563640830301958, 1.781325664397461, 1.4695128914595554],
            ),
            Bar(
                birth=0,
                trace=[3, 3, 3, 3],
                coefs=[14.505224751771648, -0.023521371221015164, 1.1720102535767072],
            ),
            Bar(
                birth=0,
                trace=[4, 4, 4, 4],
                coefs=[-0.10725728552549008, 20.56986361560095, 0.7097573873398205],
            ),
            Bar(
                birth=0,
                trace=[5, 5, 5, 5],
                coefs=[0.1407772231532048, 2.5423912055339883, -0.4656382749436836],
            ),
            Bar(
                birth=0,
                trace=[6, 6, 6, 6],
                coefs=[0.33416787263388065, 0.08335094617226416, -1.4572669558938975],
            ),
            Bar(
                birth=0,
                trace=[7, 7, 7, 7],
                coefs=[-0.16529278908364814, -0.5225493638434295, 0.582448814597611],
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
