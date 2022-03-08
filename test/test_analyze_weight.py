import unittest
import pytest

from matfact.barcode import Barcode
from matfact.analyze_weight import analyze


class TestAnalyzeWeight(unittest.TestCase):
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_analyze_lnn(self) -> None:
        # I am forced to move the import here to capture the DeprecationWarning
        # FIXME: find a better solution
        from lnn.train import train

        loss, weights = train()
        self.assertLessEqual(loss, 3)  # FIXME: find a sensible threshold

        barcodes = analyze(weights)

        print(barcodes)

        self.assertEqual(
            barcodes,
            [
                Barcode(
                    birth=0,
                    trace=[0, 0, 0, 0],
                    coefs=[
                        -2.9596298609309732,
                        -1.2075536437054823,
                        0.06035567820072174,
                    ],
                ),
                Barcode(
                    birth=0,
                    trace=[1, 1, 1, 1],
                    coefs=[
                        -11.465535969187812,
                        -0.03447097593000098,
                        -0.5075821074647223,
                    ],
                ),
                Barcode(
                    birth=0,
                    trace=[2, 2, 2, 2],
                    coefs=[-0.00572473391950723, -9.423974351077364, 0.9757686313489],
                ),
                Barcode(
                    birth=0,
                    trace=[3, 3, 3, 3],
                    coefs=[
                        -1468.5034898056006,
                        -0.0011456084461221039,
                        0.9918906811696515,
                    ],
                ),
                Barcode(
                    birth=0,
                    trace=[4, 4, 4, 4],
                    coefs=[
                        -0.008902776687676806,
                        116.81907559531624,
                        0.8270490877888004,
                    ],
                ),
                Barcode(
                    birth=0,
                    trace=[5, 5, 5, 5],
                    coefs=[
                        0.09947818904668956,
                        3.7245767903727938,
                        -0.5257652640121336,
                    ],
                ),
                Barcode(
                    birth=0,
                    trace=[6, 6, 6, 6],
                    coefs=[
                        0.16698322760485287,
                        0.11233834547496035,
                        -0.9871734209970051,
                    ],
                ),
                Barcode(
                    birth=0,
                    trace=[7, 7, 7, 7],
                    coefs=[
                        -0.17722290865747997,
                        -1.4685007434344624,
                        0.9691187897492604,
                    ],
                ),
            ],
        )
