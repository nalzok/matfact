from matfact.typing import Mat

from .shape_commute import reduce_boundary_maps
from .barcode import Barcode, extract_barcode


def analyze(weights: list[Mat]) -> list[Barcode]:
    _, reduced, _, _ = reduce_boundary_maps(weights)
    barcodes = extract_barcode(reduced)
    return barcodes
