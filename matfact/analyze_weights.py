from typing import Sequence, List

from .typing import Mat
from .shape_commute import reduce_boundary_maps
from .barcode import Bar, extract_barcode


def analyze(weights: Sequence[Mat]) -> List[Bar]:
    _, reduced, _, _ = reduce_boundary_maps(weights)
    barcodes = extract_barcode(reduced)
    return barcodes
