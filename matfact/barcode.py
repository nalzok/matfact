from dataclasses import dataclass

import numpy as np

from .typing import EchelonMat


@dataclass
class Barcode:
    birth: int
    trace: list[int]

    def __len__(self) -> int:
        return len(self.trace) - 1


def extract_barcode(reduced: list[EchelonMat]) -> list[Barcode]:
    barcodes = []

    bases: list[Barcode] = []
    for basis in range(reduced[-1].shape[1]):
        # born at index 0, with empty trace
        bases.append(Barcode(0, [basis]))

    for i, E in enumerate(reduced[::-1]):
        next_bases: list[Barcode] = []

        # Assuming the direction is $V_{i-1} -> V_i$
        for basis, row in enumerate(E):
            nonzero = np.flatnonzero(row)
            assert nonzero.size <= 1

            if nonzero.size == 0:
                # birth
                next_bases.append(Barcode(i + 1, [basis]))
            else:
                # persist
                barcode = bases[nonzero[0]]
                barcode.trace.append(basis)
                next_bases.append(barcode)

        m, n = E.shape
        assert len(next_bases) == m
        assert len(bases) == n

        for j, col in enumerate(E.T):
            nonzero = np.flatnonzero(col)
            if nonzero.size == 0:
                # death
                barcodes.append(bases[j])

        bases = next_bases

    # Add bases which persist until now
    barcodes.extend(bases)

    return barcodes
