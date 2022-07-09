from typing import Sequence, List
from dataclasses import dataclass

import numpy as np

from .typing import EchelonMat


@dataclass
class Bar:
    birth: int
    trace: List[int]
    coefs: List[float]

    def __len__(self) -> int:
        return len(self.coefs)


def extract_barcode(reduced: Sequence[EchelonMat]) -> List[Bar]:
    barcode = []

    bases: List[Bar] = []
    for basis in range(reduced[-1].shape[1]):
        # born at index 0, with empty trace
        bases.append(Bar(0, [basis], []))

    for i, E in enumerate(reduced[::-1]):
        next_bases: List[Bar] = []

        # Assuming the direction is $V_{i-1} -> V_i$
        for basis, row in enumerate(E):
            nonzero = np.flatnonzero(row)
            assert nonzero.size <= 1

            if nonzero.size == 0:
                # birth
                next_bases.append(Bar(i + 1, [basis], []))
            else:
                # persist
                bar = bases[nonzero[0]]
                bar.trace.append(basis)
                bar.coefs.append(row[nonzero[0]])
                next_bases.append(bar)

        m, n = E.shape
        assert len(next_bases) == m
        assert len(bases) == n

        for j, col in enumerate(E.T):
            nonzero = np.flatnonzero(col)
            if nonzero.size == 0:
                # death
                barcode.append(bases[j])

        bases = next_bases

    # Add bases which persist until now
    barcode.extend(bases)

    return barcode
