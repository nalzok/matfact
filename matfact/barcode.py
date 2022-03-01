import numpy as np

from .typing import EchelonMat


def extract_barcode(reduced: list[EchelonMat]) -> list[tuple[int, list[int]]]:
    barcodes = []

    bases: list[tuple[int, list[int]]] = []
    for _ in range(reduced[-1].shape[1]):
        # born at index 0, with empty trace
        bases.append((0, []))

    for i, E in enumerate(reduced[::-1]):
        next_bases: list[tuple[int, list[int]]] = []

        # Assuming the direction is $V_{i-1} -> V_i$
        for j, row in enumerate(E):
            nonzero = np.flatnonzero(row)
            if nonzero.size == 0:
                # birth
                next_bases.append((i + 1, []))
            else:
                # persist
                birth, trace = bases[nonzero[0]]
                trace.append(j)
                next_bases.append((birth, trace))

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
