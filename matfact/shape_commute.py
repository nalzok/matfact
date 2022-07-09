from typing import Sequence, Tuple, List
import numpy as np

from .leup import leup
from .typing import Mat, LowerTriMat, EchelonMat, UpperTriMat, PermMat


def EL_pseudoinverse(EL: EchelonMat) -> EchelonMat:
    """
    Return left pseudoinverse
    """
    EL2 = EL.copy().T
    for i in range(EL2.shape[0]):
        for j in range(EL2.shape[1]):
            if EL2[i, j] != 0:
                EL2[i, j] = 1 / EL2[i, j]

    return EchelonMat(EL2)


def EL_L_commute(EL: EchelonMat, L: LowerTriMat) -> EchelonMat:
    return EchelonMat(EL @ L @ EL_pseudoinverse(EL))


def reduce_boundary_maps(
    boundary_maps: Sequence[Mat],
) -> Tuple[LowerTriMat, List[EchelonMat], UpperTriMat, PermMat]:
    # TODO: rename argument; they are not boundary maps
    # TODO: track change of basis
    lower = []
    reduced = []
    U = UpperTriMat(np.eye(boundary_maps[0].shape[0]))
    P = PermMat(np.eye(boundary_maps[0].shape[0]))
    for A in boundary_maps:
        L, E, U, P = leup(U @ P @ A)
        lower.append(L)
        reduced.append(E)

    carry = LowerTriMat(np.eye(boundary_maps[-1].shape[1]))
    for i in reversed(range(len(boundary_maps))):
        L, E = lower[i], reduced[i]
        L2 = EL_L_commute(E, carry)
        carry = L @ L2

    return carry, reduced, U, P
