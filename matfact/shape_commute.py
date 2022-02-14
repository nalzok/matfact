from fractions import Fraction
import numpy as np


def EL_pseudoinverse(EL):
    """
    Return left pseudoinverse
    """
    EL2 = EL.copy().T
    for i in range(EL2.shape[0]):
        for j in range(EL2.shape[1]):
            if EL2[i, j] != 0:
                EL2[i, j] = 1 / EL2[i, j]

    return EL2


def EL_L_commute(EL, L):
    return EL @ L @ EL_pseudoinverse(EL)


EL = np.array([[2, 0], [0, 3]])
L = np.array([[1, 0], [2, 1]])
L2 = EL_L_commute(EL, L)
print(L2 @ EL - EL @ L)

EL = np.array([[0, 0], [2, 0]])
L = np.array([[1, 0], [2, 1]])
L2 = EL_L_commute(EL, L)
print(L2 @ EL - EL @ L)
